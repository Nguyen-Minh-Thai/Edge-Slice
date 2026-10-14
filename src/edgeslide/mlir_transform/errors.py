"""Typed errors and structured stage results for the EdgeSlide MLIR pipeline.

Every stage that shells out to an external tool reports failure through one of
these exceptions rather than a printed warning plus ``None``. A stage that
returns ``None`` on failure is indistinguishable from a stage that legitimately
had nothing to produce, and the pipeline then reports success for a compile that
never happened.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


class EdgeSlideError(Exception):
    """Base class for every EdgeSlide compiler failure."""


class ToolNotFoundError(EdgeSlideError, FileNotFoundError):
    """A required external binary (mlir-opt, mlir-translate, clang) is missing.

    Inherits ``FileNotFoundError`` so existing ``except FileNotFoundError``
    handlers keep working while new code can catch the precise type.
    """

    def __init__(
        self,
        tool: str,
        *,
        env_var: Optional[str] = None,
        searched: Sequence[str] = (),
    ):
        self.tool = tool
        self.env_var = env_var or tool.upper().replace("-", "_")
        self.searched = list(searched)
        message = (
            f"{tool} not found. Install the LLVM/MLIR toolchain or set "
            f"${self.env_var} to its absolute path."
        )
        if self.searched:
            message += " Searched: " + ", ".join(self.searched)
        super().__init__(message)


class ExternalToolError(EdgeSlideError, RuntimeError):
    """An external tool ran and failed.

    Carries the full forensic record — command, return code, stdout, stderr —
    so the caller never has to re-run the tool by hand to see why it failed.
    Inherits ``RuntimeError`` for backward compatibility with callers that
    already catch it.
    """

    #: Human-readable stage label used in the message; overridden per subclass.
    stage = "external tool"

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
        *,
        stage: Optional[str] = None,
    ):
        self.command = [str(part) for part in command]
        self.returncode = returncode
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        if stage is not None:
            self.stage = stage
        super().__init__(
            f"{self.stage} failed (exit {returncode})\n"
            f"  command: {' '.join(self.command)}\n"
            f"  stdout: {self.stdout.strip() or '<empty>'}\n"
            f"  stderr: {self.stderr.strip() or '<empty>'}"
        )


class MLIRTransformError(ExternalToolError):
    """mlir-opt failed while transforming or lowering MLIR."""

    stage = "mlir-opt"


class LLVMCompilationError(ExternalToolError):
    """mlir-translate, clang, or cc failed while producing LLVM/native output."""

    stage = "LLVM compilation"


class TransformVerificationError(EdgeSlideError, RuntimeError):
    """The interpreter ran clean but the requested rewrite never materialized.

    A transform script that matches nothing is not an error to mlir-opt: it
    exits 0 and echoes the payload unchanged. Without this check the pipeline
    would hand back an untransformed program as a successful compile.

    Inherits ``RuntimeError`` so callers that already treat a failed transform
    as a runtime failure keep working, matching :class:`ExternalToolError`.
    """


class SilentDowngradeError(EdgeSlideError, RuntimeError):
    """A requested compilation mode is unavailable and degrading it would lie.

    Raised instead of quietly producing a different artifact than the one the
    caller asked for. Two cases exist, and both had shipped as silent
    downgrades:

    - vectorization was requested but the vectorized IR did not compile, so the
      artifact would be scalar under a vectorized label;
    - real buffer deallocation is unavailable, so the artifact would leak every
      buffer and its peak-memory number would measure nothing.

    In each case the fallback still *produced a binary*, which is what made it
    dangerous: the run succeeded, the numbers looked ordinary, and nothing in
    them recorded that the mode had changed. Both downgrades remain reachable
    behind an explicit opt-in on ``SlicedPayloadLoweringConfig``.
    """


class MLIRVectorizationError(SilentDowngradeError):
    """Vectorization was requested and could not be delivered."""


class DeallocationUnavailableError(SilentDowngradeError):
    """Real buffer deallocation is unavailable on this mlir-opt."""


class InvalidTransformStyleError(EdgeSlideError, ValueError):
    """An unknown transform style was requested.

    Silently falling back to ``structured`` for a typo'd style means an ablation
    run labelled ``aggressive`` would report structured numbers.
    """

    def __init__(self, style: Any, valid: Sequence[str]):
        self.style = style
        self.valid = list(valid)
        super().__init__(
            f"Unknown transform style {style!r}. "
            f"Valid styles: {', '.join(sorted(self.valid))}"
        )


@dataclass
class StageResult:
    """Outcome of one pipeline stage.

    ``status`` is ``"ok"``, ``"skipped"``, or ``"failed"``. A stage that failed
    is never reported with an output path, so a caller cannot read a failure as
    a usable artifact.
    """

    name: str
    status: str
    output_path: Optional[str] = None
    command: Optional[list] = None
    diagnostics: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "output_path": self.output_path,
            "command": list(self.command) if self.command else None,
            "diagnostics": self.diagnostics,
            "details": dict(self.details),
        }


def require_tool(
    name: str,
    explicit: Optional[str] = None,
    *,
    extra_prefixes: Sequence[str] = ("/usr/lib/llvm-19/bin", "/usr/bin"),
) -> str:
    """Resolve an external binary or raise :class:`ToolNotFoundError`.

    Resolution order: explicit argument, ``$TOOL_NAME`` env var, ``PATH``, then
    the well-known LLVM prefixes. Checked up front so a missing tool fails
    before any partial artifact is written.
    """
    import os

    if explicit:
        if Path(explicit).is_file():
            return str(explicit)
        raise ToolNotFoundError(name, searched=[str(explicit)])

    env_var = name.upper().replace("-", "_")
    searched = []

    override = os.environ.get(env_var)
    if override:
        searched.append(f"${env_var}={override}")
        if Path(override).is_file():
            return override

    found = shutil.which(name)
    if found:
        return found
    searched.append("PATH")

    for prefix in extra_prefixes:
        candidate = Path(prefix) / name
        searched.append(str(candidate))
        if candidate.is_file():
            return str(candidate)

    raise ToolNotFoundError(name, env_var=env_var, searched=searched)


def require_input_file(path: PathLike, description: str) -> Path:
    """Validate an input path before it is handed to an external tool.

    mlir-opt reports a missing input as a generic parse diagnostic, which reads
    as "your IR is broken" rather than "that file does not exist".
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is not a regular file: {resolved}")
    if resolved.stat().st_size == 0:
        raise ValueError(f"{description} is empty: {resolved}")
    return resolved


def prepare_output_path(path: PathLike, description: str) -> Path:
    """Validate an output path and create its parent directory."""
    resolved = Path(path)
    if resolved.is_dir():
        raise IsADirectoryError(f"{description} is a directory: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def run_tool(
    command: Sequence[str],
    *,
    stage: str,
    error_cls: type = ExternalToolError,
) -> subprocess.CompletedProcess:
    """Run an external tool, raising *error_cls* with full context on failure.

    Uses ``check=False`` and raises explicitly so the exception carries stdout
    and stderr; ``check=True`` would raise ``CalledProcessError`` whose message
    omits the tool diagnostics that actually explain the failure.
    """
    printable = [str(part) for part in command]
    logger.info("%s: %s", stage, " ".join(printable))
    result = subprocess.run(printable, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise error_cls(
            printable, result.returncode, result.stdout, result.stderr, stage=stage
        )
    if result.stderr.strip():
        logger.debug("%s stderr: %s", stage, result.stderr.strip())
    return result


def atomic_write_text(path: PathLike, text: str, *, description: str = "output") -> Path:
    """Write *text* to *path* so a reader never observes a partial file.

    The temp file is created in the destination directory, because
    ``os.replace`` is only atomic within a single filesystem.
    """
    target = prepare_output_path(path, description)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    return target


@contextlib.contextmanager
def atomic_output(
    path: PathLike, *, description: str = "output"
) -> Iterator[Path]:
    """Yield a temp path that is moved onto *path* only if the block succeeds.

    Stages that hand an output path to an external tool cannot write directly
    to the destination: a crashed or non-zero-exit tool leaves a truncated file
    that the next stage happily reads as valid IR. On failure the temp file is
    removed and the destination is left untouched — so a failed run cannot be
    mistaken for a successful one by looking at the directory.
    """
    target = prepare_output_path(path, description)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    if not tmp.exists():
        raise FileNotFoundError(
            f"{description} was not produced at {tmp} (tool reported success)"
        )
    os.replace(tmp, target)
