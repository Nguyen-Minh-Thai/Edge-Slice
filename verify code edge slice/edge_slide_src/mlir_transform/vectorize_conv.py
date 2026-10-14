"""Outline stride-1 conv2d tiles into vectorizable helper functions.

``--convert-linalg-to-loops`` (used for the whole payload in ``lowering.py``)
produces pure scalar loops — MLIR 19's Transform-dialect vectorizer refuses
2D convolutions entirely (see docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md). The
legacy ``--affine-super-vectorize`` pass *can* vectorize a conv's affine loop
nest, but only two conditions hold at once:

1. The loop nest must not be nested inside an ``scf.for`` — EdgeSlice's own
   tiling produces exactly that nesting (``scf.forall`` lowered to
   ``scf.for``), so the pass silently skips every tiled conv with 0 vector
   ops and no diagnostic.
2. Every memref the loop touches needs an identity (non-strided) layout —
   subview/function-boundary memrefs carry a non-trivial ``strided<...>``
   layout and the pass hard-refuses with "NYI: non-trivial layout map".

Both are satisfied by outlining the conv into its own top-level function that
copies its operands into local, identity-layout ``memref.alloc``s first.

**Correctness-critical restriction: stride 1 only.** The legacy vectorizer
does not detect non-unit-stride affine access patterns — verified directly by
JIT-executing a stride-2 conv through this same pass: it produces `vector<>`
ops with no error, but the numeric result is silently wrong (max abs diff
~47 against a NumPy reference, vs. exact match for stride 1). This module
must never outline a stride>1 (in either spatial dimension) conv. Ops that
don't match — wrong op name, unparsable attrs/types, stride != 1 — are left
untouched and continue through the existing scalar
``--convert-linalg-to-loops`` path; a missed match only forfeits the speedup,
never correctness.

The outlined loop nest is *hand-authored* directly in the affine dialect
(never routed through ``--convert-linalg-to-affine-loops``), so nothing else
in the payload is ever exposed to that pass — every op this module doesn't
touch keeps going through the exact same pipeline stages it always has.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CONV_OP_RE = re.compile(
    r"linalg\.conv_2d_nchw_fchw\s*"
    r"\{(?P<attrs>[^{}]*)\}\s*"
    r"ins\((?P<ins>[^()]*)\)\s*"
    r"outs\((?P<outs>[^()]*)\)"
)
_STRIDES_RE = re.compile(r"strides\s*=\s*dense<([^>]+)>")
_DILATIONS_RE = re.compile(r"dilations\s*=\s*dense<([^>]+)>")
_FUNC_HEADER_RE = re.compile(r"^(\s*)func\.func\s+@(\w+)\(", re.MULTILINE)


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split ``text`` on ``sep`` at bracket-depth 0 only.

    MLIR type lists mix the separator with nested ``<...>``/``[...]``
    delimiters (e.g. ``memref<4x4xf32, strided<[..], offset: ?>>``), so a
    naive ``str.split`` would cut a single type in half.
    """
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closers = set(pairs.values())
    depth = 0
    parts: List[str] = []
    current: List[str] = []
    for ch in text:
        if ch in pairs:
            depth += 1
        elif ch in closers:
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return [p.strip() for p in parts]


def _parse_dense_pair(raw: str) -> Optional[Tuple[int, int]]:
    """Parse a ``dense<1>`` or ``dense<[1, 2]>`` attribute body to (h, w)."""
    raw = raw.strip()
    try:
        if raw.startswith("["):
            values = [int(v.strip()) for v in raw.strip("[]").split(",")]
            if len(values) == 1:
                return values[0], values[0]
            if len(values) >= 2:
                return values[0], values[1]
            return None
        value = int(raw)
        return value, value
    except ValueError:
        return None


def _memref_shape_and_dtype(memref_type: str) -> Optional[Tuple[List[int], str]]:
    """Extract ``([dims], dtype)`` from a memref type, ignoring any layout.

    The character class must exclude ``>`` as well as ``,``. With ``[^,]+`` it
    was greedy over the closing bracket, so ``memref<1x8x34x34xf32>`` yielded a
    dtype of ``f32>`` — and every type this module then built came out as
    ``memref<1x8x34x34xf32>>`` with a doubled bracket, alongside
    ``arith.mulf %a, %b : f32>``. mlir-opt rejected the whole helper module
    with "expected operation name in quotes", `_lower_with_vectorization_attempt`
    caught that and fell back to the scalar pipeline, and the warning it logged
    read as "this payload shape is not vectorizable yet" rather than "the IR
    this module emits does not parse". The result was a feature that reported
    itself as working-but-unlucky on every model, and had never emitted valid
    IR at all.
    """
    match = re.match(r"memref<([^,>]+)", memref_type)
    if not match:
        return None
    body = match.group(1)
    parts = body.split("x")
    if len(parts) < 2:
        return None
    dtype = parts[-1]
    try:
        dims = [int(p) for p in parts[:-1]]
    except ValueError:
        return None
    return dims, dtype


def _identity_memref(dims: List[int], dtype: str) -> str:
    return "memref<" + "x".join(str(d) for d in dims) + f"x{dtype}>"


@dataclass
class _ConvMatch:
    span: Tuple[int, int]
    op_id: str
    in_name: str
    in_type: str
    filt_name: str
    filt_type: str
    out_name: str
    out_type: str
    dilation_h: int
    dilation_w: int


def _find_stride1_convs(mlir_text: str) -> List[_ConvMatch]:
    matches: List[_ConvMatch] = []
    for m in _CONV_OP_RE.finditer(mlir_text):
        attrs = m.group("attrs")
        strides_m = _STRIDES_RE.search(attrs)
        if strides_m is None:
            continue  # linalg default is stride 1, but require it explicit
        strides = _parse_dense_pair(strides_m.group(1))
        if strides != (1, 1):
            continue

        dilations_m = _DILATIONS_RE.search(attrs)
        dilations = _parse_dense_pair(dilations_m.group(1)) if dilations_m else (1, 1)
        if dilations is None:
            continue

        # `ins`/`outs` are "%name0, %name1 : type0, type1" — the *first*
        # top-level `:` is always the names/types separator (SSA names never
        # contain a colon; a later colon like `offset: ?` inside a type is
        # nested inside `<...>`, never at depth 0), so partition once on the
        # raw text, then split each half on its own top-level commas.
        names_part, sep, types_part = m.group("ins").partition(":")
        if not sep:
            continue
        in_names = _split_top_level(names_part, sep=",")
        in_types = _split_top_level(types_part, sep=",")
        if len(in_names) != 2 or len(in_types) != 2:
            continue
        out_name, out_sep, out_type_raw = m.group("outs").partition(":")
        if not out_sep:
            continue
        out_name = out_name.strip()
        out_type = out_type_raw.strip()

        op_id_m = re.search(r'edgeslice\.op_id\s*=\s*"([^"]+)"', attrs)
        op_id = op_id_m.group(1) if op_id_m else f"anon_{m.start()}"

        matches.append(
            _ConvMatch(
                span=m.span(),
                op_id=op_id,
                in_name=in_names[0].strip(),
                in_type=in_types[0].strip(),
                filt_name=in_names[1].strip(),
                filt_type=in_types[1].strip(),
                out_name=out_name,
                out_type=out_type,
                dilation_h=dilations[0],
                dilation_w=dilations[1],
            )
        )
    return matches


def _build_helper_function(name: str, match: _ConvMatch) -> Optional[str]:
    in_shape = _memref_shape_and_dtype(match.in_type)
    filt_shape = _memref_shape_and_dtype(match.filt_type)
    out_shape = _memref_shape_and_dtype(match.out_type)
    if in_shape is None or filt_shape is None or out_shape is None:
        return None
    (in_dims, dtype), (filt_dims, _), (out_dims, _) = in_shape, filt_shape, out_shape
    if len(in_dims) != 4 or len(filt_dims) != 4 or len(out_dims) != 4:
        return None

    in_local_ty = _identity_memref(in_dims, dtype)
    filt_local_ty = _identity_memref(filt_dims, dtype)
    out_local_ty = _identity_memref(out_dims, dtype)

    n, out_c, out_h, out_w = out_dims
    _, in_c, _, _ = in_dims
    _, _, kh, kw = filt_dims

    return f"""\
    func.func private @{name}(%vec_in: {match.in_type}, %vec_filt: {match.filt_type}, %vec_out: {match.out_type}) {{
      %vec_in_local = memref.alloc() : {in_local_ty}
      %vec_filt_local = memref.alloc() : {filt_local_ty}
      %vec_out_local = memref.alloc() : {out_local_ty}
      memref.copy %vec_in, %vec_in_local : {match.in_type} to {in_local_ty}
      memref.copy %vec_filt, %vec_filt_local : {match.filt_type} to {filt_local_ty}
      memref.copy %vec_out, %vec_out_local : {match.out_type} to {out_local_ty}
      affine.for %vn = 0 to {n} {{
        affine.for %voc = 0 to {out_c} {{
          affine.for %voh = 0 to {out_h} {{
            affine.for %vow = 0 to {out_w} {{
              affine.for %vic = 0 to {in_c} {{
                affine.for %vkh = 0 to {kh} {{
                  affine.for %vkw = 0 to {kw} {{
                    %vih = affine.apply affine_map<(d0, d1) -> (d0 + d1 * {match.dilation_h})>(%voh, %vkh)
                    %viw = affine.apply affine_map<(d0, d1) -> (d0 + d1 * {match.dilation_w})>(%vow, %vkw)
                    %va = affine.load %vec_in_local[%vn, %vic, %vih, %viw] : {in_local_ty}
                    %vb = affine.load %vec_filt_local[%voc, %vic, %vkh, %vkw] : {filt_local_ty}
                    %vc = affine.load %vec_out_local[%vn, %voc, %voh, %vow] : {out_local_ty}
                    %vp = arith.mulf %va, %vb : {dtype}
                    %vs = arith.addf %vc, %vp : {dtype}
                    affine.store %vs, %vec_out_local[%vn, %voc, %voh, %vow] : {out_local_ty}
                  }}
                }}
              }}
            }}
          }}
        }}
      }}
      memref.copy %vec_out_local, %vec_out : {out_local_ty} to {match.out_type}
      return
    }}
"""


def outline_stride1_convs(mlir_text: str) -> Tuple[str, int]:
    """Outline every stride-1 ``linalg.conv_2d_nchw_fchw`` into a vectorizable helper.

    Returns ``(transformed_text, count)``. On any parse failure for a given
    op, that op is left untouched (falls back to the existing scalar path) —
    a missed match only forfeits the speedup, never correctness. Only
    applies to a payload that has already been bufferized (memref, not
    tensor, operands) — call this after ``--normalize-memrefs`` /
    ``--scf-forall-to-for`` and before ``--convert-linalg-to-loops``.
    """
    matches = _find_stride1_convs(mlir_text)
    if not matches:
        return mlir_text, 0

    helpers: List[str] = []
    replacements: List[Tuple[Tuple[int, int], str]] = []
    skipped: List[str] = []
    for index, match in enumerate(matches):
        helper_name = f"__edgeslide_vec_conv_{match.op_id}_{index}"
        helper_src = _build_helper_function(helper_name, match)
        if helper_src is None:
            # Unsupported operand shape (dynamic extent, non-rank-4). Recorded
            # rather than dropped: the count alone cannot distinguish "nothing
            # matched" from "everything matched and every helper failed", and
            # the latter is what a generator bug looks like from outside.
            skipped.append(f"{match.op_id}({match.in_type})")
            continue
        helpers.append(helper_src)
        call_text = (
            f"func.call @{helper_name}({match.in_name}, {match.filt_name}, {match.out_name}) "
            f": ({match.in_type}, {match.filt_type}, {match.out_type}) -> ()"
        )
        replacements.append((match.span, call_text))

    if skipped:
        logger.info(
            "outline_stride1_convs: %d of %d matched conv(s) had unsupported "
            "operand shapes and were left scalar: %s",
            len(skipped),
            len(matches),
            ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""),
        )
    if not helpers:
        logger.warning(
            "outline_stride1_convs matched %d stride-1 conv(s) but could not "
            "build a helper for any of them, so nothing will be vectorized. "
            "Matching without building is a generator problem, not a payload "
            "one — check _build_helper_function's shape preconditions.",
            len(matches),
        )
        return mlir_text, 0

    # Replace matched ops with calls, back to front so earlier spans stay valid.
    new_text = mlir_text
    for span, call_text in sorted(replacements, key=lambda item: item[0][0], reverse=True):
        new_text = new_text[: span[0]] + call_text + new_text[span[1] :]

    header_match = _FUNC_HEADER_RE.search(new_text)
    if header_match is None:
        return mlir_text, 0
    insert_at = header_match.start()
    new_text = new_text[:insert_at] + "".join(helpers) + "\n" + new_text[insert_at:]

    return new_text, len(helpers)
