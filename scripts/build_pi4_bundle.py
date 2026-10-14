#!/usr/bin/env python3
"""Build a self-contained measurement bundle to carry to a Raspberry Pi 4.

The Pi runs no toolchain. Everything is compiled here -- both are aarch64 and
the payload is baseline ARMv8-A -- and linked statically, because this host is
glibc 2.39 while Raspberry Pi OS bookworm is 2.36 and a dynamically linked
binary built here will not start there.

What the bundle answers is the question the evaluation cannot answer on this
host: whether an IR-level effect shows up as wall-clock and resident-memory
difference on the actual target. Both arms of an A/B are built, so the Pi
measures a difference rather than a level. `--axis` picks what "before"/"after"
vary:

  --axis deallocation (default): the Route A dealloc-placement fix, measured
      at 4.1x-7.7x peak heap in the IR and in massif (docs/ROUTE_A_RESULTS.md).
  --axis vectorization: --enable-stride1-vectorization codegen for stride-1
      3x3 convs (vectorize_conv.py) against plain scalar codegen. Only
      cache-miss simulation has measured this before (Phase 4,
      docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md); this is the first real
      wall-clock read. The outliner only reaches dense stride-1 convs, so the
      default model set narrows to resnet18/resnet50, and the default arm
      narrows to untiled -- combined with tiling it has a documented
      footprint blow-up (same doc, "the vectorization workaround is not
      free"), so a tiled/tiled_fused run on this axis needs `--arms` spelled
      out deliberately.

    python scripts/build_pi4_bundle.py --models resnet18,mobilenet_v2 \\
        --resolution 224 --out build/pi4_bundle
    python scripts/build_pi4_bundle.py --axis vectorization \\
        --out build/pi4_bundle_vectorization
    scp -r build/pi4_bundle pi@raspberrypi:~/
    ssh pi@raspberrypi 'cd pi4_bundle && ./run_on_pi.sh'

Then bring `pi4_results.json` back and feed it to
`scripts/report_pi4.py` (or read it directly -- it is one flat record per arm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

TIME_SHIM = r'''#!/usr/bin/env python3
"""Stand-in for `/usr/bin/time -f '%M'` on a Pi with no `time` package installed.

Raspberry Pi OS does not ship `time` by default and installing it needs root,
which this measurement path does not have. The counter itself is still
reachable without the package: `%M` is `getrusage`'s `ru_maxrss`, and
`os.wait4` hands back exactly that struct for the child it reaps, in kilobytes
on Linux -- the same units and the same kernel field.

Only one part of /usr/bin/time's contract matters to run_on_pi.sh: the peak RSS
must be the last line of stderr, digits recoverable by `tail -1 | tr -cd
'0-9'`. The child inherits this process's stdout and stderr, so the benchmark's
`mlir_bench_checksum=` line still reaches the caller's command substitution and
the child's own diagnostics still land in rss.tmp above the RSS line.

Measuring with wait4 rather than RUSAGE_CHILDREN keeps the number scoped to
this one child, and this interpreter's own footprint is never counted -- the
child is a separate process.
"""

import os
import resource  # noqa: F401  -- documents that ru_maxrss is the measured field
import sys


def main(argv):
    # Accept and discard the format flag so the call site reads the same as the
    # /usr/bin/time invocation it replaces. The format is fixed at '%M' here.
    args = list(argv)
    while args and args[0].startswith("-"):
        flag = args.pop(0)
        if flag in ("-f", "--format", "-o", "--output") and args:
            args.pop(0)
    if not args:
        sys.stderr.write("time_shim: no command given\n")
        return 127

    pid = os.fork()
    if pid == 0:
        try:
            os.execvp(args[0], args)
        except OSError as exc:
            sys.stderr.write("time_shim: %s: %s\n" % (args[0], exc))
        os._exit(127)

    _, status, usage = os.wait4(pid, 0)

    # Last line of stderr, digits only -- what run_on_pi.sh greps for.
    sys.stderr.write("%d\n" % usage.ru_maxrss)
    sys.stderr.flush()

    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status)
    return os.WEXITSTATUS(status)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'''

RUN_ON_PI = r"""#!/usr/bin/env bash
# Runs on the Pi. Needs bash, python3, the binaries here, and GNU time --
# falling back to the bundled time_shim.py where `time` is not installed.
#
# Trials are interleaved round-robin across binaries rather than run
# binary-by-binary. A Pi 4 throttles under sustained load, so consecutive
# blocks would confound binary order with configuration: whichever ran second
# ran hotter. Round-robin also makes trial index a blocking factor, which is
# what licenses the paired statistic the report uses.
#
# Every trial is appended to a TSV the moment it is measured. A run killed
# halfway -- OOM, thermal shutdown, a dropped ssh session -- then still yields
# everything measured up to that point instead of a truncated JSON.
set -u
cd "$(dirname "$0")"

TRIALS="${TRIALS:-11}"
ITERS="${ITERS:-1}"
SLEEP_MS="${SLEEP_MS:-2000}"
PIN_CPU="${PIN_CPU:-3}"
TSV="${TSV:-pi4_trials.tsv}"
OUT="${OUT:-pi4_results.json}"

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 missing" >&2; exit 1; }

# bash's builtin time cannot report peak RSS of an exited child, so this needs
# GNU time -- or, on a target where the `time` package is absent and there is
# no root to install it, the bundled shim reading the same getrusage field.
if [ -x /usr/bin/time ]; then
  TIME=(/usr/bin/time -f '%M')
elif [ -f time_shim.py ]; then
  TIME=(python3 time_shim.py -f '%M')
  echo "note: /usr/bin/time absent, using bundled time_shim.py (getrusage ru_maxrss)"
else
  echo "ERROR: neither /usr/bin/time nor time_shim.py is available." >&2
  echo "       install GNU time (apt install time) or restore time_shim.py." >&2
  exit 1
fi

PIN=()
if command -v taskset >/dev/null 2>&1; then
  PIN=(taskset -c "$PIN_CPU")
else
  echo "note: taskset missing, runs are not pinned"
fi

temp_now()      { cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo ""; }
throttled_now() { vcgencmd get_throttled 2>/dev/null | sed 's/^throttled=//' || echo ""; }

binaries=()
for b in bin/*; do
  [ -x "$b" ] || continue
  binaries+=("$b")
done
if [ "${#binaries[@]}" -eq 0 ]; then
  echo "ERROR: no executables under bin/" >&2
  exit 1
fi

UNAME="$(uname -srm)"
CPU="$(grep -m1 'model name\|Model' /proc/cpuinfo | cut -d: -f2- | xargs)"
NPROC="$(nproc)"
MEM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
SWAP_KB="$(awk '/SwapTotal/{print $2}' /proc/meminfo)"
GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"

echo "host: $UNAME"
echo "cpu : $CPU"
echo "libc: $(ldd --version 2>/dev/null | head -1)"
echo "mem : $((MEM_KB/1024)) MB   swap: $((SWAP_KB/1024)) MB   governor: $GOV"
echo "binaries: ${#binaries[@]}   trials: $TRIALS   iters/run: $ITERS   pin: cpu$PIN_CPU"
echo "temp at start: $(temp_now)  throttled: $(throttled_now)"
echo

printf 'binary\ttrial\twall_ns\trss_kb\texit_code\ttemp_pre\ttemp_post\tthrottled\titers_reported\tchecksum\n' > "$TSV"

# One discarded sweep: the first run of each binary pays page-in for the whole
# static image, which is not part of what is being compared.
echo "warm-up sweep..."
for b in "${binaries[@]}"; do
  "${PIN[@]+"${PIN[@]}"}" "./$b" "$ITERS" >/dev/null 2>&1
done

for t in $(seq 1 "$TRIALS"); do
  for b in "${binaries[@]}"; do
    name="$(basename "$b")"
    tpre="$(temp_now)"
    t0=$(date +%s%N)
    out=$("${TIME[@]}" "${PIN[@]+"${PIN[@]}"}" "./$b" "$ITERS" 2>rss.tmp)
    rc=$?
    t1=$(date +%s%N)
    tpost="$(temp_now)"
    wall_ns=$((t1 - t0))
    rss="$(tail -1 rss.tmp | tr -cd '0-9')"
    [ -n "$rss" ] || rss=0
    checksum="$(printf '%s' "$out" | sed -n 's/.*mlir_bench_checksum=\([0-9.eE+-]*\).*/\1/p')"
    iters_rep="$(printf '%s' "$out" | sed -n 's/.*iters=\([0-9]*\).*/\1/p')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "$t" "$wall_ns" "$rss" "$rc" "$tpre" "$tpost" "$(throttled_now)" \
      "$iters_rep" "$checksum" >> "$TSV"
    if [ "$rc" -ne 0 ]; then
      echo "  trial $t  $name  EXIT $rc  $(head -2 rss.tmp | tr '\n' ' ')"
    fi
    [ "$SLEEP_MS" -gt 0 ] && sleep "$(awk "BEGIN{print $SLEEP_MS/1000}")"
  done
  echo "trial $t/$TRIALS done  temp=$(temp_now)  throttled=$(throttled_now)"
done
rm -f rss.tmp

echo
echo "temp at end: $(temp_now)  throttled: $(throttled_now)"

TSV="$TSV" OUT="$OUT" TRIALS="$TRIALS" ITERS="$ITERS" \
UNAME="$UNAME" CPU="$CPU" NPROC="$NPROC" MEM_KB="$MEM_KB" SWAP_KB="$SWAP_KB" \
GOV="$GOV" TEMP_END="$(temp_now)" THROT_END="$(throttled_now)" python3 - <<'PYEOF'
import collections
import json
import os


def num(text, cast=float, default=None):
    try:
        return cast(text)
    except (TypeError, ValueError):
        return default


rows = []
with open(os.environ["TSV"]) as handle:
    header = handle.readline().rstrip("\n").split("\t")
    for line in handle:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))

grouped = collections.OrderedDict()
for row in rows:
    grouped.setdefault(row["binary"], []).append(row)

arms = []
for name, trials in grouped.items():
    checksums = [t["checksum"] for t in trials]
    # The last non-empty checksum, so a trailing crashed trial does not erase
    # the value the successful trials agreed on.
    settled = next((c for c in reversed(checksums) if c), "")
    arms.append({
        "binary": name,
        "trial_index": [num(t["trial"], int) for t in trials],
        "wall_ms": [num(t["wall_ns"], int, 0) / 1e6 for t in trials],
        "peak_rss_kb": [num(t["rss_kb"], int, 0) for t in trials],
        "exit_codes": [num(t["exit_code"], int, -1) for t in trials],
        "temp_c_pre": [num(t["temp_pre"], float, 0.0) / 1000.0 for t in trials],
        "temp_c_post": [num(t["temp_post"], float, 0.0) / 1000.0 for t in trials],
        "throttled": [t["throttled"] for t in trials],
        "iters_reported": [num(t["iters_reported"], int, 0) for t in trials],
        "checksums": checksums,
        "checksum": settled,
    })

report = {
    "host": {
        "uname": os.environ["UNAME"],
        "cpu": os.environ["CPU"],
        "nproc": num(os.environ["NPROC"], int, 0),
        "mem_kb": num(os.environ["MEM_KB"], int, 0),
        "swap_kb": num(os.environ["SWAP_KB"], int, 0),
        "governor": os.environ["GOV"],
        "temp_c_end": num(os.environ["TEMP_END"], float, 0.0) / 1000.0,
        "throttled_end": os.environ["THROT_END"],
    },
    # The report refuses a paired test unless this says round_robin, so an
    # older result file cannot be silently run through a paired statistic.
    "schedule": "round_robin",
    "trials": num(os.environ["TRIALS"], int, 0),
    "iters": num(os.environ["ITERS"], int, 1),
    "arms": arms,
}
with open(os.environ["OUT"], "w") as handle:
    json.dump(report, handle, indent=2)
print("wrote %s (%d arms) and %s (%d trials)"
      % (os.environ["OUT"], len(arms), os.environ["TSV"], len(rows)))
PYEOF
"""


def preflight_tools() -> None:
    """Resolve every external tool before anything is written or deleted.

    A missing mlir-opt used to surface once per model, long after the
    destination had been cleared -- which is how a working bundle was lost to
    nothing worse than an unexported PATH. Both MLIR tools and the C compiler
    are resolved up front so that failure costs a second and no artifacts.
    """
    from edgeslide.mlir_transform.errors import require_tool

    for tool in ("mlir-opt", "mlir-translate"):
        resolved = require_tool(tool)
        print(f"  {tool}: {resolved}")
    if shutil.which("cc") is None:
        raise RuntimeError("cc not found; the bundle cannot be linked without it")


def strip_binary(path: Path) -> None:
    """Drop symbols, best effort.

    Static glibc drags in a few MB of symbols per binary that the Pi never
    reads. This must run before the manifest hashes the file, or the recorded
    sha256 stops describing what the Pi actually executes.
    """
    if shutil.which("strip") is None:
        return
    subprocess.run(["strip", "--strip-unneeded", str(path)],
                   check=False, capture_output=True)


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", choices=("deallocation", "vectorization"),
                        default="deallocation",
                        help="what before/after vary: the Route A dealloc fix "
                             "(default), or --enable-stride1-vectorization "
                             "codegen against scalar")
    parser.add_argument("--models", default=None,
                        help="comma-separated (default: all five for "
                             "--axis deallocation; resnet18,resnet50 for "
                             "--axis vectorization -- the stride-1 outliner "
                             "only reaches dense 3x3 convs)")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--l1-kb", type=int, default=32,
                        help="tile budget; 32 is the Cortex-A72 L1D")
    parser.add_argument("--out", default="build/pi4_bundle")
    parser.add_argument("--arms", default=None,
                        help="default: untiled,tiled,tiled_fused for "
                             "--axis deallocation; untiled for "
                             "--axis vectorization (tiling+vectorization has "
                             "a documented footprint blow-up, see "
                             "docs/EDGESLICE_TRANSFORM_ARCHITECTURE.md)")
    args = parser.parse_args()
    if args.models is None:
        args.models = ("resnet18,resnet50" if args.axis == "vectorization"
                       else "resnet18,mobilenet_v2,efficientnet_b0,unet,resnet50")
    if args.arms is None:
        args.arms = "untiled" if args.axis == "vectorization" else "untiled,tiled,tiled_fused"

    os.environ["EDGESLICE_STATIC_LINK"] = "1"

    import edgeslide.mlir_transform.lowering as low
    from benchmarks.alloc_attribution import build_arms
    from benchmarks.hardware_profiler import compile_mlir_artifacts
    from benchmarks.workloads import load_model, weights_provenance

    print("toolchain:")
    preflight_tools()

    shipped = list(low.SLICED_PAYLOAD_OPT_PASSES_STAGE1)
    if args.axis == "vectorization":
        # Stage-1 (dealloc) passes are identical in both arms here -- only
        # stage-2 codegen for stride-1 convs differs -- so no monkeypatching
        # of the global pass list is needed, unlike the deallocation axis.
        configs = (("before", {"enable_vectorization": False}),
                  ("after", {"enable_vectorization": True}))
        pass_lists = {"before": shipped, "after": shipped}
    else:
        # The configuration the fix replaced, rebuilt from the shipped one so
        # the two never drift apart.
        before, skip = [], False
        for p in shipped:
            if p == "--buffer-deallocation":
                before.append("--buffer-deallocation-pipeline"); skip = True
            elif skip and p in ("--cse", "--canonicalize"):
                continue
            else:
                skip = False; before.append(p)
        configs = (("before", {"passes": before}), ("after", {"passes": shipped}))
        pass_lists = {"before": before, "after": shipped}

    out = Path(args.out)
    # Build beside the destination and swap at the end. A failed build must
    # never be able to destroy a bundle that works.
    staging = out.with_name(out.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "bin").mkdir(parents=True)
    work = staging / "_build"
    work.mkdir()

    manifest = {
        "axis": args.axis,
        "resolution": args.resolution,
        "l1_cache_kb": args.l1_kb,
        "pass_lists": pass_lists,
        "vectorization": {label: cfg.get("enable_vectorization", False)
                          for label, cfg in configs},
        "binaries": [],
        "skipped": [],
    }

    for name in args.models.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            model = load_model(name).eval()
            provenance = weights_provenance(model)
            arms = build_arms(model, (1, 3, args.resolution, args.resolution),
                              l1_cache_kb=args.l1_kb, work_dir=work / name,
                              model_name=name)
        except Exception as exc:
            print(f"  ARMS FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            manifest["skipped"].append({"model": name, "stage": "build_arms",
                                        "error": str(exc)[:300]})
            continue

        for arm in args.arms.split(","):
            arm = arm.strip()
            if arm not in arms:
                continue
            built = {}
            for label, cfg in configs:
                if "passes" in cfg:
                    low.SLICED_PAYLOAD_OPT_PASSES_STAGE1[:] = cfg["passes"]
                try:
                    binary, in_shape, out_shape, _, _ = compile_mlir_artifacts(
                        arms[arm], work / f"{name}_{arm}_{label}", label,
                        enable_vectorization=cfg.get("enable_vectorization", False))
                    built[label] = (Path(binary), in_shape, out_shape)
                except Exception as exc:
                    print(f"  {arm:12s} {label:7s} BUILD FAILED "
                          f"{type(exc).__name__}: {str(exc)[:140]}", flush=True)
                    manifest["skipped"].append({"model": name, "arm": arm,
                                                "config": label,
                                                "error": str(exc)[:300]})
                finally:
                    if "passes" in cfg:
                        low.SLICED_PAYLOAD_OPT_PASSES_STAGE1[:] = shipped

            if len(built) != 2:
                continue
            # The guard that earned its place: a first attempt patched a
            # constant the binary path does not read, and both "arms" were the
            # same binary. Refuse to ship a pair that cannot show a difference.
            if sha16(built["before"][0]) == sha16(built["after"][0]):
                print(f"  {arm:12s} SKIPPED: both configs produced the same binary",
                      flush=True)
                manifest["skipped"].append({"model": name, "arm": arm,
                                            "error": "identical binaries"})
                continue

            for label, (binary, in_shape, out_shape) in built.items():
                dest = staging / "bin" / f"{name}__{arm}__{label}"
                shutil.copy2(binary, dest)
                dest.chmod(0o755)
                strip_binary(dest)
                entry = {
                    "model": name, "arm": arm, "config": label,
                    "binary": dest.name, "sha256_16": sha16(dest),
                    "bytes": dest.stat().st_size,
                    "input_shape": list(in_shape), "output_shape": list(out_shape),
                    "weights_provenance": provenance,
                }
                manifest["binaries"].append(entry)
                print(f"  {arm:12s} {label:7s} {dest.name}  "
                      f"{dest.stat().st_size/1e6:.1f} MB  sha={entry['sha256_16']}",
                      flush=True)

    runner = staging / "run_on_pi.sh"
    runner.write_text(RUN_ON_PI)
    runner.chmod(0o755)
    (staging / "time_shim.py").write_text(TIME_SHIM)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.rmtree(work, ignore_errors=True)

    if not manifest["binaries"]:
        shutil.rmtree(staging, ignore_errors=True)
        print("\nno binaries were built -- the existing bundle is untouched")
        for entry in manifest["skipped"]:
            print(f"  skipped {entry}")
        return 1

    if out.exists():
        backup = out.with_name(out.name + ".bak")
        shutil.rmtree(backup, ignore_errors=True)
        out.rename(backup)
        print(f"\nprevious bundle kept at {backup}")
    staging.rename(out)

    total = sum(b["bytes"] for b in manifest["binaries"])
    print(f"\nbundle: {out}  {len(manifest['binaries'])} binaries, "
          f"{total/1e6:.1f} MB, {len(manifest['skipped'])} skipped")
    for entry in manifest["skipped"]:
        print(f"  skipped: {entry}")
    print(f"copy with:  rsync -av {out}/ pi@<host>:~/pi4_bundle/")
    # A partial bundle is a result worth acting on, not a success: say so in
    # the exit code so a caller does not ship four models believing it has five.
    return 2 if manifest["skipped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
