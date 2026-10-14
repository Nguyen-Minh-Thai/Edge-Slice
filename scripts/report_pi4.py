#!/usr/bin/env python3
"""Turn `pi4_results.json` from the measurement bundle into an A/B report.

The bundle ships two binaries per (model, arm): `before`, built with the
deallocation configuration the fix replaced, and `after`, built with the
shipped one. The Pi measures both, so what comes back is a difference rather
than a level and the Pi's absolute speed never enters the claim.

    python scripts/report_pi4.py build/pi4_bundle/pi4_results.json

Reads `manifest.json` next to the results when it is there, to recover the
resolution and tile budget the bundle was built with.

Wall-clock is summarised by median, not mean: a single scheduler preemption on
a 4-core Pi moves a mean and not a median. Peak RSS is summarised by max,
because the question it answers -- will this fit -- is about the worst trial,
not the typical one.

The statistic depends on how the run was scheduled, and the runner says which
it was. With `"schedule": "round_robin"` the two configurations alternate
trial by trial, so before-trial *t* and after-trial *t* execute seconds apart
under the same temperature and load: the pairing is real, and Wilcoxon
signed-rank over those pairs (via `benchmarks.metrics`) removes exactly the
drift variance a thermal ramp injects. Without that declaration the samples
are two separate blocks in two different thermal epochs, nothing pairs them,
and only the unpaired Mann-Whitney U below is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr_pct(xs):
    """Spread as a percentage of the median -- the noise floor of the trial."""
    if len(xs) < 4:
        return float("nan")
    q = statistics.quantiles(xs, n=4, method="inclusive")
    med = statistics.median(xs)
    return 100.0 * (q[2] - q[0]) / med if med else float("nan")


def mann_whitney_p(a, b):
    """Two-sided p, normal approximation with tie correction. nan if degenerate.

    Kept as the pairing-free robustness check, and as the only test available
    for a result file the round-robin runner did not produce.
    """
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan")
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(pooled)
    ties = []
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        ties.append(j - i + 1)
        i = j + 1

    r1 = sum(r for r, (_, grp) in zip(ranks, pooled) if grp == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)

    n = n1 + n2
    mu = n1 * n2 / 2.0
    tie_term = sum(t**3 - t for t in ties)
    var = n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return float("nan")
    z = (abs(u - mu) - 0.5) / math.sqrt(var)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def parse_binary_name(name):
    """`{model}__{arm}__{config}` -- the naming build_pi4_bundle.py writes."""
    parts = name.split("__")
    if len(parts) != 3:
        return None
    return tuple(parts)


def pct_delta(before, after):
    if not before or before != before:
        return float("nan")
    return 100.0 * (after - before) / before


def field(rec, name, length, default):
    """A per-trial column, padded to the trial count for older result files."""
    values = rec.get(name) or []
    values = list(values)[:length]
    return values + [default] * (length - len(values))


def align_pairs(before, after):
    """Line the two arms up trial by trial, dropping any trial either one failed.

    Dropping in pairs, not individually, is what keeps the pairing intact: a
    crashed `before` trial removes its `after` twin too. Keeping the survivor
    would silently compare trial 7 of one arm against trial 8 of the other.
    """
    n = min(len(before.get("wall_ms", [])), len(after.get("wall_ms", [])))
    b_rc = field(before, "exit_codes", n, 0)
    a_rc = field(after, "exit_codes", n, 0)
    b_idx = field(before, "trial_index", n, None)
    a_idx = field(after, "trial_index", n, None)

    keep, dropped = [], []
    for i in range(n):
        if b_idx[i] is not None and a_idx[i] is not None and b_idx[i] != a_idx[i]:
            dropped.append((i, "trial index mismatch"))
        elif b_rc[i] != 0 or a_rc[i] != 0:
            dropped.append((i, f"exit before={b_rc[i]} after={a_rc[i]}"))
        else:
            keep.append(i)
    return keep, dropped


def thermal_notes(model, arm, label, rec, n):
    """Throttling and drift -- the failure mode that quietly inverts a result."""
    notes = []
    for state in set(field(rec, "throttled", n, "")):
        if state and state not in ("0x0", "0"):
            notes.append(f"{model}/{arm}/{label}: vcgencmd get_throttled={state} "
                         f"-- the SoC capped its own clock during this arm")
    temps = [t for t in field(rec, "temp_c_post", n, 0.0) if t]
    if len(temps) >= 2 and temps[-1] - temps[0] > 5.0:
        notes.append(f"{model}/{arm}/{label}: SoC warmed {temps[0]:.0f} -> "
                     f"{temps[-1]:.0f} C across the run")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="pi4_results.json brought back from the Pi")
    parser.add_argument("--manifest", default=None,
                        help="manifest.json (default: alongside the results)")
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.is_file():
        print(f"ERROR: no such file: {results_path}", file=sys.stderr)
        return 1
    data = json.loads(results_path.read_text())

    manifest_path = Path(args.manifest) if args.manifest \
        else results_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}

    host = data.get("host", {})
    iters = int(data.get("iters", 1) or 1)
    paired = data.get("schedule") == "round_robin"

    # What "before"/"after" mean is a property of how the bundle was built
    # (build_pi4_bundle.py --axis), not of this report -- read it rather than
    # assume, so a vectorization-axis bundle doesn't get labelled as if it
    # were the deallocation fix.
    axis = manifest.get("axis", "deallocation")
    axis_titles = {
        "deallocation": "deallocation configuration",
        "vectorization": "stride-1 conv vectorization (vs scalar codegen)",
    }
    axis_before_after = {
        "deallocation": "before = legacy --buffer-deallocation-pipeline (block-exit free), "
                        "after = shipped (last-use free)",
        "vectorization": "before = scalar codegen, after = --enable-stride1-vectorization "
                         "(outline-and-vectorize, see vectorize_conv.py)",
    }

    print("=" * 78)
    print(f" EdgeSlide Pi 4 A/B report -- {axis_titles.get(axis, axis)}")
    print("=" * 78)
    print(f" uname : {host.get('uname', '?')}")
    print(f" cpu   : {host.get('cpu', '?')}")
    print(f" cores : {host.get('nproc', '?')}   mem: {int(host.get('mem_kb', 0)) / 1e6:.2f} GB"
          f"   swap: {int(host.get('swap_kb', 0)) / 1e6:.2f} GB"
          f"   governor: {host.get('governor', '?')}")
    if manifest:
        print(f" build : resolution={manifest.get('resolution')} "
              f"l1_kb={manifest.get('l1_cache_kb')}")
        print(f" axis  : {axis_before_after.get(axis, axis)}")
    print(f" run   : {data.get('trials', '?')} trials x {iters} iters/run, "
          f"schedule={data.get('schedule', 'sequential (legacy)')}")
    print(f" stat  : {'Wilcoxon signed-rank (paired)' if paired else 'Mann-Whitney U (unpaired)'}"
          f"{'' if paired else ' -- runner did not interleave, so nothing pairs'}")
    print()

    compare = None
    if paired:
        from benchmarks.metrics import compare_paired_timings
        compare = compare_paired_timings

    # Static weights are baked into a statically-linked binary's read-only
    # data, so they inflate every RSS number in both arms by the same
    # constant and are not part of what the deallocation fix touches (see
    # PHASE1_METRIC_CORRECTION.md). Subtracting parameter_count*4 bytes
    # (float32) turns the raw RSS delta into the heap-only estimate that is
    # actually comparable to the massif numbers in ROUTE_A_RESULTS.md --
    # without it, a b->a ratio computed straight from RSS understates the
    # fix's effect because the constant term sits in both numerator and
    # denominator.
    param_count_by_model: Dict[str, int] = {}
    for entry in manifest.get("binaries", []):
        count = (entry.get("weights_provenance") or {}).get("parameter_count")
        if isinstance(count, int) and entry.get("model"):
            param_count_by_model.setdefault(entry["model"], count)

    pairs, unparsed = {}, []
    for rec in data.get("arms", []):
        parsed = parse_binary_name(rec.get("binary", ""))
        if parsed is None:
            unparsed.append(rec.get("binary"))
            continue
        model, arm, config = parsed
        pairs.setdefault((model, arm), {})[config] = rec

    header = (f"{'model':<16}{'arm':<14}{'ms/inference b→a':>22}{'Δ%':>8}{'p':>9}"
              f"{'peak RSS MB b→a':>22}{'Δ%':>8}{'Δ MB':>9}")
    print(header)
    print("-" * len(header))

    report = {"host": host, "schedule": data.get("schedule"), "iters": iters,
              "statistic": "wilcoxon_signed_rank" if paired else "mann_whitney_u",
              "rows": [], "warnings": []}
    warnings = report["warnings"]

    for (model, arm) in sorted(pairs):
        configs = pairs[(model, arm)]
        if "before" not in configs or "after" not in configs:
            have = ",".join(sorted(configs)) or "nothing"
            warnings.append(f"{model}/{arm}: incomplete pair (have: {have}) -- no comparison")
            continue

        b, a = configs["before"], configs["after"]
        keep, dropped = align_pairs(b, a)
        if dropped:
            reasons = ", ".join(sorted({r for _, r in dropped}))
            warnings.append(f"{model}/{arm}: dropped {len(dropped)} trial pair(s) "
                            f"({reasons})")
        n_all = min(len(b.get("wall_ms", [])), len(a.get("wall_ms", [])))
        if not keep:
            warnings.append(f"{model}/{arm}: every trial failed -- "
                            f"reported as 'did not run', not as a timing")
            continue

        # Per inference, not per process: with ITERS > 1 the process pays
        # exec and page-in once and amortises it over the iterations.
        bw = [b["wall_ms"][i] / iters for i in keep]
        aw = [a["wall_ms"][i] / iters for i in keep]
        br = [b["peak_rss_kb"][i] for i in keep if i < len(b.get("peak_rss_kb", []))]
        ar = [a["peak_rss_kb"][i] for i in keep if i < len(a.get("peak_rss_kb", []))]

        wall_b, wall_a = median(bw), median(aw)
        rss_b = max(br) if br else float("nan")
        rss_a = max(ar) if ar else float("nan")
        d_wall, d_rss = pct_delta(wall_b, wall_a), pct_delta(rss_b, rss_a)

        row = {
            "model": model, "arm": arm, "trials_used": len(keep),
            "trials_dropped": len(dropped),
            "ms_per_inference": {"before": wall_b, "after": wall_a,
                                 "delta_pct": d_wall},
            "wall_iqr_pct": {"before": iqr_pct(bw), "after": iqr_pct(aw)},
            "peak_rss_kb_max": {"before": rss_b, "after": rss_a,
                                "delta_pct": d_rss,
                                "delta_mb": (rss_a - rss_b) / 1024.0},
            "mann_whitney_p": mann_whitney_p(bw, aw),
        }

        # Heap-only estimate: both arms carry the same read-only weight
        # pages, so subtracting them out of RSS is what makes the Pi number
        # comparable to the dev-host massif peak instead of understating it.
        params = param_count_by_model.get(model)
        if params is not None:
            weight_kb = params * 4 / 1024.0
            heap_b = max(rss_b - weight_kb, 0.0)
            heap_a = max(rss_a - weight_kb, 0.0)
            row["heap_est_kb_max"] = {
                "before": heap_b, "after": heap_a,
                "delta_pct": pct_delta(heap_b, heap_a),
                "weight_kb": weight_kb, "parameter_count": params,
            }

        if compare is not None:
            row["paired"] = compare(bw, aw)
            p = row["paired"]["wilcoxon"].get("p_value")
        else:
            p = row["mann_whitney_p"]
        p = float("nan") if p is None else p

        print(f"{model:<16}{arm:<14}"
              f"{wall_b:>9.1f} → {wall_a:<9.1f}{d_wall:>+8.1f}{p:>9.4f}"
              f"{rss_b / 1024:>9.1f} → {rss_a / 1024:<9.1f}{d_rss:>+8.1f}"
              f"{row['peak_rss_kb_max']['delta_mb']:>+9.1f}")
        if "heap_est_kb_max" in row:
            h = row["heap_est_kb_max"]
            print(f"{'':<30}~heap-only (RSS - {params:,} params x4B): "
                  f"{h['before'] / 1024:.1f} -> {h['after'] / 1024:.1f} MiB "
                  f"({h['delta_pct']:+.1f}%)")

        # The checksum is the numerical equivalence guard: the two binaries are
        # the same computation under two deallocation configurations, so a
        # divergence here is a correctness failure, not a performance result.
        cb, ca = b.get("checksum", ""), a.get("checksum", "")
        if cb and ca and cb != ca:
            warnings.append(f"{model}/{arm}: CHECKSUM MISMATCH before={cb} after={ca} "
                            f"-- the two configurations do not compute the same thing")
        elif not cb or not ca:
            warnings.append(f"{model}/{arm}: checksum missing "
                            f"(before={cb or 'none'} after={ca or 'none'}) "
                            f"-- equivalence unverified")
        for label, rec in (("before", b), ("after", a)):
            warnings.extend(thermal_notes(model, arm, label, rec, n_all))

        for label, xs in (("before", bw), ("after", aw)):
            spread = iqr_pct(xs)
            if spread == spread and spread > 10.0:
                warnings.append(f"{model}/{arm}/{label}: wall-clock IQR is {spread:.0f}% "
                                f"of the median -- the host was not quiet")

        row["checksum"] = {"before": cb, "after": ca,
                           "match": bool(cb) and cb == ca}
        report["rows"].append(row)

    for name in unparsed:
        warnings.append(f"unrecognised binary name, skipped: {name}")

    print()
    rows = report["rows"]
    if rows:
        best = min(rows, key=lambda r: r["peak_rss_kb_max"]["delta_pct"])
        print(f"largest RSS reduction: {best['model']}/{best['arm']} "
              f"{best['peak_rss_kb_max']['delta_pct']:+.1f}% "
              f"({best['peak_rss_kb_max']['delta_mb']:+.1f} MB)")

        def pval(r):
            if "paired" in r:
                v = r["paired"]["wilcoxon"].get("p_value")
                return float("nan") if v is None else v
            return r["mann_whitney_p"]

        sig = [r for r in rows if pval(r) == pval(r) and pval(r) < 0.05]
        print(f"wall-clock differences at p<0.05: {len(sig)}/{len(rows)} arms")
        print()
        print("Reading the RSS column: these binaries are statically linked with the")
        print("weights baked in, so peak RSS carries constant weight pages that are")
        print("byte-identical in both arms. The raw RSS delta therefore UNDERSTATES the")
        print("heap effect; the '~heap-only' line under each row subtracts")
        print("parameter_count*4 bytes (float32) from both arms before taking the delta,")
        print("which is what is actually comparable to the massif peaks in")
        print("docs/ROUTE_A_RESULTS.md. It is still an estimate -- it assumes every")
        print("weight is fp32 and that no other constant data scales with the arm -- so")
        print("treat it as a sanity check on ROUTE_A's numbers, not a replacement for them.")
        missing_params = sorted({r["model"] for r in rows if "heap_est_kb_max" not in r})
        if missing_params:
            print(f"no parameter_count in manifest for: {', '.join(missing_params)} "
                  f"-- heap-only estimate skipped for those rows")
    else:
        print("no complete before/after pairs -- nothing to compare")

    if warnings:
        print(f"\nwarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
