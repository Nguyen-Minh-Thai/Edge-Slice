#!/usr/bin/env python3
"""Generate presentation-ready charts from EdgeSlice/SConv benchmark JSON.

The script intentionally uses only the Python standard library and writes SVG
files, so it works in minimal Ubuntu environments without matplotlib.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
# Must match run_sixcam_sconv_comparison.RESULTS_PATH — the only producer of
# the schema this script reads.
DEFAULT_INPUT = ROOT / "benchmarks" / "results" / "sixcam_sconv_comparison.json"
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks" / "plots"

METHOD_ORDER = ("baseline", "edgeslice", "sconv")
METHOD_COLORS = {
    "baseline": "#6b7280",
    "edgeslice": "#2563eb",
    "sconv": "#dc2626",
}
METHOD_LABELS = {
    "baseline": "Baseline",
    "edgeslice": "EdgeSlice",
    "sconv": "SConv",
}


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt_ms(value: Optional[float]) -> str:
    """Milliseconds, or "n/a" when the statistic was never computed.

    `summarize_timing_samples` returns None rather than 0.0 for an empty
    sample set — a timing of 0 ms is not a fast measurement, it is the absence
    of one — so every formatter here has to survive that.
    """
    return "n/a" if value is None else f"{value:.1f} ms"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _maybe_float(value: Any) -> Optional[float]:
    """Coerce to float, preserving None instead of raising on it."""
    return None if value is None else float(value)


def _write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>
  <style>
    .title {{ font: 700 28px Inter, Arial, sans-serif; fill: #111827; }}
    .subtitle {{ font: 500 14px Inter, Arial, sans-serif; fill: #4b5563; }}
    .axis {{ stroke: #374151; stroke-width: 1.4; }}
    .grid {{ stroke: #e5e7eb; stroke-width: 1; }}
    .tick {{ font: 12px Inter, Arial, sans-serif; fill: #4b5563; }}
    .label {{ font: 13px Inter, Arial, sans-serif; fill: #111827; }}
    .value {{ font: 700 12px Inter, Arial, sans-serif; fill: #111827; }}
    .small {{ font: 11px Inter, Arial, sans-serif; fill: #6b7280; }}
  </style>
{body}
</svg>
"""
    path.write_text(svg)


def _metadata_subtitle(report: Mapping[str, Any]) -> str:
    metadata = report.get("metadata", {})
    input_mode = metadata.get("input_mode", "unknown")
    resolution = metadata.get("resolution_hw", ["?", "?"])
    runs = metadata.get("runs", "?")
    warmup = metadata.get("warmup", "?")
    image_count = metadata.get("image_count") or metadata.get("flattened_image_count")
    l1_cache_kb = metadata.get("l1_cache_kb", "?")
    return (
        f"real input={input_mode}, images={image_count}, "
        f"resolution={resolution[0]}x{resolution[1]}, runs={runs}, "
        f"warmup={warmup}, L1={l1_cache_kb}KB"
    )


def _results_by_model(report: Mapping[str, Any]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for result in report.get("results", []):
        grouped.setdefault(result["model"], {})[result["method"]] = result
    return grouped


def _available_methods(model_results: Mapping[str, Mapping[str, Any]]) -> List[str]:
    ordered = [method for method in METHOD_ORDER if method in model_results]
    extras = sorted(method for method in model_results if method not in METHOD_ORDER)
    return ordered + extras


def _nice_upper(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    return ((int(value / magnitude) + 1) * magnitude)


def _bar_chart(
    *,
    title: str,
    subtitle: str,
    groups: Sequence[str],
    series: Sequence[str],
    values: Mapping[Tuple[str, str], float],
    value_formatter,
    output_path: Path,
    y_axis_label: str,
    zero_center: bool = False,
) -> None:
    width = 1280
    height = 720
    margin_left = 100
    margin_right = 50
    margin_top = 105
    margin_bottom = 115
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    all_values = [values[(group, item)] for group in groups for item in series if (group, item) in values]
    if not all_values:
        raise ValueError(f"No values to plot for {title}")

    if zero_center:
        max_abs = max(abs(value) for value in all_values) or 1.0
        y_min = -max_abs * 1.2
        y_max = max_abs * 1.2
    else:
        y_min = 0.0
        y_max = _nice_upper(max(all_values) * 1.12)

    def y_pos(value: float) -> float:
        normalized = (value - y_min) / (y_max - y_min)
        return margin_top + chart_height - normalized * chart_height

    parts = [
        f'  <text x="{margin_left}" y="46" class="title">{_escape(title)}</text>',
        f'  <text x="{margin_left}" y="74" class="subtitle">{_escape(subtitle)}</text>',
        f'  <text x="22" y="{margin_top + chart_height / 2}" class="small" transform="rotate(-90 22 {margin_top + chart_height / 2})">{_escape(y_axis_label)}</text>',
    ]

    tick_count = 5
    for tick_index in range(tick_count + 1):
        tick_value = y_min + (y_max - y_min) * tick_index / tick_count
        tick_y = y_pos(tick_value)
        parts.append(f'  <line x1="{margin_left}" y1="{tick_y:.1f}" x2="{width - margin_right}" y2="{tick_y:.1f}" class="grid"/>')
        parts.append(f'  <text x="{margin_left - 12}" y="{tick_y + 4:.1f}" text-anchor="end" class="tick">{_escape(value_formatter(tick_value))}</text>')

    zero_y = y_pos(0.0)
    parts.append(f'  <line x1="{margin_left}" y1="{zero_y:.1f}" x2="{width - margin_right}" y2="{zero_y:.1f}" class="axis"/>')
    parts.append(f'  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" class="axis"/>')

    group_width = chart_width / max(1, len(groups))
    bar_gap = 8
    inner_width = min(group_width * 0.72, 230)
    bar_width = max(14, (inner_width - bar_gap * (len(series) - 1)) / max(1, len(series)))

    for group_index, group in enumerate(groups):
        group_center = margin_left + group_width * group_index + group_width / 2
        start_x = group_center - inner_width / 2
        parts.append(f'  <text x="{group_center:.1f}" y="{height - 58}" text-anchor="middle" class="label">{_escape(group)}</text>')
        for series_index, item in enumerate(series):
            key = (group, item)
            if key not in values:
                continue
            value = values[key]
            bar_x = start_x + series_index * (bar_width + bar_gap)
            bar_y = min(y_pos(value), zero_y)
            bar_height = abs(y_pos(value) - zero_y)
            color = METHOD_COLORS.get(item, "#7c3aed")
            parts.append(
                f'  <rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>'
            )
            label_y = bar_y - 8 if value >= 0 else bar_y + bar_height + 17
            parts.append(f'  <text x="{bar_x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" class="value">{_escape(value_formatter(value))}</text>')

    legend_x = width - margin_right - 310
    legend_y = 34
    for legend_index, item in enumerate(series):
        item_x = legend_x + legend_index * 104
        parts.append(f'  <rect x="{item_x}" y="{legend_y}" width="14" height="14" rx="3" fill="{METHOD_COLORS.get(item, "#7c3aed")}"/>')
        parts.append(f'  <text x="{item_x + 20}" y="{legend_y + 12}" class="small">{_escape(METHOD_LABELS.get(item, item))}</text>')

    _write_svg(output_path, width, height, "\n".join(parts))


def plot_latency(report: Mapping[str, Any], output_dir: Path) -> Path:
    grouped = _results_by_model(report)
    groups = list(grouped)
    series = list(METHOD_ORDER)
    values: Dict[Tuple[str, str], float] = {}
    for model, model_results in grouped.items():
        for method in _available_methods(model_results):
            values[(model, method)] = float(model_results[method]["wall_clock_ms"]["mean"])
    output_path = output_dir / "single_image_latency_mean.svg"
    _bar_chart(
        title="Mean Latency by Method",
        subtitle=_metadata_subtitle(report),
        groups=groups,
        series=series,
        values=values,
        value_formatter=_fmt_ms,
        output_path=output_path,
        y_axis_label="Mean latency (ms, lower is better)",
    )
    return output_path


def plot_edgeslice_vs_sconv(report: Mapping[str, Any], output_dir: Path) -> Path:
    grouped = _results_by_model(report)
    values: Dict[Tuple[str, str], float] = {}
    groups: List[str] = []
    for model, model_results in grouped.items():
        if "edgeslice" not in model_results or "sconv" not in model_results:
            continue
        groups.append(model)
        edge_mean = float(model_results["edgeslice"]["wall_clock_ms"]["mean"])
        sconv_mean = float(model_results["sconv"]["wall_clock_ms"]["mean"])
        values[(model, "edgeslice")] = ((sconv_mean - edge_mean) / sconv_mean * 100.0) if sconv_mean > 0 else 0.0
    output_path = output_dir / "edgeslice_vs_sconv_speedup.svg"
    _bar_chart(
        title="EdgeSlice vs SConv Speedup",
        subtitle="Positive means EdgeSlice is faster than SConv | " + _metadata_subtitle(report),
        groups=groups,
        series=("edgeslice",),
        values=values,
        value_formatter=_fmt_pct,
        output_path=output_path,
        y_axis_label="Speedup vs SConv (%)",
        zero_center=True,
    )
    return output_path


def plot_runtime_sliced_layers(report: Mapping[str, Any], output_dir: Path) -> Path:
    grouped = _results_by_model(report)
    groups = list(grouped)
    series = ("edgeslice", "sconv")
    values: Dict[Tuple[str, str], float] = {}
    for model, model_results in grouped.items():
        for method in series:
            result = model_results.get(method)
            if result is None:
                continue
            values[(model, method)] = float(result.get("method_meta", {}).get("runtime_sliced_layers", 0))
    output_path = output_dir / "runtime_sliced_layers.svg"
    _bar_chart(
        title="Runtime-Tiled Layers",
        subtitle="Layers that actually sliced/tiled during measured forwards | " + _metadata_subtitle(report),
        groups=groups,
        series=series,
        values=values,
        value_formatter=lambda value: f"{int(round(value))}",
        output_path=output_path,
        y_axis_label="Layer count",
    )
    return output_path


def plot_peak_rss(report: Mapping[str, Any], output_dir: Path) -> Path:
    grouped = _results_by_model(report)
    groups = list(grouped)
    series = list(METHOD_ORDER)
    values: Dict[Tuple[str, str], float] = {}
    for model, model_results in grouped.items():
        for method in _available_methods(model_results):
            rss_kb = float(model_results[method]["rss_kb"]["peak_during_forward_loop"])
            values[(model, method)] = rss_kb / 1024.0
    output_path = output_dir / "peak_rss_mb.svg"
    _bar_chart(
        title="Peak RSS During Forward Loop",
        subtitle=_metadata_subtitle(report),
        groups=groups,
        series=series,
        values=values,
        value_formatter=lambda value: f"{value:.0f} MB",
        output_path=output_path,
        y_axis_label="Peak RSS (MB)",
    )
    return output_path


def write_summary_markdown(report: Mapping[str, Any], output_dir: Path, plot_paths: Sequence[Path]) -> Path:
    output_path = output_dir / "benchmark_plot_summary.md"
    lines = [
        "# EdgeSlice vs SConv Presentation Plots",
        "",
        f"Input: `{_metadata_subtitle(report)}`",
        "",
        "## Generated Charts",
    ]
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    lines.extend(["", "## Summary", ""])
    for model, entries in report.get("summary", {}).items():
        lines.append(f"### {model}")
        for method in METHOD_ORDER:
            method_entry = entries.get(method)
            if not method_entry:
                continue
            # Median, not mean: the producer reports both, and the paired
            # comparison it computes is median-based, so quoting the mean here
            # would put a different statistic next to its own speedup figure.
            line = (
                f"- {METHOD_LABELS[method]} median: "
                f"{_fmt_ms(_maybe_float(method_entry.get('median_ms')))}"
            )
            # Key names must match run_sixcam_sconv_comparison's summary
            # schema; these two lines read the wrong ones and were silently
            # unreachable, so the report never carried any speedup at all.
            if "median_speedup_vs_baseline_pct" in method_entry:
                line += (
                    f" ({_fmt_pct(_maybe_float(method_entry['median_speedup_vs_baseline_pct']))}"
                    " vs baseline)"
                )
            lines.append(line)
        edgeslice_vs_sconv = entries.get("edgeslice_vs_sconv")
        if isinstance(edgeslice_vs_sconv, Mapping):
            lines.append(
                "- EdgeSlice vs SConv: "
                f"{_fmt_pct(_maybe_float(edgeslice_vs_sconv.get('median_speedup_pct')))}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot EdgeSlice/SConv benchmark output")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Benchmark JSON path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for SVG plots")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        plot_latency(report, args.output_dir),
        plot_edgeslice_vs_sconv(report, args.output_dir),
        plot_runtime_sliced_layers(report, args.output_dir),
        plot_peak_rss(report, args.output_dir),
    ]
    summary_path = write_summary_markdown(report, args.output_dir, plot_paths)
    print("Generated presentation plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

