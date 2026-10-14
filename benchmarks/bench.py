#!/usr/bin/env python3
"""Experimental PyTorch bypass benchmark.

This benchmark exercises the legacy PyTorch implementation, not the MLIR
compiler. It is intentionally opt-in so its numbers cannot be mistaken for
end-to-end compiler evidence.
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Both the package root (for `benchmarks.*`) and src (for `edgeslide.*`), so
# this runs as a script as well as an import.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from benchmarks.metrics import summarize_timing_samples
from benchmarks.workloads import SUPPORTED_MODELS, load_model
from edgeslide.edgeslide_pytorch import (
    EdgeSlideTransform,
    MemoryProfiler,
    SlicingConfig,
)


def require_explicit_bypass_opt_in(args: argparse.Namespace) -> None:
    if not args.allow_pytorch_bypass:
        raise SystemExit(
            "This benchmark uses the legacy PyTorch bypass and is not compiler evidence. "
            "Re-run with --allow-pytorch-bypass if you explicitly need it."
        )



def _pct(value: Optional[float]) -> str:
    """Render a percentage, or "n/a" when it was never measured."""
    return "n/a" if value is None else f"{value:.1f}%"


def _ms(timing: Dict[str, Any]) -> str:
    """Render a timing summary as "median [lo, hi] ms (n=N)", or "n/a"."""
    median = timing.get("median_ms")
    if median is None:
        return "n/a"
    interval = timing.get("median_ci95_ms")
    span = f" [{interval[0]:.1f}, {interval[1]:.1f}]" if interval else ""
    return f"{median:.1f}{span} ms (n={timing.get('sample_count', 0)})"


def _significant_speedup_pct(
    baseline: Dict[str, Any], transformed: Dict[str, Any]
) -> Optional[float]:
    """Speedup only when the two 95% CIs do not overlap; otherwise None.

    The previous version divided two five-run means and reported the result as
    a speedup. That gave 62.5 ms -> 53.7 ms, a 14% "win", for a transform whose
    own memory reduction in the same run was 0.0%. With overlapping intervals
    there is no speedup to report, and None says so rather than a number that
    reads like one.
    """
    base_ci = baseline.get("median_ci95_ms")
    fast_ci = transformed.get("median_ci95_ms")
    base_median = baseline.get("median_ms")
    fast_median = transformed.get("median_ms")
    if not base_ci or not fast_ci or not base_median or fast_median is None:
        return None
    # Disjoint intervals, transformed strictly below baseline.
    if fast_ci[1] >= base_ci[0]:
        return None
    return 100.0 * (1.0 - fast_median / base_median)


@dataclass
class BenchmarkResult:
    """Benchmark result for a model."""
    model_name: str
    model_type: str
    num_parameters: int
    model_size_mb: float
    baseline_peak_memory_mb: float
    edgeslide_peak_memory_mb: float
    # None when there was no baseline to reduce against, i.e. the baseline
    # measurement failed. A 0.0 there would be a real-looking number standing
    # in for a missing one.
    memory_reduction_percent: Optional[float]
    # Full summaries (median + bootstrap CI + sample count), not bare means:
    # the previous fields were means of five runs, and their ratio was reported
    # as a speedup with nothing to say whether it cleared the noise.
    baseline_timing: Dict[str, Any]
    edgeslide_timing: Dict[str, Any]
    # None unless the two CIs are disjoint. A point-estimate ratio that does
    # not clear the intervals is not a speedup, and this repo reports
    # runtime_speedup_claim_supported=False for exactly that reason.
    speedup_pct_if_significant: Optional[float]
    num_layers_transformed: int
    layers_transformed: List[str]
    input_shape: tuple
    method: str = "edgeslide_standalone"


class EdgeSlideStandaloneBenchmark:
    """Run EdgeSlide benchmarks on standalone models."""

    # Real pretrained workloads only. This file used to define two of its own
    # architectures — a "Simple CNN" and a "Large CNN simulating 3D detection
    # (PointPillars-like)" — with random weights and no matching checkpoint.
    # Numbers measured on those describe an architecture that exists nowhere
    # else, and the "simulating" one describes a model this project never had.
    # Same disposition as UNetWorkload: removed, see docs/RELIABILITY.md §3.
    MODELS = {
        name: {
            "input_shape": (1, 3, 224, 224),
            "description": f"pretrained {name} (torchvision/torch.hub weights)",
        }
        for name in SUPPORTED_MODELS
    }

    def __init__(self, l1_cache_kb: int = 512, device: str = "cpu", verbose: bool = True):
        self.l1_cache_kb = l1_cache_kb
        self.device = device
        self.verbose = verbose

        if torch.cuda.is_available():
            self.device = "cuda"
            if self.verbose:
                print(f"Using CUDA: {torch.cuda.get_device_name(0)}")

    def create_model(self, model_name: str) -> nn.Module:
        """Load a pretrained model by name, with its published weights."""
        return load_model(model_name)

    def measure_inference_time(self, model: nn.Module, input_data: torch.Tensor,
                               num_runs: int = 30) -> List[float]:
        """Per-run inference times in ms.

        Returns every sample rather than their mean. A mean of five runs was
        what this used to report, and a difference between two such means is
        not distinguishable from scheduling noise; the caller summarises these
        with a bootstrap CI so a speedup either clears the interval or is not
        claimed. 30 runs by default because that is the smallest count at which
        the bootstrap is worth computing.
        """
        model.eval()
        times = []

        # Warmup. Failures propagate: a model that cannot run must not
        # produce timing numbers (run_all_benchmarks reports and skips it).
        with torch.no_grad():
            _ = model(input_data)

        # Timed runs
        for _ in range(num_runs):
            gc.collect()

            start = time.perf_counter()

            with torch.no_grad():
                _ = model(input_data)

            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        return times

    def estimate_model_memory(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """Estimate memory usage by computing tensor sizes.

        This is more accurate than tracemalloc for PyTorch models.
        """
        model.eval()

        # Model parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

        # Input data
        input_memory = input_data.numel() * input_data.element_size()

        # Estimate activations by running inference and capturing output sizes
        activation_memory = 0
        hooks = []

        def capture_output(module, input, output):
            nonlocal activation_memory
            if isinstance(output, torch.Tensor):
                activation_memory += output.numel() * output.element_size()
            elif isinstance(output, (list, tuple)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        activation_memory += o.numel() * o.element_size()

        # Register hooks on all leaf modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(capture_output))

        # Run inference; always detach hooks, but let failures propagate so
        # a broken model never yields a bogus memory estimate.
        try:
            with torch.no_grad():
                _ = model(input_data)
        finally:
            for hook in hooks:
                hook.remove()

        # Total memory estimate
        total_bytes = param_memory + input_memory + activation_memory
        return total_bytes / (1024 * 1024)  # Convert to MB

    def measure_peak_memory(self, model: nn.Module, input_data: torch.Tensor) -> float:
        """Measure peak memory during inference."""
        model.eval()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # For CUDA, use built-in memory tracking
        if torch.cuda.is_available() and input_data.is_cuda:
            with torch.no_grad():
                _ = model(input_data)

            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # For CPU, estimate based on tensor sizes
            peak_mb = self.estimate_model_memory(model, input_data)

        return peak_mb

    def apply_edgeslide_transform(self, model: nn.Module) -> tuple:
        """Apply EdgeSlide transform to model."""
        if self.verbose:
            print("  Applying EdgeSlide transform...")

        config = SlicingConfig(
            l1_cache_bytes=self.l1_cache_kb * 1024,
            enabled=True,
            verbose=self.verbose
        )

        transform = EdgeSlideTransform(config)
        transformed_model = transform.transform_model(model)

        num_layers = len(transform.transformed_layers)
        layer_names = list(transform.transformed_layers.keys())

        if self.verbose:
            print(f"  Transformed {num_layers} layers")
            if layer_names and self.verbose:
                print(f"  Layers: {layer_names}")

        return transformed_model, num_layers, layer_names

    def benchmark_model(self, model_name: str) -> BenchmarkResult:
        """Run complete benchmark for a model."""
        print("\n" + "=" * 70)
        print(f"EDGESSLIDE BENCHMARK: {model_name}")
        print("=" * 70)

        model_info = self.MODELS[model_name]

        # Create model
        if self.verbose:
            print(f"\nCreating {model_name}...")

        baseline_model = self.create_model(model_name)
        baseline_model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in baseline_model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in baseline_model.parameters()) / (1024 * 1024)

        if self.verbose:
            print(f"  Parameters: {num_params:,}")
            print(f"  Model size: {model_size_mb:.1f} MB")
            print(f"  Input shape: {model_info['input_shape']}")

        # Synthetic input — provenance: dummy_randn with a fixed seed, so runs
        # are reproducible and results are attributable to the same data.
        torch.manual_seed(0)
        input_data = torch.randn(*model_info["input_shape"]).to(self.device)

        # Measure baseline
        if self.verbose:
            print("\n  Measuring baseline...")

        baseline_samples = self.measure_inference_time(baseline_model, input_data)
        baseline_timing = summarize_timing_samples(baseline_samples)
        baseline_memory = self.measure_peak_memory(baseline_model, input_data)

        if self.verbose:
            print(f"    Time: {_ms(baseline_timing)}")
            print(f"    Memory: {baseline_memory:.1f} MB")

        # Profile memory
        if self.verbose:
            print("\n  Profiling memory usage...")
            profiler = MemoryProfiler(baseline_model)
            memory_usage = profiler.profile(input_data)
            # Only print top 5 layers
            sorted_layers = sorted(
                memory_usage.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for name, size_mb in sorted_layers:
                print(f"    {name}: {size_mb:.2f} MB")

        # Apply EdgeSlide transform
        # Create a fresh model for transformation
        edgeslide_model = self.create_model(model_name)
        edgeslide_model.eval()

        transformed_model, num_transformed, transformed_layers = self.apply_edgeslide_transform(
            edgeslide_model
        )

        # Measure EdgeSlide
        if self.verbose:
            print("\n  Measuring EdgeSlide...")

        edgeslide_samples = self.measure_inference_time(transformed_model, input_data)
        edgeslide_timing = summarize_timing_samples(edgeslide_samples)
        edgeslide_memory = self.measure_peak_memory(transformed_model, input_data)

        if self.verbose:
            print(f"    Time: {_ms(edgeslide_timing)}")
            print(f"    Memory: {edgeslide_memory:.1f} MB")

        # None, not 0, when there is no baseline to reduce against: a
        # zero-byte baseline means the measurement failed, and a "0% reduction"
        # for it is a real-looking number standing in for a missing one.
        memory_reduction = (
            100 * (1 - edgeslide_memory / baseline_memory)
            if baseline_memory > 0
            else None
        )

        return BenchmarkResult(
            model_name=model_name,
            model_type=model_info["description"],
            num_parameters=num_params,
            model_size_mb=model_size_mb,
            baseline_peak_memory_mb=baseline_memory,
            edgeslide_peak_memory_mb=edgeslide_memory,
            memory_reduction_percent=memory_reduction,
            baseline_timing=baseline_timing,
            edgeslide_timing=edgeslide_timing,
            speedup_pct_if_significant=_significant_speedup_pct(
                baseline_timing, edgeslide_timing
            ),
            num_layers_transformed=num_transformed,
            layers_transformed=transformed_layers,
            input_shape=model_info["input_shape"],
            method="edgeslide_standalone",
        )

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        results = []

        for model_name in self.MODELS.keys():
            try:
                result = self.benchmark_model(model_name)
                results.append(result)
            except Exception as e:
                print(f"\n❌ {model_name} failed: {e}")
                import traceback
                traceback.print_exc()

        return results

    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save results to JSON."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        output_data = {
            "metadata": {
                "benchmark_suite": "EdgeSlide",
                "benchmark_type": "standalone",
                "description": "EdgeSlide transform on standalone PyTorch models",
                "method": "EdgeSlide PyTorch transform",
                "input_source": "dummy_randn (torch.manual_seed(0))",
                "l1_cache_kb": self.l1_cache_kb,
                "device": self.device,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_models": len(results),
            },
            "results": [asdict(r) for r in results],
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to {output_file}")

    def print_results_table(self, results: List[BenchmarkResult]):
        """Print results in table format."""
        print("\n" + "=" * 100)
        print("EDGESSLIDE STANDALONE BENCHMARK RESULTS")
        print("=" * 100)

        print(f"\n{'Model':<15} {'Params':>10} {'Baseline':>12} {'EdgeSlide':>12} {'Reduction':>10} {'Time':>10} {'Layers':>8}")
        print("-" * 100)

        for r in results:
            print(f"{r.model_name:<15} {r.num_parameters:>10,} "
                  f"{r.baseline_peak_memory_mb:>10.1f}MB "
                  f"{r.edgeslide_peak_memory_mb:>10.1f}MB "
                  f"{_pct(r.memory_reduction_percent):>10} "
                  f"{_ms(r.edgeslide_timing):>26s} "
                  f"{r.num_layers_transformed:>6}")

        print("\n" + "=" * 100)
        print("NOTES:")
        print("- Baseline = Actual peak memory measured during inference")
        print("- EdgeSlide = Actual memory with real transform applied")
        print(f"- L1 Cache threshold: {self.l1_cache_kb} KB")
        print(f"- Device: {self.device}")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="EdgeSlide Standalone Benchmark")
    parser.add_argument("--allow-pytorch-bypass", action="store_true",
                        help="run the legacy PyTorch bypass (not compiler evidence)")
    parser.add_argument("--model", "-m", type=str,
                        choices=list(SUPPORTED_MODELS),
                        help="Model to benchmark (default: all)")
    # 32 KB, the real Cortex-A72 L1D (EDGE_CACHE_LEVELS["pi4"]). The old
    # default of 512 was an L2-sized number in an L1-named flag; every tile
    # decision taken under it was admitted against a cache that does not exist
    # on the target. See docs/PHASE3_CACHE_AND_BASELINES.md.
    parser.add_argument("--l1-cache", type=int, default=32,
                        help="L1 cache size in KB (default: 32, the Cortex-A72 L1D)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--output", "-o", type=str,
                        default="benchmarks/results/edgeslide_standalone_results.json",
                        help="Output JSON file")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce verbosity")

    args = parser.parse_args()
    require_explicit_bypass_opt_in(args)

    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    runner = EdgeSlideStandaloneBenchmark(
        l1_cache_kb=args.l1_cache,
        device=device,
        verbose=not args.quiet
    )

    print("=" * 70)
    print("EdgeSlide Standalone Benchmark")
    print("=" * 70)
    print(f"L1 cache: {args.l1_cache} KB")
    print(f"Device: {runner.device}")

    if args.model:
        results = [runner.benchmark_model(args.model)]
    else:
        results = runner.run_all_benchmarks()

    if results:
        runner.print_results_table(results)
        runner.save_results(results, args.output)
    else:
        print("\n❌ No results to save")


if __name__ == "__main__":
    main()
