# EdgeSlice vs SConv Presentation Plots

Input: `real input=single-image, images=10, resolution=224x224, runs=10, warmup=3, L1=64KB`

## Generated Charts
- `C:\Users\Acer\Downloads\edge_slide\verify code edge slice\plots\single_image_latency_mean.svg`
- `C:\Users\Acer\Downloads\edge_slide\verify code edge slice\plots\edgeslice_vs_sconv_speedup.svg`
- `C:\Users\Acer\Downloads\edge_slide\verify code edge slice\plots\runtime_sliced_layers.svg`
- `C:\Users\Acer\Downloads\edge_slide\verify code edge slice\plots\peak_rss_mb.svg`

## Summary

### resnet50
- Baseline median: 126.1 ms
- EdgeSlice median: 116.5 ms (+7.6% vs baseline)
- SConv median: 140.2 ms (-11.2% vs baseline)
- EdgeSlice vs SConv: +16.9%

### mobilenet_v2
- Baseline median: 51.0 ms
- EdgeSlice median: 43.9 ms (+13.9% vs baseline)
- SConv median: 55.2 ms (-8.2% vs baseline)
- EdgeSlice vs SConv: +20.5%
