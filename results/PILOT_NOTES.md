# Pilot (opacity1m, 30 frames, RTX A6000)

- `pilot_summary.csv`: cam00 PSNR / train time per frame / fraction of AABB-inside gaussians sent to the NTC.
- `pilot_profile_breakdown.csv`: torch.profiler on frames 25-26 of each pilot run, CUDA kernel time grouped:
  render_backward / render_forward = the two renderCUDA kernels of the 3DGStream rasterizer (larger one = backward),
  raster_preprocess_sort = preprocessCUDA + radix sort of Gaussian-tile keys + tile ranges,
  ntc_tcnn = tiny-cuda-nn hash grid + fused MLP (+ cutlass GEMM), torch_other = optimizer / loss / elementwise ops.
  Only kernel-level entries are summed (autograd wrapper entries contain their child kernels and are excluded).
- Profiled frames include profiler overhead; they are excluded from nothing in `pilot_summary.csv` (2 of 30 frames).
- A and B ran on pod BE, C on pod AC: compare kernel times across pods with care (same GPU model, different machines).

Reading: the rasterizer is ~72-79% of the listed GPU kernel time, the NTC 2-9%. Mode B cuts NTC kernel time from ~1.35 s to ~0.40 s per frame,
but the frame's train time only drops 24.8 -> 23.0 s (-7%), because rendering + backward of all gaussians is unchanged.
