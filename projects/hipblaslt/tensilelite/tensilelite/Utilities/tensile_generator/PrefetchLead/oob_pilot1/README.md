# oob_pilot1: lead from the prefetch chunks vs the constant lead, 12 kernels

Run and read as described in `../README.md` (REF = constant lead, EXP = lead from the prefetch
chunks; `<name>.yaml` runs REF, EXP and `<name>_rev.yaml` runs EXP, REF).

## Question

A workgroup's GL2 prefetch requests (one per 256-byte chunk) leave through the same request path
as the tensor_load that follows them. With the constant lead (30 WMMAs, 25 for FP4 A) a large
prefetch group can still be queued when the tensor_load is issued, which then starts late. The
lead from the chunks moves the group earlier when it has many chunks per workgroup, and later
(shorter lead) when it has few. Does that hold on kernels that were tuned for their sizes and on
two perf configs not used so far?

- Kernels with many chunks per workgroup (ClusterDim [1, 1], 512-612 chunks): the constant lead
  leaves the largest modeled tensor_load delay of any kernel measured so far.
- Kernels where the constant lead leaves no modeled delay and the new lead is shorter (24-28) or
  longer (40): no gain expected; a loss would be a cost of the move itself.

## Kernels

From the gfx1250 BF16 TN exact-tuning library logic (`OOBExactTuningBBSTN`, branch
`user/victorwu/OOBExactTuning`; category and solution index in each YAML header), converted with
`TensileLibLogicToYaml --no-run-config --keep-epilogues` (solutions keep the logic's problem type;
the logic's fused kernels are called with this config's default epilogue arguments), one problem
size per kernel with the leading dimensions of its tuning config, rotating buffer 384 MB as in the
tuning configs. Plus `perf_yamls/gfx1250_96wgp/maf/bbs_tn_maf.yaml` (the same kernel as
`gfx1250/maf/bbs_tn_maf.yaml` at N = 3072, fewer workgroups) and one point of
`gfx1250/maf/tuning/f8_tn_maf_finetune_sweep.yaml` (ClusterDim [2, 4], PrefetchGL2 1, the other
swept knobs at their first value, StinkyTofu options at their defaults).

Columns, from the develop asm of each kernel: chunks = 256-byte prefetch chunks one workgroup
issues for one tensor_load, per prefetch instruction in issue order; c_w = cycles of one WMMA;
lead = WMMAs (develop -> this branch); first prefetch = WMMAs from the first prefetch to its
tensor_load in the asm; delay = modeled tensor_load start delay at develop's placement (request
path drains 1 chunk per cycle, the first 27 chunks free; a model, not a measurement).

| name | type | WMMA | macro tile | DepthU | ClusterDim | PrefetchGL2 | chunks / WG | c_w | lead | first prefetch | delay (cycles) | size (M, N, B, K) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c11_pure_s2` | BF16 | 16x16x32 | 384x224 | 128 | [1, 1] | 2 | 128 + 128 + 128 + 112 + 112 = 608 | 8 | 30 -> 84 | 34 -> 88 | 309 | 4144, 40208, 1, 24656 |
| `c12_pure_s7` | BF16 | 16x16x32 | 320x288 | 128 | [1, 1] | 2 | 108 + 108 + 108 + 96 + 96 + 96 = 612 | 8 | 30 -> 84 | 34 -> 88 | 313 | 51904, 848, 1, 31376 |
| `c8_fused_s2` | BF16 | 16x16x32 | 256x256 | 128 | [1, 1] | 2 | 128 + 128 + 128 + 128 = 512 | 8 | 30 -> 72 | 34 -> 76 | 213 | 7104, 1648, 1, 80048 |
| `c10_pure_s2` | BF16 | 16x16x32 | 256x352 | 128 | [1, 4] | 2 | 64 + 120 + 120 + 120 = 424 | 8 | 30 -> 67 | 34 -> 72 | 125 | 24208, 32416, 1, 2336 |
| `c3_fused_ldak_s5` | BF16 | 16x16x32 | 256x320 | 128 | [2, 1] | 2 | 128 + 128 + 80 + 80 = 416 | 8 | 30 -> 60 | 34 -> 64 | 117 | 7808, 7088, 1, 5424 |
| `c2_pure_s6` | BF16 | 16x16x32 | 256x288 | 128 | [4, 1] | 1 | 128 + 128 + 72 = 328 | 8 | 30 -> 49 | 34 -> 53 | 29 | 6592, 2592, 1, 3296 |
| `mxf4_s16` | MXFP4 (FP8 out) | 16x16x128 | 128x512 | 256 | [4, 4] | 2 | 32 + 128 + 1 + 4 = 165 | 4 | 25 -> 62 (address limit: placed at 54) | 29 -> 54 | 22 | 4480, 6016, 1, 6528 |
| `perf_f8_sweep_cd2x4_pgl1` | FP8 | 16x16x128 | 256x256 | 256 | [2, 4] | 1 | 64 + 128 = 192 | 8 | 30 -> 40 | 34 -> 44 | 0 | 4096, 4096, 1, 65536 |
| `c12_pure_s8` | BF16 | 16x16x32 | 256x320 | 64 | [4, 4] | 2 | 64 + 80 = 144 | 8 | 30 -> 28 | 34 -> 32 | 0 | 59472, 7632, 1, 35168 |
| `c13_s4` | BF16 | 16x16x32 | 320x256 | 128 | [4, 4] | 2 | 80 + 64 = 144 | 8 | 30 -> 26 | 34 -> 30 | 0 | 13120, 13264, 1, 14880 |
| `c2_pure_s5` | BF16 | 16x16x32 | 256x256 | 64 | [4, 4] | 1 | 64 + 64 = 128 | 8 | 30 -> 24 | 34 -> 28 | 0 | 6128, 4784, 1, 3504 |
| `perf_96wgp_bbs_maf` | BF16 | 16x16x32 | 256x256 | 128 | [4, 4] | 2 | 64 + 64 = 128 | 8 | 30 -> 24 | 34 -> 28 | 0 | 4096, 3072, 1, 65536 |

Every kernel uses one workgroup per WGP (LDS 200-320 KB), so no other workgroup shares its
request path.

CPU check done on this branch: in every YAML REF builds the same code as develop, and EXP builds
the same code as develop with `PrefetchLeadWmmas` set to the lead in the table.

## Not included

- Kernels whose asm does not change (the lead from the chunks is not applied or lands on the same
  placement): kernels without HalfPLR or without prefetch, and HalfPLR kernels with fewer than 64
  WMMAs per stage (StinkyTofu drops the lead there).
- `gfx1250/maf/f8_tn_maf_finetune.yaml`: it already forks `StinkyTofuParameters`.
- Tuned solutions with `TDMStoreInst` (not on this branch's base).
