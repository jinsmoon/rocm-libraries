# DAG placement of prefetches and wait-prone fillers (gfx1250)

How the CDNA5 DAG scheduler places global/flat prefetches, SALU/VALU fillers that would cost
an `s_wait_alu`, and the rest of the fillers, the knobs that control it, and what it measured.

## Why the DAG needs rules for these

InsertWaitAlu, which runs after the DAG, decides waits by **op counts** (`HWModel::WaitHide`).
The DAG used to model the same hazards in cycles and disagreed with it; it now asks the pass's
scoreboard instead (see [Scoreboard-driven filler placement](waitalu-placement.md)).

Prefetch placement is not a hazard but a **performance** placement. Left alone, the DAG
issues a prefetch as soon as its address is ready, far ahead of the `tensor_load_to_lds` it
belongs with, so the rules below place it explicitly.

## Rules, in order of measured impact

1. **Prefetch placement** (`PrefetchLeadWmmas`). Each prefetch is grouped with the
   tensor_load that follows it; the load's window is the planned window of the stage
   barrier above it (`barrierWmmaThresholds_`).
   - Multi-stage loop body (tensilelite `HalfPLR`, 3 TDM stages): the k-th of n prefetches is
     held until window `load - lead + k * lead / n`.
   - Single-stage loop body (lead < 8): the whole group is released at the load window.
     The stage barrier does not wait for the group. When both are ready, the barrier issues
     first.
   - With a lead set, prefetches have their own ready queue; they are not fillers and never
     count against the filler quota.
   - `PrefetchLeadCycles` gives the multi-stage lead in cycles instead (see "Lead from the
     prefetch chunks" below). It becomes `ceil(cycles / c) - 2` windows, at least 8, where `c` is
     the latency of the block's first WMMA after its operand-format override (8 for
     `v_wmma_scale_f32_16x16x128_f8f6f4`, 4 for its FP4 x FP4 form). The 2 windows are the
     distance the first prefetch lands beyond its lead (measured +2.0 to +2.8), because the stage
     barrier's planned window stands in for the tensor_load's. Below 8 the scheduler would switch
     to single-stage grouping, so the lead stays at 8 or more.
2. **Wait-prone fillers** (`WaitAluHoldStrictCount`, only with ESM2, the only mode in which
   InsertWaitAlu runs). `WaitAluTracker` is InsertWaitAlu's
   own scoreboard (shared `stepInstruction()`, so the answers cannot diverge); `query(inst)`
   returns the `s_wait_alu` the pass would emit if `inst` issued next. A filler whose wait
   would be strict (count ≤ N, and in multi-stage loops any `vm_vsrc` wait, which drains
   older LDS reads whatever its count) is held until the two windows before the next
   `s_barrier_wait`. Not held when it is the last pending predecessor of real work, or when
   it feeds a prefetch's address.
3. **ds_load pacing** (`DsSpread`, `DsReadQueueDepth`). The ds credit queue bounds the
   ds_loads in flight, and with `DsSpread` each batch of `DsBatchWmmas` WMMAs takes the even
   share of its stage's ds_loads (`ceil(stage loads / stage batches)`, a stage running between
   the two barriers a load sits behind and feeds), so the stream stays even and the
   tensor_load is not pushed a window late. A held ds_load is released early only when no WMMA
   is ready in its place. See `DsSpreadPacer.hpp` and `DsSpread` in
   [scheduler-tuning-parameters](../user/scheduler-tuning-parameters.md); the
   [spread plan](wmma-batch-ds-spread-plan.md) is the earlier slot design.
4. **Critical chains.** A filler whose consumer (prefetch, tensor_load, ds_load) is planned
   within 20 windows skips the filler quota and takes the earliest window with a free
   co-issue slot, or issues as soon as it is ready once every remaining window is full.
5. **Even spread** (`EvenSpreadFillers`) of the remaining fillers: each window is owed
   `ceil(fillers / WMMAs)` and closes once that quota is met; the quota never holds a
   ds_load or tensor_load.

The Phase G fallback skips held prefetches and fillers while anything else can go, and
takes them last, so a hold can delay work but never strand it.

## Knobs

All are `ModuleOptions`, settable from tensilelite. The full list of scheduler tuning values
is in [Scheduler tuning parameters](../user/scheduler-tuning-parameters.md). `stinkytofu-opt` flags set the matching
`dagFeatures` field (default off there).

| ModuleOption | default | stinkytofu-opt flag | meaning |
|---|---|---|---|
| `PrefetchLeadWmmas` | 25 (KernelWriter: no HalfPLR 4; else 25 sub-byte A, 30 otherwise; a lead of 8 or more is dropped where a stage has < 64 WMMAs) | `--prefetch-lead-wmmas=N` | prefetch lead; < 8 = single-stage grouping; 0 = off |
| `PrefetchLeadCycles` | 0 (KernelWriter: the prefetch group's drain time with HalfPLR) | `--prefetch-lead-cycles=N` | > 0: the lead in cycles, replaces `PrefetchLeadWmmas` in blocks with WMMAs |
| `WaitAluHoldStrictCount` | 2 | `--wait-alu-hold-strict-count=N` | strict-wait threshold; < 0 = no hold |
| `DsSpread` | true | `--ds-spread` / `--no-ds-spread` | spread the region's ds_loads evenly over its WMMAs |
| `EvenSpreadFillers` | true | – | filler quota per window |
| `WarGateWmmas` | -1 | `--war-gate-wmmas=N` | WMMA-src → ds_load overwrite gap; -1 = derived |
| `WmmaQueueCoverCycles` | -1 = automatic, (queue depth - 1) x the next WMMA's latency (`stinkytofu-opt`: 0 = off) | `--wmma-queue-cover-cycles=N` | cycles of queued WMMA work that must remain before a ds_load, filler or tensor_load may issue; below it, with room in the queue, the next ready WMMA goes first. Forced picks (a promoted barrier) are not held. Not 0 also merges the waits of back-to-back WMMAs onto the first one. 0 = queue model off (one WMMA at a time). The queue depth comes from each matrix opcode's `HwInstDesc`, generated from the architecture `.def`: 5 for `v_wmma_scale*`, 8 otherwise |

## Measured effect

gfx1250, b8-3, develop 40b5cbed as REF, 3 reps with REF before and after each kernel,
median of the 6 comparisons (medium mxf4: 10). MAF = 4096×4096×65536 TN, medium =
2048×2048×65536 TN, henry = F8BS 256×256×256.

| | % vs develop |
|---|---|
| henry | +2.3 (handwritten golden: +1.4) |
| MAF bbs / f8 / mxf4 / mxf8 / nvf4 | +0.4 / +0.5 / +0.2 / +0.5 / +0.15 |
| medium bbs / f8 / mxf4 / mxf8 | +2.2 / −0.1 / ≈ +0.4 / −1.2 |

SQTT WMMA-issue bubbles per main-loop iteration, henry: develop 687, this change 497,
handwritten 578. Most of what remains is the stage-barrier and tensorcnt-barrier waits
(waves waiting for each other).

Findings behind the rules:

- A prefetch early in its stage costs 2–5%; a single-stage loop wants the group right
  before the stage barrier (+2% over develop on medium mxf4 in a hand edit), a multi-stage
  loop a staggered lead. A lead of 80 on henry is −14%.
- A stage under 64 WMMAs (MAB, medium mxf8) has no room for a multi-stage lead: the address
  chain is ready only a few windows before the tensor_load, and a prefetch issued there is
  worse than one grouped at the tensor_load. Lead 0 issues them mid-stage among the ds_loads,
  and the next WMMAs wait longer for their LDS data: MAB bf16 is 0.41% slower and MAB nvf4
  0.63%. The single-stage grouping (lead below 8) is therefore kept at any stage length; it
  brings MAB nvf4 level with develop and MAB bf16 to 0.25% behind.
- An LDS-address swap (`v_xor`) issued while ds_loads still read the old address pays a
  `vm_vsrc` wait (about 100 cycles/iteration on medium mxf4). Moving it after the barrier
  is worse: the next stage's first ds_loads read the new address right after the barrier.
- A filler or a forced VALU that takes a ds_load's slot in a saturated stream pushes the
  tensor_load one window late, which is −3% on medium f8.
- Even spread helps the f4 MAF kernels (+1.5 vs off) and is neutral elsewhere once the
  critical chains are exempt.

## Lead from the prefetch chunks (KernelWriter)

With HalfPLR, KernelWriter sets `PrefetchLeadCycles` (`GL2PrefetchLoad.leadCycles`) from the GL2
prefetches in front of one tensor_load.

Why a lead is needed: each prefetch instruction sends one GL2 request per 256-byte chunk
(`GlobalPrefetchSize`), and a workgroup's prefetch requests leave through the same request path
as its tensor_load. A tensor_load issued while prefetch chunks are still queued starts late by the
time they take to drain, and the wave waits for that tensor_load at `s_wait_tensorcnt`, so the
delay becomes stall. The lead is how long before the tensor_load the prefetches must start so that
they have drained by then, plus a margin.

```
cycles = max_j [ sum_{i>=j} C_i / r * n / (n - j) ] + M
```

- `C_i`: chunks one workgroup issues with the i-th of the n prefetch instructions, in issue order.
  KernelWriter already counts the chunks of the cluster's prefetch block (`gl2nc`); the cluster's
  workgroups share them evenly.
- `r = 1` chunk per cycle: how fast the request path drains. In a microbenchmark, a tensor_load
  issued right after k prefetch chunks started about k - 27 cycles late, i.e. one cycle per chunk.
  The 27 chunks that caused no delay there are not credited here; they stay inside the margin.
- `M = 75` cycles: a fixed margin. Placing the prefetches exactly at the drain time still left some
  kernels slightly slower (up to about 0.6%); with this margin all 13 measured MAF kernels were at
  least as fast as the constant lead, within 0.5% noise. It is a measured setting, not derived.
- The scheduler spreads the prefetches over the lead (the j-th at `lead * (n - j) / n`), so the
  prefetches from j on get only part of it; each such suffix must drain in its share. With the
  largest prefetches first, the whole group (j = 0) sets the lead.

Example: 128 + 128 + 16 + 16 = 288 chunks give 288 + 75 = 363 cycles; with 8-cycle WMMAs the lead
is ceil(363 / 8) - 2 = 44 WMMAs.

What the model does not cover:

- A lead that is too far. A prefetch early in its stage costs (see above), and measured MAF
  kernels lose again past a lead of about 56 to 70 windows at 128 WMMAs per stage. There is no
  model for that side, so the lead has no upper bound other than the address chain: a prefetch
  is never issued before its address is ready.
- Stage-end ds_load effects, and single-stage loops (no HalfPLR, which keep `PrefetchLeadWmmas`).

## Open

- An upper bound for the lead (the far side above) and an address-chain cap in the scheduler.
- The `- 2` windows (`kLeadPlacementWindows`) correct for the placement, not the model: the lead
  is counted from the stage barrier's planned window, and the first prefetch lands about 2
  windows further from the tensor_load. Counting from the tensor_load's own window would remove
  the correction.
