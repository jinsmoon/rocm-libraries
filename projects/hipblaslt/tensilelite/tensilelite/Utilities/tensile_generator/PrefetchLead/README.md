# PrefetchLead: GL2 prefetch lead experiments (gfx1250)

Each subdirectory is one experiment: Tensile YAMLs that build and measure the compared kernels in
one client launch, and a README with the question, the kernels and how to read the result.

## What is compared

On gfx1250 multi-stage (HalfPLR) kernels the GL2 prefetches for a later stage are placed a number
of WMMAs before the tensor_load they precede (the lead). This branch derives the lead from the
256-byte prefetch chunks one workgroup issues (`GL2PrefetchLoad.leadCycles`, module option
`PrefetchLeadCycles`); `PrefetchLeadCycles: 0` keeps the constant lead (`PrefetchLeadWmmas`).

Every YAML forks `StinkyTofuParameters` (per-solution StinkyTofu module options, applied after
the KernelWriter defaults; Tensile tags the kernel name with `_STP_<options>`):

| label | StinkyTofuParameters | kernel |
|---|---|---|
| REF | `{PrefetchLeadCycles: 0}` | constant lead, same asm as develop |
| EXP | `{}` | this branch's default: the lead from the prefetch chunks |

`<name>.yaml` runs REF then EXP in every round and `<name>_rev.yaml` runs EXP then REF, so a
position effect in the round shows up as a difference between the two files.

## Run

Needs this branch (after `invoke rocisa`), gfx1250, and the reserved GPU only: one job at a time,
check that the device is free before each launch, stop on a refusal or hang. `prefetch_lead.py`
runs the steps below (with the venv's python, from any directory); each step can also be done by
hand as described.

1. CPU check (no GPU):
   `prefetch_lead.py run <exp> <cpu_out> --cpu-only`, then `prefetch_lead.py asm <cpu_out>`:
   every build has one REF and one EXP kernel, and their code differs where the experiment
   README says so. By hand: `Tensile --cpu-only --gpu-targets gfx1250 <exp>/<name>.yaml <out>`;
   the asm is under `<out>/1_BenchmarkProblems/*/00_Final/caches/*/source/build_tmp/SOURCE/assembly/`
   (`KeepBuildTmp: true`).
2. Measure:
   `ROCR_VISIBLE_DEVICES=<gpu> prefetch_lead.py run <exp> <out> [<name> ...] --check '<device-free check>'`
   runs Tensile on `<name>.yaml` then `<name>_rev.yaml` for each kernel, one at a time, into
   `<out>/<file>/` (log in `<out>/<file>.log`). It skips files already done and stops at the
   first failure of the check or of Tensile; nothing is retried. `Device: 0` in the YAML is the
   first visible device.
3. Read: `prefetch_lead.py collect <out> [--csv rounds.csv]`. It reads the client CSV of each run
   (`<out>/<file>/2_BenchmarkData/*.csv`: one row per round, one GFlops column per solution;
   `NumBenchmarks` rounds, `CSVMergeSameProblemID: false`). The solution is identified by its
   kernel-name tag (none = EXP; `_STP_PrefetchLeadCycles0` = REF). Check in the first run that the
   solutions run in the YAML's order within a round (the client log).
   - per round: speed = REF time / EXP time - 1 (+ = EXP faster than REF);
   - per kernel: the median of each file and over the rounds of both files;
   - a median below 0.5% is not a difference, and neither is one where the two files disagree in
     sign (then the order in the round matters more than the lead); the count of rounds with EXP
     faster shows how consistent the sign is.

Measurement settings (in every YAML, not to be changed between REF and EXP): 100 warmups,
1000 enqueues per sync, 1 sync per round, 5 rounds, no validation, the rotating buffer of the
kernel's source config.

## Directories

- `oob_pilot1/`: 12 kernels from tuned library-logic solutions and two perf configs; the lead
  from the chunks vs the constant lead.
