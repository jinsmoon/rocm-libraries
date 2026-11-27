#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run and read the PrefetchLead experiments (see README.md).

  run EXP_DIR OUT [NAME ...] [--cpu-only] [--check CMD]
      Tensile on <NAME>.yaml, then <NAME>_rev.yaml (every kernel of EXP_DIR without NAME), one
      at a time, into OUT/<file stem>/ with the output in OUT/<file stem>.log. A file whose output
      directory already exists is skipped. Without --cpu-only ROCR_VISIBLE_DEVICES must be set,
      and --check CMD (e.g. a device-free check) runs before every launch; a non-zero exit of
      CMD or of Tensile stops the run. Nothing is retried.
  asm OUT
      Per build: the REF and EXP kernels found, and whether their code differs (kernel names and
      labels replaced).
  collect OUT [--csv FILE]
      Per kernel: speed = REF time / EXP time - 1 per round (+ = EXP faster), from the client CSV
      in OUT/<file stem>/2_BenchmarkData; median per file and over both files, rounds with EXP
      faster, and a verdict: 'noise' when |median| < 0.5% or the two files disagree in sign,
      else 'faster' / 'slower'. --csv writes every round.
"""
import argparse
import csv
import glob
import hashlib
import os
import pathlib
import re
import statistics
import subprocess
import sys

TENSILELITE = pathlib.Path(__file__).resolve().parents[3]
REF_TAG = "_STP_PrefetchLeadCycles0"
NOISE_PCT = 0.5
LABEL = re.compile(r"label_[A-Za-z0-9]+_[A-Za-z0-9=_-]+")


def kernels(exp_dir, names):
    found = sorted(p.stem for p in pathlib.Path(exp_dir).glob("*.yaml") if not p.stem.endswith("_rev"))
    for n in names:
        if n not in found:
            sys.exit(f"{n}.yaml not in {exp_dir}")
    return names or found


def label(solution):
    if solution.endswith(REF_TAG):
        return "ref"
    return "exp" if "_STP_" not in solution else None


def run(a):
    if not a.cpu_only and not os.environ.get("ROCR_VISIBLE_DEVICES"):
        sys.exit("set ROCR_VISIBLE_DEVICES to the reserved GPU (or use --cpu-only)")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tensile = [sys.executable, str(TENSILELITE / "bin" / "Tensile")]
    if a.cpu_only:
        tensile += ["--cpu-only", "--gpu-targets", "gfx1250"]
    for name in kernels(a.exp_dir, a.names):
        for stem in (name, name + "_rev"):
            dest = out / stem
            if dest.exists():
                print(f"{stem}: {dest} exists, skipped")
                continue
            if a.check and subprocess.run(a.check, shell=True).returncode != 0:
                sys.exit(f"{stem}: --check failed; stopped before launching")
            print(f"{stem}: running", flush=True)
            with open(out / f"{stem}.log", "w") as log:
                rc = subprocess.run(tensile + [str(pathlib.Path(a.exp_dir) / f"{stem}.yaml"), str(dest)],
                                    cwd=TENSILELITE.parent, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                sys.exit(f"{stem}: Tensile exit {rc}, see {out / f'{stem}.log'}; stopped")


def asm(a):
    for d in sorted(p for p in pathlib.Path(a.out).iterdir() if p.is_dir()):
        code = {}
        for s in glob.glob(f"{d}/1_BenchmarkProblems/*/00_Final/caches/*/source/build_tmp/SOURCE/assembly/*.s"):
            text = open(s).read()
            name = re.search(r"\.amdhsa_kernel (\S+)", text).group(1)
            text = text.replace(name, "KERNEL").replace(pathlib.Path(s).stem, "KERNEL_FILE")
            code[label(name)] = hashlib.md5(LABEL.sub("label_X", text).encode()).hexdigest()[:12]
        if set(code) != {"ref", "exp"}:
            print(f"{d.name}: expected one REF and one EXP kernel, found {sorted(map(str, code))}")
        else:
            print(f"{d.name}: ref {code['ref']} exp {code['exp']} "
                  f"{'differ' if code['ref'] != code['exp'] else 'SAME CODE'}")


def rounds(build):
    """Per round: (ref time / exp time - 1) in %, from GFlops (same flops for both)."""
    files = [f for f in glob.glob(f"{build}/2_BenchmarkData/*.csv") if "Granularity" not in f]
    if len(files) != 1:
        return None
    rows = list(csv.reader(open(files[0]), skipinitialspace=True))
    col = {label(h): i for i, h in enumerate(rows[0]) if label(h)}
    if set(col) != {"ref", "exp"}:
        sys.exit(f"{files[0]}: expected one REF and one EXP column, found {sorted(map(str, col))}")
    return [(float(r[col["exp"]]) / float(r[col["ref"]]) - 1) * 100 for r in rows[1:] if r]


def collect(a):
    out = pathlib.Path(a.out)
    names = sorted({p.name[:-4] if p.name.endswith("_rev") else p.name for p in out.iterdir() if p.is_dir()})
    per_round = []
    print(f"{'kernel':32} {'fwd':>7} {'rev':>7} {'all':>7} {'EXP faster':>11}  verdict")
    for name in names:
        res = {f: rounds(out / (name + s)) for f, s in (("fwd", ""), ("rev", "_rev"))}
        res = {f: v for f, v in res.items() if v}
        if not res:
            continue
        for f, v in res.items():
            per_round += [dict(kernel=name, file=f, round=i, speed_pct=round(x, 3)) for i, x in enumerate(v)]
        med = {f: statistics.median(v) for f, v in res.items()}
        both = [x for v in res.values() for x in v]
        m = statistics.median(both)
        signs = {x > 0 for x in med.values()}
        verdict = ("noise" if abs(m) < NOISE_PCT or len(signs) > 1 else "faster" if m > 0 else "slower")
        if len(res) < 2:
            verdict += " (one file)"
        fmt = lambda f: f"{med[f]:+7.2f}" if f in med else f"{'-':>7}"
        print(f"{name:32} {fmt('fwd')} {fmt('rev')} {m:+7.2f} {sum(x > 0 for x in both):>5}/{len(both):<5}  {verdict}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, ["kernel", "file", "round", "speed_pct"])
            w.writeheader()
            w.writerows(per_round)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("exp_dir")
    r.add_argument("out")
    r.add_argument("names", nargs="*")
    r.add_argument("--cpu-only", action="store_true")
    r.add_argument("--check")
    s = sub.add_parser("asm")
    s.add_argument("out")
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("--csv")
    a = ap.parse_args()
    {"run": run, "asm": asm, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
