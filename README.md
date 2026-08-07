# MFPC: Maximum Flow with Conflicts — Compact MILPs vs. Single-Tree Benders

This repo solves the Maximum Flow Problem with Conflicts (MFPC) three different ways in Python/Gurobi, following Şuvak, Altınel & Aras (2020, *EJOR* 287:410–437). MFPC is ordinary max-flow with a twist: certain pairs of arcs are mutually exclusive and can never both carry flow.

Code and experiments of my bachelor's thesis in Management Engineering at Politecnico di Torino (supervisor: Prof. Federico Della Croce).

The main part of the work is an optimization of the slowest of the three methods, the Benders decomposition. This README reports what was changed and what was measured, including the attempts that did not work.

## Summary

- Three exact solvers: `strong.py`, `weak.py` (compact MILPs), `benders.py` (decomposition).
- The decomposition was 6–8x *slower* than the compact formulations out of the box, on modern Gurobi.
- `benders_v2.py` — nine targeted changes, benchmarked individually and in combination — gets it to **1.6x faster than the original decomposition**, consistently.
- One specific regime (high conflict density) was found where it actually **beats the compact formulation by 46%** — with a Gurobi-internals explanation for why that regime exists and why the general case doesn't close.
- Two architectural rewrites were tried and killed. They're still in the repo, with a note explaining why.

## The three solvers

| | How it decides which arcs to activate | How it routes flow |
|---|---|---|
| `strong.py` | One constraint per conflicting *pair* of arcs | Full flow-conservation MILP, solved in one shot |
| `weak.py` | One constraint per arc, aggregating *all* its conflicts | Full flow-conservation MILP, solved in one shot |
| `benders.py` / `benders_v2.py` | A small master problem, refined iteratively | A max-flow subproblem, called on demand, feeding cuts back into the master |

The decomposition's whole premise is that solving a small master repeatedly should beat solving one big MILP. On the 2016-era solver the original paper used, it did. On Gurobi 13.0.2 in 2026, it mostly doesn't — Gurobi's automatic clique-cut generation on the conflict structure gives the "solve it all at once" formulations a much bigger edge than it used to. That gap is what this repo spends its time closing.

## The numbers that matter

**Optimized vs. original decomposition** (thesis reference parameters, `p=0.6, d=0.3`, same hardware):

| n | `benders.py` | `benders_v2.py` | Speedup |
|---|---|---|---|
| 40 | 29.6 s | 17.87 s | **1.66x** |
| 60 | 1058.5 s | 652.0 s | **1.62x** |

**The regime where decomposition wins outright** — sweeping conflict density `d` at real network density `p=0.6`:

| n | d | Weak | `benders_v2.py --no-strong-master --cuts 0` |
|---|---|---|---|
| 60 | 0.3 | fast | Benders behind |
| 60 | **0.7** | 18.42 s | **12.64 s — Benders 1.46x faster** |
| 60 | 0.85 | 11.77 s | Benders slightly behind again |

Weak's conflict handling aggregates every conflicting pair into one constraint per arc; the more conflicts there are, the looser that aggregate gets as a relaxation. Benders cuts come from a real max-flow computation every time, so they don't degrade the same way — hence the crossover at high `d`. It isn't monotonic (`d=0.85` loses again), and it's measured, not proven, but it's real and reproducible. Full sweep with every data point in [`BENCHMARKS.md`](BENCHMARKS.md).

## Nine changes, benchmarked one at a time

`benders_v2.py` ships all of these; most are off by default because they didn't hold up outside the specific case they were found in.

1. **Precomputed max-flow subproblem** (`maxflow_fast.py`) — the subproblem's network topology is static across the whole run, only capacities change per call. Rebuilding a sparse matrix from a dict every call (the original approach) turned out to be pure waste. *On by default.*
2. **Pairwise conflict constraints in the master**, not just the aggregated ones — gives Gurobi's own presolve the clique structure it wants. *On by default.*
3. **Cuts injected at fractional B&B nodes**, not just integer solutions — provably valid (it's a fixed cut in the network topology, independent of which `x` produced it), but empirically counter-productive once combined with #2. *Off by default.*
4. **Batched variable reads** in the callback instead of one Gurobi API call per arc.
5. **Heuristic warm start** from a greedy conflict-respecting construction. *On by default.*
6. **Static cuts injected before the search starts** (`--extra-preseed N`), not during — sidesteps the "changes the search path" risk of lazy cuts. Helped at n=40, hurt at n=60. *Off by default, N=0.*
7. **Gurobi's own tuning knobs exposed** (`--cuts`, `--mipfocus`, `--threads`, `--presolve`, `--heuristics`). `--cuts 0` alone was the single best lever found at one scale — and one of the worst at another.
8. **Branch priority** on high-conflict-degree arcs. Tested, measured no effect.
9. Two full architectural alternatives — see below.

Every number behind every one of these claims: [`BENCHMARKS.md`](BENCHMARKS.md).

## Two architectures that didn't make it

- **`benders_v3.py`** — solve the master's LP relaxation up front for a tighter starting bound, and inject a second "free" cut per callback alongside the main one. Both ideas are individually sound. Both made things worse on real instances: the tighter bound cost 80 extra seconds of setup for a bound Gurobi's own presolve reached anyway, and the free cut slowed the search down despite costing nothing to compute.
- **`benders_v4.py`** — go back to *classical* multi-tree Benders (re-solve the master from scratch after every cut) specifically to dodge a Gurobi quirk described below. Wins when only a few dozen re-solves are needed. At real conflict density, hundreds to thousands are needed, each one an expensive from-scratch MILP solve — a timed run got through 38 iterations in 300 seconds without converging.

Both are kept in the repo with a note at the top pointing here, rather than deleted — the negative result is as informative as the positive one.

## The wall that tuning can't get past

Gurobi's `LazyConstraints=1` — required for `cbLazy`, which is how every version of the Benders master injects cuts — **disables the solver's dual-reduction presolve for the entire run**, unconditionally, regardless of any other parameter. It's not a missed setting: dual reductions need to know the full feasible region up front, and with lazy constraints that region is still being discovered live. Compact formulations never trip this; the decomposition, by construction, always does. It's a smaller-scale version of the same trade-off the underlying thesis already documents for the original (Xpress/Mosel) implementation of this algorithm, which had to disable presolve entirely to inject cuts safely. ([Gurobi: What are dual reductions?](https://support.gurobi.com/hc/en-us/articles/27875045953041-What-are-dual-reductions))

## What's in here

```
generatore.py        instance generator
common.py             shared loading + scipy max-flow helper
strong.py, weak.py    the two compact MILP formulations
benders.py            original decomposition (baseline)
benders_v2.py         optimized decomposition — start here
benders_v3.py         rejected variant, kept for the record
benders_v4.py         rejected variant, kept for the record
maxflow_fast.py       precomputed max-flow solver used by v2
benders.cpp           C++ reimplementation (ruled out "Python is slow")
automazione.py        batch-runs every solver across a scenario list, logs everything
BENCHMARKS.md         every experiment, every number
```

## Running it

```bash
pip install gurobipy numpy scipy

python generatore.py --n 60 --p 0.6 --d 0.3 --seed 12345 --dir istanza
python strong.py --dir istanza --timelimit 3600
python weak.py --dir istanza --timelimit 3600
python benders_v2.py --dir istanza --maxtime 3600
```

High-conflict-density regime, where the decomposition wins:

```bash
python generatore.py --n 60 --p 0.6 --d 0.7 --seed 12345 --dir istanza_hi
python weak.py --dir istanza_hi --timelimit 3600
python benders_v2.py --dir istanza_hi --maxtime 3600 --no-strong-master --cuts 0
```

Or batch everything at once — edit the scenario list at the top of `automazione.py`, then:

```bash
python automazione.py
```

## Caveats

- Everything here was measured on one machine with a handful of random seeds. The same instance, same seed, run on a different machine, gave measurably different relative timings between methods during this work — Branch-and-Bound with lazy constraints is genuinely sensitive to that. Treat absolute numbers as illustrative, not universal.
- The `d≈0.7` crossover point is an empirical observation, not a derived threshold, and it moved between instance sizes tested. There's a regime where this wins; there isn't a formula for exactly where it starts.
- `benders_v2.py` with default settings is the only configuration that never made things worse across every test performed. Everything past that is a real, documented, situational trade-off — not a universal upgrade.

## Reference

Şuvak, Z., Altınel, İ. K., & Aras, N. (2020). Exact solution algorithms for the maximum flow problem with additional conflict constraints. *European Journal of Operational Research*, 287(2), 410–437.