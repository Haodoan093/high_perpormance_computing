# Slides Outline (15 minutes) — 2D Shallow Water (MPI)

## 1. Title
- 2D Shallow Water Equations (Conservative) — Serial C + MPI
- Group: ___

## 2. Problem
- What is SWE and why it matters
- Goal: simulate + accelerate with MPI

## 3. Mathematical model
- Conservative variables $U=[h,hu,hv]^T$
- Fluxes $F(U)$ and $G(U)$

## 4. Numerical scheme
- 2D Lax–Friedrichs finite difference update
- Reflective boundary conditions

## 5. Parallel design
- Domain decomposition in x into `p` slabs
- Halo exchange each timestep (neighbor-only)

## 6. Implementation
- Data layout + ghost columns
- Communication with `MPI_Sendrecv`
- Output gathering with `MPI_Gather`

## 7. Experiments
- Benchmark setup: grid sizes, steps, dt
- Runtime results + speedup plot (from `results/bench_*.csv`)

## 8. Conclusions
- What worked
- Limitations and next improvements (2D decomposition, better flux, CFL control)
