# Report — 2D Shallow Water Equations (Conservative Form) with MPI

## 1. Problem specification
Goal: simulate 2D shallow-water dynamics on a rectangular grid using an explicit conservative scheme, and accelerate runtime via MPI parallelization.

We focus on:
- Serial baseline (C)
- Distributed-memory parallel version (MPI, C)

Reference algorithm + parallelization concept is taken from the provided document in this folder: **"2D Shallow water equations - Parallel algorithm.pdf"** (especially the domain-splitting + boundary exchange idea in the parallelization section).

## 2. Mathematical formulation
We solve the 2D shallow water equations in conservative variables:

- $U = [h,\; hu,\; hv]^T$

Fluxes:

- $F(U) = \begin{bmatrix} hu \\ \frac{(hu)^2}{h} + \tfrac{1}{2}gh^2 \\ \frac{(hu)(hv)}{h} \end{bmatrix}$
- $G(U) = \begin{bmatrix} hv \\ \frac{(hu)(hv)}{h} \\ \frac{(hv)^2}{h} + \tfrac{1}{2}gh^2 \end{bmatrix}$

The PDE is:

$$
\partial_t U + \partial_x F(U) + \partial_y G(U) = 0.
$$

## 3. Numerical method / algorithm
We implement the **2D Lax–Friedrichs finite-difference scheme** (explicit, first order, diffusive but robust):

$$
U^{n+1}_{i,j} = \tfrac{1}{4}(U^n_{i+1,j}+U^n_{i-1,j}+U^n_{i,j+1}+U^n_{i,j-1})
- \tfrac{\Delta t}{2\Delta x}(F(U^n_{i+1,j})-F(U^n_{i-1,j}))
- \tfrac{\Delta t}{2\Delta y}(G(U^n_{i,j+1})-G(U^n_{i,j-1})).
$$

Implementation details:
- We store conservative fields `h, hu, hv`.
- To avoid division by zero, we clamp depth with `safe_h()`.
- Reflective boundary conditions:
  - At y-walls: reflect `v` → flip sign of `hv`.
  - At x-walls: reflect `u` → flip sign of `hu`.

## 4. Parallel design (MPI)
Following the reference PDF’s approach:

- Split the global domain into `p` parts along x.
- Each rank owns `mc = NX/p` columns, plus 2 ghost columns.
- Each timestep:
  1. Exchange ghost columns (left/right neighbors) for `h, hu, hv`.
  2. Apply boundary conditions (global x-walls + y-walls).
  3. Update interior cells using the LF scheme.

Communication:
- Neighbor-only exchange with `MPI_Sendrecv` (avoids deadlock).

## 5. Implementation and experiments
### Code deliverables
- Serial baseline: [../shallow_water_conservative_serial.c](../shallow_water_conservative_serial.c)
- MPI solver: [../shallow_water_conservative_mpi.c](../shallow_water_conservative_mpi.c)
- Build system: [../Makefile](../Makefile)
- Scripts: [../scripts](../scripts)

### How to run experiments
Use benchmark script:

- `NX=400 NY=400 STEPS=200 DT=0.005 NP_LIST="1 2 4 8" bash scripts/bench.sh`

The script writes a CSV under `results/` with runtime and `h min/max`.

### Metrics
- Wall-clock runtime (serial vs MPI)
- Optional derived speedup: $S(p) = T(1)/T(p)$

## 6. Notes / limitations
- Current decomposition requires `NX % NP == 0`.
- LF scheme is stable but diffusive; it is used here because it matches the reference’s presented explicit schemes and is straightforward to parallelize.

## 7. References
- Provided in-folder document: **"2D Shallow water equations - Parallel algorithm.pdf"**
