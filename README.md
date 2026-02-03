# 2D Shallow Water (Conservative) — Serial C + MPI

This folder contains a minimal **serial C** and **MPI-parallel C** implementation for the **2D shallow water equations in conservative form**.

The parallel design follows the reference PDF in this folder: **"2D Shallow water equations - Parallel algorithm.pdf"** (domain splitting into `p` parts; each middle part has two neighbors; exchange boundary values each timestep).

## Problem (brief)
We solve the 2D shallow water equations (SWE) in conservative variables:

- $U = [h,\; hu,\; hv]^T$
- Fluxes:
  - $F(U) = [hu,\; hu^2/h + \tfrac{1}{2}gh^2,\; hu\,hv/h]^T$
  - $G(U) = [hv,\; hu\,hv/h,\; hv^2/h + \tfrac{1}{2}gh^2]^T$

Here `h` is water depth; `u`,`v` are velocities; `g` is gravitational constant.

## Numerical method
We implement the **2D Lax–Friedrichs finite-difference scheme** (matches the paper’s finite-difference LF form, Eq. (3.3) in the extracted text):

$$
U^{n+1}_{i,j} = \tfrac{1}{4}(U^n_{i+1,j}+U^n_{i-1,j}+U^n_{i,j+1}+U^n_{i,j-1})
- \tfrac{\Delta t}{2\Delta x}(F^n_{i+1,j}-F^n_{i-1,j})
- \tfrac{\Delta t}{2\Delta y}(G^n_{i,j+1}-G^n_{i,j-1}).
$$

Boundary conditions: **reflective walls**.

Initial condition: a simple **water hump** at the domain center.

## Method details (Vietnamese)
Mục này mô tả chi tiết đúng “pipeline” mà code đang làm (để đưa vào report/phần trình bày).

### (1) Mô hình toán (theo paper)
Paper viết hệ SWE 2D dạng bảo toàn:

$$
U_t + F(U)_x + G(U)_y = S.
$$

Trong đó:
- $U = [h,\; hu,\; hv]^T$.
- $F(U) = [hu,\; hu^2/h + \tfrac{1}{2}gh^2,\; hu\,hv/h]^T$.
- $G(U) = [hv,\; hu\,hv/h,\; hv^2/h + \tfrac{1}{2}gh^2]^T$.
- $S = [0,\; -gh b_x,\; -gh b_y]^T$ là nguồn do đáy $b(x,y)$.

**Trong code hiện tại**: giả sử đáy phẳng $b(x,y)=0$ nên **$S=0$** (bài toán SWE thuần bảo toàn).

### (2) Rời rạc hoá và công thức cập nhật (LF finite difference)
Lưới đều với bước $\Delta x=DX$, $\Delta y=DY$, thời gian $\Delta t=DT$.

Ở mỗi ô nội bộ $(i,j)$, code dùng công thức LF finite difference 2D (đúng dạng (3.3) paper):

$$
U^{n+1}_{i,j} = \tfrac{1}{4}(U^n_{i+1,j}+U^n_{i-1,j}+U^n_{i,j+1}+U^n_{i,j-1})
- \tfrac{\Delta t}{2\Delta x}(F(U^n_{i+1,j})-F(U^n_{i-1,j}))
- \tfrac{\Delta t}{2\Delta y}(G(U^n_{i,j+1})-G(U^n_{i,j-1})).
$$

Diễn giải:
- Hạng $\tfrac{1}{4}(\cdot)$ là **trung bình 4 láng giềng** (tạo khuếch tán số → ổn định hơn nhưng “diffusive”).
- Hai hạng sau là **sai phân trung tâm** của thông lượng theo x/y.

**Lưu ý triển khai**:
- Flux được tính trong `flux_F()` và `flux_G()`.
- Để tránh chia cho 0 khi $h$ rất nhỏ, code dùng `safe_h()` (clamp $h$ tối thiểu) trước khi tính $u=hu/h$, $v=hv/h$.
- Có guard: nếu $h_{new}$ âm/nhỏ, clamp về `1e-10` và đặt `hu_new=hv_new=0` để tránh nổ số.

### (3) Điều kiện biên (reflective wall)
Code đang dùng **reflective boundary condition** (tường phản xạ):
- Ở biên theo **y** (trên/dưới): đảo dấu vận tốc pháp tuyến theo y → **đảo dấu `hv`**, còn `hu` giữ nguyên.
- Ở biên theo **x** (trái/phải): đảo dấu vận tốc pháp tuyến theo x → **đảo dấu `hu`**, còn `hv` giữ nguyên.
- `h` được copy từ ô bên trong ra ghost/biên.

Trong paper có nhắc transmissive/reflective; bản nộp này dùng **reflective** để dễ kiểm soát và đúng với demo trong code.

### (4) Ổn định (CFL note)
Paper đưa dạng điều kiện ổn định cần (necessary stability condition) cho explicit schemes.
Trong code hiện tại, `DT` là tham số nhập vào (mặc định `0.01`) và không tự động điều chỉnh theo CFL.
Khi chạy case lớn, nếu thấy dao động/nổ số: giảm `DT` hoặc tăng độ khuếch tán (LF vốn đã diffusive nên thường ổn định hơn LW/MC).

### (5) Luồng thuật toán (pseudo)
**Serial** và **MPI** dùng cùng update LF; khác nhau ở bước trao đổi halo.

Pseudo 1 timestep:
1. (MPI only) exchange ghost columns (halo) cho `h,hu,hv`
2. apply boundary conditions (reflective)
3. for i,j interior: compute fluxes F/G từ láng giềng và cập nhật LF → `*_new`
4. apply boundary conditions cho `*_new`
5. swap pointers (`h <-> h_new`, ...)

## Parallel design (MPI)
- **1D domain decomposition in x**: the global grid `NX × NY` is split into `p` slabs; each rank owns `mc = NX/p` columns.
- Each rank allocates **2 ghost columns** (left/right) and exchanges them every timestep using `MPI_Sendrecv`.
- Reflective boundary conditions are applied on the **global x-walls** (rank 0 and rank `p-1`) and on **all y-walls**.

### MPI communication detail (what is sent)
Vì chia miền theo x, mỗi rank cần biết **1 cột bên trái** và **1 cột bên phải** để cập nhật stencil LF.
Mỗi cột là vector dài `NY` (toàn bộ theo y) cho từng trường `h`, `hu`, `hv`.

Trong code, mỗi timestep có 2 lần `MPI_Sendrecv` cho mỗi trường:
- gửi cột `i=1` sang rank trái và nhận vào ghost phải `i=mc+1`
- gửi cột `i=mc` sang rank phải và nhận vào ghost trái `i=0`

Điều này đúng với mô tả trong paper: với domain splitting 1D, mỗi phần giữa chỉ có 2 láng giềng → giao tiếp đơn giản.

## Files
- [shallow_water_conservative_serial.c](shallow_water_conservative_serial.c): baseline serial solver.
- [shallow_water_conservative_serial_lw.c](shallow_water_conservative_serial_lw.c): serial Lax–Wendroff solver.
- [shallow_water_conservative_serial_mc.c](shallow_water_conservative_serial_mc.c): serial MacCormack solver.
- [shallow_water_conservative_mpi.c](shallow_water_conservative_mpi.c): MPI solver (parallel code).
- [shallow_water_conservative_mpi_lw.c](shallow_water_conservative_mpi_lw.c): MPI Lax–Wendroff solver.
- [shallow_water_conservative_mpi_mc.c](shallow_water_conservative_mpi_mc.c): MPI MacCormack solver.
- [Makefile](Makefile): build both binaries.
- [scripts](scripts/): build/run/verify/bench scripts.
- [outputs](outputs/): generated CSV outputs.
- [results](results/): benchmark CSVs.
- [parallel_algorithm_extracted.txt](parallel_algorithm_extracted.txt): rough plaintext extracted from the provided PDF (for quick searching).

## Build
From this folder:

Preferred (does not require `make`):

- `bash scripts/build.sh`

Optional (requires `make` installed):

- `make -j`

If scripts are not executable: `chmod +x scripts/*.sh`.

## Run (serial)
Example:

- `./sw_serial --nx 200 --ny 200 --steps 400 --dt 0.01 --out outputs/h_serial.csv`

Compute-only (skip CSV I/O, for fair timing):

- `./sw_serial --nx 200 --ny 200 --steps 400 --dt 0.01 --no-output`

Or via script:

- `NX=200 NY=200 STEPS=400 DT=0.01 OUT=outputs/h_serial.csv bash scripts/run_serial.sh`

Other serial schemes:
- `NX=200 NY=200 STEPS=400 DT=0.01 OUT=outputs/h_serial_lw.csv bash scripts/run_serial_lw.sh`
- `NX=200 NY=200 STEPS=400 DT=0.01 OUT=outputs/h_serial_mc.csv bash scripts/run_serial_mc.sh`

All run scripts support `NO_OUTPUT=1` to pass `--no-output`:

- `NX=400 NY=400 STEPS=200 DT=0.005 NO_OUTPUT=1 bash scripts/run_serial.sh`

Unified serial binary (choose scheme at runtime):
- `./sw_serial_schemes --scheme lf --nx 200 --ny 200 --steps 400 --dt 0.01 --out outputs/h_lf.csv`
- `./sw_serial_schemes --scheme lw --nx 200 --ny 200 --steps 400 --dt 0.01 --out outputs/h_lw.csv`
- `./sw_serial_schemes --scheme mc --nx 200 --ny 200 --steps 400 --dt 0.01 --out outputs/h_mc.csv`

## Run (MPI)
Example:

- `mpirun -np 4 ./sw_mpi --nx 200 --ny 200 --steps 400 --dt 0.01 --out outputs/h_mpi.csv`

Compute-only (skip gather + CSV I/O, for fair timing):

- `mpirun -np 4 ./sw_mpi --nx 200 --ny 200 --steps 400 --dt 0.01 --no-output`

Or via script:

- `NP=4 NX=200 NY=200 STEPS=400 DT=0.01 OUT=outputs/h_mpi.csv bash scripts/run_mpi.sh`

All MPI run scripts support `NO_OUTPUT=1`:

- `NP=4 NX=800 NY=800 STEPS=200 DT=0.005 NO_OUTPUT=1 bash scripts/run_mpi.sh`

Notes:
- `NX` must be divisible by `NP` (current decomposition).

## Quick correctness check (serial vs MPI)
Runs a small case and compares output CSVs:

- `bash scripts/verify.sh`

Environment variables (optional):
- `NP` (default `4`), `NX` (default `120`), `NY` (default `120`), `STEPS` (default `50`)
- `DT`, `DX`, `DY`, `G`
- `EPS` tolerance for max absolute difference (default `1e-12`)

Pass/Fail criteria:
- Script prints CSV sanity metrics (shape/min/max/sum) for both outputs.
- Script prints `max_abs_diff` / `mean_abs_diff` / `max_rel_diff` and exits non-zero if `max_abs_diff >= EPS`.

Example:
- `NP=4 NX=120 NY=120 STEPS=50 EPS=1e-12 bash scripts/verify.sh`

Artifacts:
- `outputs/verify_serial.csv`
- `outputs/verify_mpi.csv`

## Test suite (recommended for grading)
Runs build + multiple correctness cases + sanity checks + a quick benchmark:

- `bash scripts/test_all.sh`

This is the recommended “one command” to validate the submission.

## Unified correctness suite (LF/LW/MC)
Runs serial vs MPI comparisons for all three schemes under the same IC/BC and the same `DT`/`STEPS`:

- `GRID_LIST="80x60 120x80" NP_LIST="1 2 4" STEPS=50 DT=0.01 EPS=1e-12 bash scripts/unified_correctness.sh`

Artifacts:
- `outputs/correctness_<timestamp>/scheme/NxM/{serial.csv,mpi_npK.csv}`

## Benchmark / experiments
Runs serial baseline and MPI for a list of process counts, writing a long-format CSV under [results](results/).
The runtime is measured using `CLOCK_MONOTONIC` (serial) and the global max across ranks using `CLOCK_MONOTONIC` (MPI).

- `NX=400 NY=400 STEPS=200 DT=0.005 NP_LIST="1 2 4 8" BENCH_LW=1 BENCH_MC=1 NO_OUTPUT=1 bash scripts/bench.sh`

Output CSV columns:
`scheme,impl,np,nx,ny,steps,dt,dx,dy,g,time_s,t_seq_s,speedup,efficiency,hmin,hmax,no_output`

## Grid sweep 200x200 → 1800x1800 (organized folders)
Runs multiple grid sizes (square grids) and produces one combined CSV for plotting runtime/speedup/efficiency:

- `NP_LIST="1 2 4" STEPS=200 DT=0.005 NO_OUTPUT=1 bash scripts/sweep_200_1800.sh`

Artifacts:
- `results/sweep_<timestamp>/bench_long.csv`
- `results/sweep_<timestamp>/table_{lf,lw,mc}.csv`

Common overrides:
- `MIN_N`, `MAX_N`, `STEP_N`
- `STEPS`, `DT`, `NP_LIST` (non-divisible `NX%NP!=0` runs are skipped)

## Packaging requirement reminder
When submitting, place **Report / Slides / Code / README** in one `.zip/.tar` and name it as:

`GroupNumber_Problem_ParallelModel`

Example: `Group1_ShallowWater_MPI`.
