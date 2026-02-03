#define _POSIX_C_SOURCE 199309L

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
  ============================================================
  2D SHALLOW WATER EQUATIONS (SWE) - MPI, CONSERVATIVE FORM
  ============================================================

  Biến bảo toàn (conservative variables):
    U = [ h, hu, hv ]^T
    - h  : chiều cao mực nước
    - hu : động lượng theo x (h*u)
    - hv : động lượng theo y (h*v)

  Lưới:
    - Miền vật lý: NX x NY (cell-centered)
    - Song song hoá: chia miền theo phương x (cột i) thành 'size' khối liên tiếp
        + Mỗi rank giữ mc = NX/size cột vật lý
        + Cộng thêm 2 cột ghost theo x: i = 0 và i = mc+1 (local index)
        + Theo y: không cấp ghost riêng; dùng j=0 và j=NY-1 làm biên phản xạ (wall)

  Thuật toán số (giống bản serial):
    - Lax–Friedrichs 2D:
        U^{n+1}_{i,j} =
            1/4 (U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1})
          - dt/(2dx) (F(U_{i+1,j}) - F(U_{i-1,j}))
          - dt/(2dy) (G(U_{i,j+1}) - G(U_{i,j-1}))
    - Mỗi timestep:
        (1) Halo exchange theo x cho h,hu,hv để điền ghost columns
        (2) Apply reflective BC cho biên vật lý (y-walls và x-walls tại rank biên)
        (3) Update nội bộ i=1..mc, j=1..NY-2
        (4) Apply BC lên mảng *_new
        (5) Swap con trỏ

  Ghi chú:
    - Code hiện yêu cầu NX chia hết cho số process (NX % size == 0).
    - Output: gather h về rank 0, ghi CSV.
*/

/* ================= DEFAULT PARAMETERS ================= */
#define DEFAULT_NX 200
#define DEFAULT_NY 200
#define DEFAULT_NSTEPS 400
#define DEFAULT_DX 1.0
#define DEFAULT_DY 1.0
#define DEFAULT_DT 0.01
#define DEFAULT_G 9.81

/*
  Indexing:
  - Mảng 2D được trải phẳng 1D theo dạng [i][j]
  - ny là stride theo chiều j
  - i là local x-index (0..mc+1), j là y-index (0..NY-1)
*/
static inline size_t IDX(int i, int j, int ny)
{
    return (size_t)i * (size_t)ny + (size_t)j;
}

/* Tham số chạy từ command line */
typedef struct Params
{
    int nx;
    int ny;
    int nsteps;
    double dx;
    double dy;
    double dt;
    double g;
    char out_csv[256];
    int write_output;
} Params;

static void params_init_default(Params *p)
{
    p->nx = DEFAULT_NX;
    p->ny = DEFAULT_NY;
    p->nsteps = DEFAULT_NSTEPS;
    p->dx = DEFAULT_DX;
    p->dy = DEFAULT_DY;
    p->dt = DEFAULT_DT;
    p->g = DEFAULT_G;
    snprintf(p->out_csv, sizeof(p->out_csv), "h_final.csv");
    p->write_output = 1;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--nx N] [--ny N] [--steps N] [--dt DT] [--dx DX] [--dy DY] [--g G] [--out FILE] [--no-output]\n",
            prog);
}

/* Parse tham số dòng lệnh */
static int parse_args(int argc, char **argv, Params *p)
{
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc)
            p->nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ny") == 0 && i + 1 < argc)
            p->ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc)
            p->nsteps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
            p->dt = atof(argv[++i]);
        else if (strcmp(argv[i], "--dx") == 0 && i + 1 < argc)
            p->dx = atof(argv[++i]);
        else if (strcmp(argv[i], "--dy") == 0 && i + 1 < argc)
            p->dy = atof(argv[++i]);
        else if (strcmp(argv[i], "--g") == 0 && i + 1 < argc)
            p->g = atof(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc)
        {
            snprintf(p->out_csv, sizeof(p->out_csv), "%s", argv[++i]);
        }
        else if (strcmp(argv[i], "--no-output") == 0)
        {
            p->write_output = 0;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown/invalid argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }

    /* Check đơn giản để tránh giá trị vô nghĩa */
    if (p->nx <= 2 || p->ny <= 2 || p->nsteps < 0)
        return -1;
    if (p->dx <= 0.0 || p->dy <= 0.0 || p->dt <= 0.0 || p->g <= 0.0)
        return -1;
    return 1;
}

/* Timer monotonic để đo runtime (compute-only friendly) */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Tránh chia cho 0 khi h rất nhỏ */
static inline double safe_h(double h)
{
    const double eps = 1e-12;
    return (h > eps) ? h : eps;
}

/*
  Reflective BC (tường phản xạ):
    - Biên theo y (j=0 và j=NY-1): vận tốc pháp tuyến theo y đổi dấu => hv đổi dấu
    - Biên theo x toàn cục:
        + rank=0 (biên trái): ghost i=0 phản xạ từ i=1 => hu đổi dấu
        + rank=size-1 (biên phải): ghost i=mc+1 phản xạ từ i=mc => hu đổi dấu
    - h được copy từ ô bên trong ra ghost/biên
*/
static void apply_bc(int rank, int size, int mc, int ny, double *h, double *hu, double *hv)
{
    /* Biên y áp dụng cho mọi rank và mọi cột local (kể cả ghost) */
    for (int i = 0; i < mc + 2; i++)
    {
        // bottom wall (j=0)
        h[IDX(i, 0, ny)] = h[IDX(i, 1, ny)];
        hu[IDX(i, 0, ny)] = hu[IDX(i, 1, ny)];
        hv[IDX(i, 0, ny)] = -hv[IDX(i, 1, ny)];

        // top wall (j=NY-1)
        h[IDX(i, ny - 1, ny)] = h[IDX(i, ny - 2, ny)];
        hu[IDX(i, ny - 1, ny)] = hu[IDX(i, ny - 2, ny)];
        hv[IDX(i, ny - 1, ny)] = -hv[IDX(i, ny - 2, ny)];
    }

    /* Biên x CHỈ áp dụng ở 2 đầu miền toàn cục (rank biên) */
    if (rank == 0)
    {
        for (int j = 0; j < ny; j++)
        {
            h[IDX(0, j, ny)] = h[IDX(1, j, ny)];
            hu[IDX(0, j, ny)] = -hu[IDX(1, j, ny)];
            hv[IDX(0, j, ny)] = hv[IDX(1, j, ny)];
        }
    }

    if (rank == size - 1)
    {
        for (int j = 0; j < ny; j++)
        {
            h[IDX(mc + 1, j, ny)] = h[IDX(mc, j, ny)];
            hu[IDX(mc + 1, j, ny)] = -hu[IDX(mc, j, ny)];
            hv[IDX(mc + 1, j, ny)] = hv[IDX(mc, j, ny)];
        }
    }
}

/*
  Flux theo x:
    F(U) = [ hu,
             hu*u + 1/2*g*h^2,
             hu*v ]
*/
static inline void flux_F(double h, double hu, double hv, double g,
                          double *F1, double *F2, double *F3)
{
    const double hs = safe_h(h);
    const double u = hu / hs;
    const double v = hv / hs;
    *F1 = hu;
    *F2 = hu * u + 0.5 * g * hs * hs;
    *F3 = hu * v;
}

/*
  Flux theo y:
    G(U) = [ hv,
             hv*u,
             hv*v + 1/2*g*h^2 ]
*/
static inline void flux_G(double h, double hu, double hv, double g,
                          double *G1, double *G2, double *G3)
{
    const double hs = safe_h(h);
    const double u = hu / hs;
    const double v = hv / hs;
    *G1 = hv;
    *G2 = hv * u;
    *G3 = hv * v + 0.5 * g * hs * hs;
}

/* Exchange left/right ghost columns for one field (count = ny). */
static void halo_exchange_x_field(int mc, int ny, double *field,
                                  int left, int right, int tag_base)
{
    MPI_Status st;

    /* send i=1 to left, recv into i=mc+1 from right (same tag) */
    MPI_Sendrecv(&field[IDX(1, 0, ny)], ny, MPI_DOUBLE, left, tag_base,
                 &field[IDX(mc + 1, 0, ny)], ny, MPI_DOUBLE, right, tag_base,
                 MPI_COMM_WORLD, &st);

    /* send i=mc to right, recv into i=0 from left (different tag) */
    MPI_Sendrecv(&field[IDX(mc, 0, ny)], ny, MPI_DOUBLE, right, tag_base + 1,
                 &field[IDX(0, 0, ny)], ny, MPI_DOUBLE, left, tag_base + 1,
                 MPI_COMM_WORLD, &st);
}

/* Halo exchange for all conservative fields (h,hu,hv). */
static void halo_exchange_x(int mc, int ny, double *h, double *hu, double *hv,
                            int left, int right)
{
    halo_exchange_x_field(mc, ny, h, left, right, 10);
    halo_exchange_x_field(mc, ny, hu, left, right, 20);
    halo_exchange_x_field(mc, ny, hv, left, right, 30);
}

/*
  One explicit LF finite-difference step (paper Eq. 3.3) on interior local cells.
  - Updates i = 1..mc (physical columns), j = 1..ny-2 (avoid y-walls).
  - Assumes ghost columns in (h,hu,hv) are already valid.
*/
static void update_lf_step_local(int mc, int ny,
                                 double dx, double dy, double dt, double g,
                                 const double *h, const double *hu, const double *hv,
                                 double *h_new, double *hu_new, double *hv_new)
{
    for (int i = 1; i <= mc; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            const size_t c = IDX(i, j, ny);
            const size_t ip = IDX(i + 1, j, ny);
            const size_t im = IDX(i - 1, j, ny);
            const size_t jp = IDX(i, j + 1, ny);
            const size_t jm = IDX(i, j - 1, ny);

            double F1p, F2p, F3p, F1m, F2m, F3m;
            double G1p, G2p, G3p, G1m, G2m, G3m;

            flux_F(h[ip], hu[ip], hv[ip], g, &F1p, &F2p, &F3p);
            flux_F(h[im], hu[im], hv[im], g, &F1m, &F2m, &F3m);
            flux_G(h[jp], hu[jp], hv[jp], g, &G1p, &G2p, &G3p);
            flux_G(h[jm], hu[jm], hv[jm], g, &G1m, &G2m, &G3m);

            h_new[c] =
                0.25 * (h[ip] + h[im] + h[jp] + h[jm]) - (dt / (2.0 * dx)) * (F1p - F1m) - (dt / (2.0 * dy)) * (G1p - G1m);
            hu_new[c] =
                0.25 * (hu[ip] + hu[im] + hu[jp] + hu[jm]) - (dt / (2.0 * dx)) * (F2p - F2m) - (dt / (2.0 * dy)) * (G2p - G2m);
            hv_new[c] =
                0.25 * (hv[ip] + hv[im] + hv[jp] + hv[jm]) - (dt / (2.0 * dx)) * (F3p - F3m) - (dt / (2.0 * dy)) * (G3p - G3m);

            if (h_new[c] < 1e-10)
            {
                h_new[c] = 1e-10;
                hu_new[c] = 0.0;
                hv_new[c] = 0.0;
            }
        }
    }
}

/* ================= MAIN ================= */
int main(int argc, char *argv[])
{
    int rank, size;

    /* Khởi tạo MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Đọc tham số */
    Params p;
    params_init_default(&p);
    int parse_rc = parse_args(argc, argv, &p);
    if (parse_rc <= 0)
    {
        MPI_Finalize();
        return (parse_rc == 0) ? 0 : 1;
    }

    const int NX = p.nx;
    const int NY = p.ny;
    const int NSTEPS = p.nsteps;
    const double DX = p.dx;
    const double DY = p.dy;
    const double DT = p.dt;
    const double G = p.g;

    /* Ràng buộc: hiện tại code chia miền theo x đều nhau => NX phải chia hết size */
    if (NX % size != 0)
    {
        if (rank == 0)
            printf("NX must be divisible by number of processes\n");
        MPI_Finalize();
        return 0;
    }

    /* mc = số cột vật lý mỗi rank giữ */
    int mc = NX / size;

    /*
      Cấp phát các biến bảo toàn:
        - Mỗi mảng có (mc+2) x NY
        - i=0 và i=mc+1 là ghost columns dùng cho halo exchange / biên toàn cục
    */
    double *h, *hu, *hv;
    double *h_new, *hu_new, *hv_new;

    h = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));
    hu = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));
    hv = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));

    h_new = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));
    hu_new = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));
    hv_new = (double *)malloc((size_t)(mc + 2) * (size_t)NY * sizeof(double));

    if (!h || !hu || !hv || !h_new || !hu_new || !hv_new)
    {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ================= INITIAL CONDITION =================
       - Nước tĩnh: h=1, hu=hv=0
       - Thêm 1 "hump" ở trung tâm miền toàn cục để tạo sóng lan truyền
    */
    for (int i = 0; i < mc + 2; i++)
        for (int j = 0; j < NY; j++)
        {
            h[IDX(i, j, NY)] = 1.0;
            hu[IDX(i, j, NY)] = 0.0;
            hv[IDX(i, j, NY)] = 0.0;
        }

    /* Hump tại global_center = NX/2
       - Xác định rank chứa điểm này và local index tương ứng
       - local i chạy 1..mc (0 và mc+1 là ghost)
    */
    int global_center = NX / 2;
    if (rank == global_center / mc)
    {
        int local_i = global_center % mc + 1;
        h[IDX(local_i, NY / 2, NY)] = 2.0;
    }

    /* Áp BC để ghost/biên hợp lệ trước khi vào time loop */
    apply_bc(rank, size, mc, NY, h, hu, hv);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = now_seconds();

    /* Rank láng giềng: dùng MPI_PROC_NULL ở biên để Sendrecv “an toàn” */
    const int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* ================= TIME LOOP ================= */
    for (int step = 0; step < NSTEPS; step++)
    {
        /* 1) HALO EXCHANGE theo x (fill ghost columns) */
        halo_exchange_x(mc, NY, h, hu, hv, left, right);

        /* 2) Apply BC:
              - y-walls cho mọi rank
              - x-walls chỉ ở rank 0 và rank size-1
           Sau bước này, ghost columns và hàng biên y đều “đúng vật lý”.
        */
        apply_bc(rank, size, mc, NY, h, hu, hv);

        /* 3) NUMERICAL UPDATE (LF finite difference, paper Eq. 3.3) */
        update_lf_step_local(mc, NY, DX, DY, DT, G, h, hu, hv, h_new, hu_new, hv_new);

        /* 4) Apply BC trên nghiệm mới để bước sau dùng ghost/biên đúng */
        apply_bc(rank, size, mc, NY, h_new, hu_new, hv_new);

        /* 5) SWAP (đổi con trỏ để tránh copy) */
        double *tmp;
        tmp = h;
        h = h_new;
        h_new = tmp;
        tmp = hu;
        hu = hu_new;
        hu_new = tmp;
        tmp = hv;
        hv = hv_new;
        hv_new = tmp;
    }

    const double t1 = now_seconds();
    const double local_elapsed = t1 - t0;

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* ---------- SANITY CHECK: min/max toàn cục ---------- */
    double local_min = 1e300, local_max = -1e300;
    for (int i = 1; i <= mc; i++)
        for (int j = 0; j < NY; j++)
        {
            double val = h[IDX(i, j, NY)];
            if (val < local_min)
                local_min = val;
            if (val > local_max)
                local_max = val;
        }

    double global_min = 0.0, global_max = 0.0;
    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /*
      ---------- OUTPUT: gather h về rank 0 ----------
      - Mỗi rank gửi miền vật lý local (i=1..mc) => mc*NY phần tử
      - Rank 0 nhận thành mảng NX*NY theo thứ tự rank 0..size-1 (liên tiếp theo x)
    */
    if (rank == 0)
    {
        printf("MPI Conservative Shallow Water finished in %f s\n", max_elapsed);
        printf("h min/max = %.6e / %.6e\n", global_min, global_max);
    }

    if (p.write_output)
    {
        double *h_global = NULL;
        if (rank == 0)
            h_global = (double *)malloc((size_t)NX * (size_t)NY * sizeof(double));

        MPI_Gather(&h[IDX(1, 0, NY)], mc * NY, MPI_DOUBLE,
                   h_global, mc * NY, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            /* Ghi CSV: NX dòng, mỗi dòng NY giá trị */
            FILE *fp = fopen(p.out_csv, "w");
            if (!fp)
            {
                fprintf(stderr, "Cannot open %s for writing\n", p.out_csv);
            }
            else
            {
                for (int i = 0; i < NX; i++)
                {
                    for (int j = 0; j < NY; j++)
                    {
                        fprintf(fp, "%.10g%s", h_global[IDX(i, j, NY)], (j == NY - 1) ? "" : ",");
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                printf("Wrote %s\n", p.out_csv);
            }

            free(h_global);
        }
    }
    else
    {
        if (rank == 0)
            printf("Output disabled (--no-output)\n");
    }

    /* Giải phóng và kết thúc MPI */
    free(h);
    free(hu);
    free(hv);
    free(h_new);
    free(hu_new);
    free(hv_new);

    MPI_Finalize();
    return 0;
}
