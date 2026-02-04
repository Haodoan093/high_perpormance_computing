#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
  ============================================================
  2D SHALLOW WATER EQUATIONS (SWE) - SERIAL, CONSERVATIVE FORM
  ============================================================

  Mục tiêu:
  - Mô phỏng hệ phương trình nước nông 2D dạng bảo toàn:
      U = [ h,  hu,  hv ]^T
    trong đó:
      h  : chiều cao mực nước
      hu : động lượng theo x (h*u)
      hv : động lượng theo y (h*v)

  Lưới:
  - Kích thước vật lý: NX x NY ô (cell-centered)
  - Có "ghost cell" theo phương x: i = 0 và i = NX+1 (tổng NX+2 cột)
  - Theo phương y: không cấp ghost riêng; dùng chính hàng biên j=0 và j=NY-1 làm biên phản xạ.

  Thuật toán:
  - Lax–Friedrichs 2D (dạng trung bình 4 láng giềng + trừ chênh lệch thông lượng trung tâm):
      U^{n+1}_{i,j} =
          1/4 (U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1})
        - dt/(2dx) (F(U_{i+1,j}) - F(U_{i-1,j}))
        - dt/(2dy) (G(U_{i,j+1}) - G(U_{i,j-1}))
  - Mỗi bước thời gian:
      1) áp điều kiện biên (reflective BC)
      2) cập nhật U_new cho vùng nội (interior)
      3) áp điều kiện biên cho U_new
      4) swap con trỏ U <-> U_new

  Ghi chú:
  - Đây là bản SERIAL để đối chiếu với bản MPI (cùng công thức, khác cách chia miền/halo).
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
  - i chạy từ 0..NX+1 (có ghost), j chạy 0..NY-1
*/
static inline size_t IDX(int i, int j, int ny)
{
    return (size_t)i * (size_t)ny + (size_t)j;
}

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
    snprintf(p->out_csv, sizeof(p->out_csv), "h_final_serial.csv");
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--nx N] [--ny N] [--steps N] [--dt DT] [--dx DX] [--dy DY] [--g G] [--out FILE]\n",
            prog);
}

/* Parse tham số dòng lệnh (để tiện benchmark/verify) */
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
            snprintf(p->out_csv, sizeof(p->out_csv), "%s", argv[++i]);
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

/* Tránh chia cho 0 khi h rất nhỏ (khô nước) */
static inline double safe_h(double h)
{
    const double eps = 1e-12;
    return (h > eps) ? h : eps;
}

/*
  Reflective boundary conditions (BC):
  - Ở biên "tường": vận tốc pháp tuyến đổi dấu, tiếp tuyến giữ nguyên.
  - Do lưu hu/hv nên:
      + Biên theo y (trên/dưới): đảo dấu hv
      + Biên theo x (trái/phải): đảo dấu hu
  - Đồng thời copy h từ cell bên trong ra cell biên/ghost.
*/
static void apply_bc_serial(int nx, int ny, double *h, double *hu, double *hv)
{
    // y boundaries: j=0 và j=ny-1
    for (int i = 0; i < nx + 2; i++)
    {
        // bottom (j=0)
        h[IDX(i, 0, ny)] = h[IDX(i, 1, ny)];
        hu[IDX(i, 0, ny)] = hu[IDX(i, 1, ny)];
        hv[IDX(i, 0, ny)] = -hv[IDX(i, 1, ny)];

        // top (j=ny-1)
        h[IDX(i, ny - 1, ny)] = h[IDX(i, ny - 2, ny)];
        hu[IDX(i, ny - 1, ny)] = hu[IDX(i, ny - 2, ny)];
        hv[IDX(i, ny - 1, ny)] = -hv[IDX(i, ny - 2, ny)];
    }

    // x boundaries (ghost columns): i=0 và i=nx+1
    for (int j = 0; j < ny; j++)
    {
        // left ghost i=0
        h[IDX(0, j, ny)] = h[IDX(1, j, ny)];
        hu[IDX(0, j, ny)] = -hu[IDX(1, j, ny)];
        hv[IDX(0, j, ny)] = hv[IDX(1, j, ny)];

        // right ghost i=nx+1
        h[IDX(nx + 1, j, ny)] = h[IDX(nx, j, ny)];
        hu[IDX(nx + 1, j, ny)] = -hu[IDX(nx, j, ny)];
        hv[IDX(nx + 1, j, ny)] = hv[IDX(nx, j, ny)];
    }
}

/*
  Flux theo phương x:
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
  Flux theo phương y:
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

/*
  One explicit LF finite-difference step (paper Eq. 3.3) on interior cells.
  - Updates i = 1..nx, j = 1..ny-2 (avoid physical y-walls).
  - Assumes boundary/ghost values in (h,hu,hv) are already valid.
*/
static void update_lf_step_serial(int nx, int ny,
                                  double dx, double dy, double dt, double g,
                                  const double *h, const double *hu, const double *hv,
                                  double *h_new, double *hu_new, double *hv_new)
{
    for (int i = 1; i <= nx; i++)
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

/* Timer monotonic để đo runtime */
static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    Params p;
    params_init_default(&p);

    int parse_rc = parse_args(argc, argv, &p);
    if (parse_rc <= 0)
        return (parse_rc == 0) ? 0 : 1;

    const int NX = p.nx;
    const int NY = p.ny;
    const int NSTEPS = p.nsteps;
    const double DX = p.dx;
    const double DY = p.dy;
    const double DT = p.dt;
    const double G = p.g;

    /*
      Cấp phát:
      - Mỗi trường có size (NX+2)*NY do có 2 cột ghost theo x.
      - h_new/hu_new/hv_new là buffer cho bước thời gian tiếp theo.
    */
    double *h = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));
    double *hu = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));
    double *hv = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));

    double *h_new = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));
    double *hu_new = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));
    double *hv_new = (double *)malloc((size_t)(NX + 2) * (size_t)NY * sizeof(double));

    if (!h || !hu || !hv || !h_new || !hu_new || !hv_new)
    {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Khởi tạo trạng thái ban đầu: nước tĩnh độ cao 1.0 */
    for (int i = 0; i < NX + 2; i++)
        for (int j = 0; j < NY; j++)
        {
            h[IDX(i, j, NY)] = 1.0;
            hu[IDX(i, j, NY)] = 0.0;
            hv[IDX(i, j, NY)] = 0.0;
        }

    /* Tạo perturbation: "uprising/hump" ở gần trung tâm */
    h[IDX(NX / 2 + 1, NY / 2, NY)] = 2.0;

    /* Áp BC trước khi vào vòng lặp */
    apply_bc_serial(NX, NY, h, hu, hv);

    double t0 = now_seconds();

    for (int step = 0; step < NSTEPS; step++)
    {
        /* Luôn giữ biên/ghost hợp lệ trước khi tính nội bộ */
        apply_bc_serial(NX, NY, h, hu, hv);

        /* LF finite-difference (paper Eq. 3.3) on interior cells */
        update_lf_step_serial(NX, NY, DX, DY, DT, G, h, hu, hv, h_new, hu_new, hv_new);

        /* Áp BC lên nghiệm mới trước khi swap */
        apply_bc_serial(NX, NY, h_new, hu_new, hv_new);

        /* Swap con trỏ (không copy mảng) */
        double *tmp = h;
        h = h_new;
        h_new = tmp;
        tmp = hu;
        hu = hu_new;
        hu_new = tmp;
        tmp = hv;
        hv = hv_new;
        hv_new = tmp;
    }

    double t1 = now_seconds();

    /* Thống kê min/max để sanity-check */
    double hmin = 1e300, hmax = -1e300;
    for (int i = 1; i <= NX; i++)
        for (int j = 0; j < NY; j++)
        {
            double val = h[IDX(i, j, NY)];
            if (val < hmin)
                hmin = val;
            if (val > hmax)
                hmax = val;
        }

    printf("Serial Conservative Shallow Water finished in %f s\n", t1 - t0);
    printf("h min/max = %.6e / %.6e\n", hmin, hmax);

    /*
      Xuất CSV:
      - Chỉ xuất miền vật lý (bỏ ghost i=0 và i=NX+1)
      - Kích thước file: NX dòng, mỗi dòng NY số, phân tách bằng dấu phẩy.
    */
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
                fprintf(fp, "%.10g%s", h[IDX(i + 1, j, NY)], (j == NY - 1) ? "" : ",");
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("Wrote %s\n", p.out_csv);
    }

    free(h);
    free(hu);
    free(hv);
    free(h_new);
    free(hu_new);
    free(hv_new);

    return 0;
}
