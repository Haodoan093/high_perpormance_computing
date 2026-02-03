#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
  ============================================================
  2D SHALLOW WATER EQUATIONS (SWE) - SERIAL, CONSERVATIVE FORM
  LAX–WENDROFF FINITE-DIFFERENCE SCHEME (2-step), paper Eq. (3.6)
  ============================================================

  Conservative variables: U = [h, hu, hv]^T

  Grid:
    - Physical domain: NX x NY
    - Ghost columns in x: i = 0 and i = NX+1 (array size (NX+2)*NY)
    - In y: no extra ghost; use j=0 and j=NY-1 as reflective walls.

  Scheme (finite difference LW, 2-stage):
    Half step:
      U^{n+1/2}_{i,j} = 1/4(neighbors)
                        - dt/(4dx)(F(U^n_{i+1,j})-F(U^n_{i-1,j}))
                        - dt/(4dy)(G(U^n_{i,j+1})-G(U^n_{i,j-1}))

    Full step:
      U^{n+1}_{i,j} = U^n_{i,j}
                      - dt/(2dx)(F(U^{n+1/2}_{i+1,j})-F(U^{n+1/2}_{i-1,j}))
                      - dt/(2dy)(G(U^{n+1/2}_{i,j+1})-G(U^{n+1/2}_{i,j-1}))
*/

#define DEFAULT_NX 200
#define DEFAULT_NY 200
#define DEFAULT_NSTEPS 400
#define DEFAULT_DX 1.0
#define DEFAULT_DY 1.0
#define DEFAULT_DT 0.01
#define DEFAULT_G 9.81

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
    snprintf(p->out_csv, sizeof(p->out_csv), "h_final_serial_lw.csv");
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--nx N] [--ny N] [--steps N] [--dt DT] [--dx DX] [--dy DY] [--g G] [--out FILE]\n",
            prog);
}

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

    if (p->nx <= 2 || p->ny <= 2 || p->nsteps < 0)
        return -1;
    if (p->dx <= 0.0 || p->dy <= 0.0 || p->dt <= 0.0 || p->g <= 0.0)
        return -1;
    return 1;
}

static inline double safe_h(double h)
{
    const double eps = 1e-12;
    return (h > eps) ? h : eps;
}

static void apply_bc_serial(int nx, int ny, double *h, double *hu, double *hv)
{
    for (int i = 0; i < nx + 2; i++)
    {
        h[IDX(i, 0, ny)] = h[IDX(i, 1, ny)];
        hu[IDX(i, 0, ny)] = hu[IDX(i, 1, ny)];
        hv[IDX(i, 0, ny)] = -hv[IDX(i, 1, ny)];

        h[IDX(i, ny - 1, ny)] = h[IDX(i, ny - 2, ny)];
        hu[IDX(i, ny - 1, ny)] = hu[IDX(i, ny - 2, ny)];
        hv[IDX(i, ny - 1, ny)] = -hv[IDX(i, ny - 2, ny)];
    }

    for (int j = 0; j < ny; j++)
    {
        h[IDX(0, j, ny)] = h[IDX(1, j, ny)];
        hu[IDX(0, j, ny)] = -hu[IDX(1, j, ny)];
        hv[IDX(0, j, ny)] = hv[IDX(1, j, ny)];

        h[IDX(nx + 1, j, ny)] = h[IDX(nx, j, ny)];
        hu[IDX(nx + 1, j, ny)] = -hu[IDX(nx, j, ny)];
        hv[IDX(nx + 1, j, ny)] = hv[IDX(nx, j, ny)];
    }
}

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

static void update_lw_half_step(int nx, int ny,
                                double dx, double dy, double dt, double g,
                                const double *h, const double *hu, const double *hv,
                                double *h_half, double *hu_half, double *hv_half)
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

            h_half[c] =
                0.25 * (h[ip] + h[im] + h[jp] + h[jm]) - (dt / (4.0 * dx)) * (F1p - F1m) - (dt / (4.0 * dy)) * (G1p - G1m);
            hu_half[c] =
                0.25 * (hu[ip] + hu[im] + hu[jp] + hu[jm]) - (dt / (4.0 * dx)) * (F2p - F2m) - (dt / (4.0 * dy)) * (G2p - G2m);
            hv_half[c] =
                0.25 * (hv[ip] + hv[im] + hv[jp] + hv[jm]) - (dt / (4.0 * dx)) * (F3p - F3m) - (dt / (4.0 * dy)) * (G3p - G3m);

            if (h_half[c] < 1e-10)
            {
                h_half[c] = 1e-10;
                hu_half[c] = 0.0;
                hv_half[c] = 0.0;
            }
        }
    }
}

static void update_lw_full_step(int nx, int ny,
                                double dx, double dy, double dt, double g,
                                const double *h, const double *hu, const double *hv,
                                const double *h_half, const double *hu_half, const double *hv_half,
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

            flux_F(h_half[ip], hu_half[ip], hv_half[ip], g, &F1p, &F2p, &F3p);
            flux_F(h_half[im], hu_half[im], hv_half[im], g, &F1m, &F2m, &F3m);
            flux_G(h_half[jp], hu_half[jp], hv_half[jp], g, &G1p, &G2p, &G3p);
            flux_G(h_half[jm], hu_half[jm], hv_half[jm], g, &G1m, &G2m, &G3m);

            h_new[c] = h[c] - (dt / (2.0 * dx)) * (F1p - F1m) - (dt / (2.0 * dy)) * (G1p - G1m);
            hu_new[c] = hu[c] - (dt / (2.0 * dx)) * (F2p - F2m) - (dt / (2.0 * dy)) * (G2p - G2m);
            hv_new[c] = hv[c] - (dt / (2.0 * dx)) * (F3p - F3m) - (dt / (2.0 * dy)) * (G3p - G3m);

            if (h_new[c] < 1e-10)
            {
                h_new[c] = 1e-10;
                hu_new[c] = 0.0;
                hv_new[c] = 0.0;
            }
        }
    }
}

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

    const size_t n = (size_t)(NX + 2) * (size_t)NY;

    double *h = (double *)malloc(n * sizeof(double));
    double *hu = (double *)malloc(n * sizeof(double));
    double *hv = (double *)malloc(n * sizeof(double));

    double *h_half = (double *)malloc(n * sizeof(double));
    double *hu_half = (double *)malloc(n * sizeof(double));
    double *hv_half = (double *)malloc(n * sizeof(double));

    double *h_new = (double *)malloc(n * sizeof(double));
    double *hu_new = (double *)malloc(n * sizeof(double));
    double *hv_new = (double *)malloc(n * sizeof(double));

    if (!h || !hu || !hv || !h_half || !hu_half || !hv_half || !h_new || !hu_new || !hv_new)
    {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (int i = 0; i < NX + 2; i++)
        for (int j = 0; j < NY; j++)
        {
            h[IDX(i, j, NY)] = 1.0;
            hu[IDX(i, j, NY)] = 0.0;
            hv[IDX(i, j, NY)] = 0.0;
        }

    h[IDX(NX / 2 + 1, NY / 2, NY)] = 2.0;

    apply_bc_serial(NX, NY, h, hu, hv);

    double t0 = now_seconds();

    for (int step = 0; step < NSTEPS; step++)
    {
        apply_bc_serial(NX, NY, h, hu, hv);

        update_lw_half_step(NX, NY, DX, DY, DT, G, h, hu, hv, h_half, hu_half, hv_half);
        apply_bc_serial(NX, NY, h_half, hu_half, hv_half);

        update_lw_full_step(NX, NY, DX, DY, DT, G, h, hu, hv, h_half, hu_half, hv_half, h_new, hu_new, hv_new);
        apply_bc_serial(NX, NY, h_new, hu_new, hv_new);

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

    printf("Serial LW Shallow Water finished in %f s\n", t1 - t0);
    printf("h min/max = %.6e / %.6e\n", hmin, hmax);

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
    free(h_half);
    free(hu_half);
    free(hv_half);
    free(h_new);
    free(hu_new);
    free(hv_new);

    return 0;
}
