#define _POSIX_C_SOURCE 199309L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
  ============================================================
  2D SHALLOW WATER EQUATIONS (SWE) - SERIAL, CONSERVATIVE FORM
  MACCORMACK FINITE-DIFFERENCE SCHEME (predictor-corrector)
  ============================================================

  Conservative variables: U = [h, hu, hv]^T

  Predictor (forward differences):
    U*_{i,j} = U^n_{i,j}
               - (dt/dx)(F(U^n_{i+1,j}) - F(U^n_{i,j}))
               - (dt/dy)(G(U^n_{i,j+1}) - G(U^n_{i,j}))

  Corrector (backward differences on predicted):
    U^{n+1}_{i,j} = 0.5 * [ U^n_{i,j} + U*_{i,j}
                            - (dt/dx)(F(U*_{i,j}) - F(U*_{i-1,j}))
                            - (dt/dy)(G(U*_{i,j}) - G(U*_{i,j-1})) ]

  BC: reflective walls (same as LF serial).
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
    snprintf(p->out_csv, sizeof(p->out_csv), "h_final_serial_mc.csv");
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

static void update_mc_predictor(int nx, int ny,
                                double dx, double dy, double dt, double g,
                                const double *h, const double *hu, const double *hv,
                                double *h_pred, double *hu_pred, double *hv_pred)
{
    for (int i = 1; i <= nx; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            const size_t c = IDX(i, j, ny);
            const size_t ip = IDX(i + 1, j, ny);
            const size_t jp = IDX(i, j + 1, ny);

            double F1_ip, F2_ip, F3_ip, F1_c, F2_c, F3_c;
            double G1_jp, G2_jp, G3_jp, G1_c, G2_c, G3_c;

            flux_F(h[ip], hu[ip], hv[ip], g, &F1_ip, &F2_ip, &F3_ip);
            flux_F(h[c], hu[c], hv[c], g, &F1_c, &F2_c, &F3_c);

            flux_G(h[jp], hu[jp], hv[jp], g, &G1_jp, &G2_jp, &G3_jp);
            flux_G(h[c], hu[c], hv[c], g, &G1_c, &G2_c, &G3_c);

            h_pred[c] = h[c] - (dt / dx) * (F1_ip - F1_c) - (dt / dy) * (G1_jp - G1_c);
            hu_pred[c] = hu[c] - (dt / dx) * (F2_ip - F2_c) - (dt / dy) * (G2_jp - G2_c);
            hv_pred[c] = hv[c] - (dt / dx) * (F3_ip - F3_c) - (dt / dy) * (G3_jp - G3_c);

            if (h_pred[c] < 1e-10)
            {
                h_pred[c] = 1e-10;
                hu_pred[c] = 0.0;
                hv_pred[c] = 0.0;
            }
        }
    }
}

static void update_mc_corrector(int nx, int ny,
                                double dx, double dy, double dt, double g,
                                const double *h, const double *hu, const double *hv,
                                const double *h_pred, const double *hu_pred, const double *hv_pred,
                                double *h_new, double *hu_new, double *hv_new)
{
    for (int i = 1; i <= nx; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            const size_t c = IDX(i, j, ny);
            const size_t im = IDX(i - 1, j, ny);
            const size_t jm = IDX(i, j - 1, ny);

            double F1_c, F2_c, F3_c, F1_im, F2_im, F3_im;
            double G1_c, G2_c, G3_c, G1_jm, G2_jm, G3_jm;

            flux_F(h_pred[c], hu_pred[c], hv_pred[c], g, &F1_c, &F2_c, &F3_c);
            flux_F(h_pred[im], hu_pred[im], hv_pred[im], g, &F1_im, &F2_im, &F3_im);

            flux_G(h_pred[c], hu_pred[c], hv_pred[c], g, &G1_c, &G2_c, &G3_c);
            flux_G(h_pred[jm], hu_pred[jm], hv_pred[jm], g, &G1_jm, &G2_jm, &G3_jm);

            h_new[c] = 0.5 * (h[c] + h_pred[c] - (dt / dx) * (F1_c - F1_im) - (dt / dy) * (G1_c - G1_jm));
            hu_new[c] = 0.5 * (hu[c] + hu_pred[c] - (dt / dx) * (F2_c - F2_im) - (dt / dy) * (G2_c - G2_jm));
            hv_new[c] = 0.5 * (hv[c] + hv_pred[c] - (dt / dx) * (F3_c - F3_im) - (dt / dy) * (G3_c - G3_jm));

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

    double *h_pred = (double *)malloc(n * sizeof(double));
    double *hu_pred = (double *)malloc(n * sizeof(double));
    double *hv_pred = (double *)malloc(n * sizeof(double));

    double *h_new = (double *)malloc(n * sizeof(double));
    double *hu_new = (double *)malloc(n * sizeof(double));
    double *hv_new = (double *)malloc(n * sizeof(double));

    if (!h || !hu || !hv || !h_pred || !hu_pred || !hv_pred || !h_new || !hu_new || !hv_new)
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

        update_mc_predictor(NX, NY, DX, DY, DT, G, h, hu, hv, h_pred, hu_pred, hv_pred);
        apply_bc_serial(NX, NY, h_pred, hu_pred, hv_pred);

        update_mc_corrector(NX, NY, DX, DY, DT, G, h, hu, hv, h_pred, hu_pred, hv_pred, h_new, hu_new, hv_new);
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

    printf("Serial MacCormack Shallow Water finished in %f s\n", t1 - t0);
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
    free(h_pred);
    free(hu_pred);
    free(hv_pred);
    free(h_new);
    free(hu_new);
    free(hv_new);

    return 0;
}
