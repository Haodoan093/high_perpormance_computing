#define _POSIX_C_SOURCE 199309L

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
  ============================================================
  2D SHALLOW WATER EQUATIONS (SWE) - MPI, CONSERVATIVE FORM
  MACCORMACK FINITE-DIFFERENCE SCHEME (predictor-corrector)
  ============================================================

  Conservative variables:
    U = [ h, hu, hv ]^T

  MacCormack scheme for U_t + F(U)_x + G(U)_y = 0:

    Predictor (forward differences):
      U*_{i,j} = U^n_{i,j}
                 - (dt/dx) (F(U^n_{i+1,j}) - F(U^n_{i,j}))
                 - (dt/dy) (G(U^n_{i,j+1}) - G(U^n_{i,j}))

    Corrector (backward differences on predicted state):
      U^{n+1}_{i,j} = 0.5 * [ U^n_{i,j} + U*_{i,j}
                              - (dt/dx) (F(U*_{i,j}) - F(U*_{i-1,j}))
                              - (dt/dy) (G(U*_{i,j}) - G(U*_{i,j-1})) ]

  Parallelization:
    - 1D domain decomposition in x into slabs.
    - Two halo exchanges per timestep: one for U^n, one for U*.

  Boundary conditions: reflective walls (same as LF/LW versions).
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
    snprintf(p->out_csv, sizeof(p->out_csv), "h_final_mc.csv");
    p->write_output = 1;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [--nx N] [--ny N] [--steps N] [--dt DT] [--dx DX] [--dy DY] [--g G] [--out FILE] [--no-output]\n",
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
        else if (strcmp(argv[i], "--no-output") == 0)
            p->write_output = 0;
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

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline double safe_h(double h)
{
    const double eps = 1e-12;
    return (h > eps) ? h : eps;
}

static void apply_bc(int rank, int size, int mc, int ny, double *h, double *hu, double *hv)
{
    for (int i = 0; i < mc + 2; i++)
    {
        h[IDX(i, 0, ny)] = h[IDX(i, 1, ny)];
        hu[IDX(i, 0, ny)] = hu[IDX(i, 1, ny)];
        hv[IDX(i, 0, ny)] = -hv[IDX(i, 1, ny)];

        h[IDX(i, ny - 1, ny)] = h[IDX(i, ny - 2, ny)];
        hu[IDX(i, ny - 1, ny)] = hu[IDX(i, ny - 2, ny)];
        hv[IDX(i, ny - 1, ny)] = -hv[IDX(i, ny - 2, ny)];
    }

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

static void halo_exchange_x_field(int mc, int ny, double *field,
                                  int left, int right, int tag_base)
{
    MPI_Status st;

    MPI_Sendrecv(&field[IDX(1, 0, ny)], ny, MPI_DOUBLE, left, tag_base,
                 &field[IDX(mc + 1, 0, ny)], ny, MPI_DOUBLE, right, tag_base,
                 MPI_COMM_WORLD, &st);

    MPI_Sendrecv(&field[IDX(mc, 0, ny)], ny, MPI_DOUBLE, right, tag_base + 1,
                 &field[IDX(0, 0, ny)], ny, MPI_DOUBLE, left, tag_base + 1,
                 MPI_COMM_WORLD, &st);
}

static void halo_exchange_x(int mc, int ny, double *h, double *hu, double *hv,
                            int left, int right)
{
    halo_exchange_x_field(mc, ny, h, left, right, 10);
    halo_exchange_x_field(mc, ny, hu, left, right, 20);
    halo_exchange_x_field(mc, ny, hv, left, right, 30);
}

static void update_mc_predictor_local(int mc, int ny,
                                      double dx, double dy, double dt, double g,
                                      const double *h, const double *hu, const double *hv,
                                      double *h_pred, double *hu_pred, double *hv_pred)
{
    for (int i = 1; i <= mc; i++)
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

static void update_mc_corrector_local(int mc, int ny,
                                      double dx, double dy, double dt, double g,
                                      const double *h, const double *hu, const double *hv,
                                      const double *h_pred, const double *hu_pred, const double *hv_pred,
                                      double *h_new, double *hu_new, double *hv_new)
{
    for (int i = 1; i <= mc; i++)
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

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (NX % size != 0)
    {
        if (rank == 0)
            printf("NX must be divisible by number of processes\n");
        MPI_Finalize();
        return 0;
    }

    const int mc = NX / size;
    const size_t n = (size_t)(mc + 2) * (size_t)NY;

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
        fprintf(stderr, "Rank %d: malloc failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < mc + 2; i++)
        for (int j = 0; j < NY; j++)
        {
            h[IDX(i, j, NY)] = 1.0;
            hu[IDX(i, j, NY)] = 0.0;
            hv[IDX(i, j, NY)] = 0.0;
        }

    int global_center = NX / 2;
    if (rank == global_center / mc)
    {
        int local_i = global_center % mc + 1;
        h[IDX(local_i, NY / 2, NY)] = 2.0;
    }

    apply_bc(rank, size, mc, NY, h, hu, hv);

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = now_seconds();

    const int left = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    const int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    for (int step = 0; step < NSTEPS; step++)
    {
        /* Ensure U^n ghosts and boundaries are valid */
        halo_exchange_x(mc, NY, h, hu, hv, left, right);
        apply_bc(rank, size, mc, NY, h, hu, hv);

        /* Predictor */
        update_mc_predictor_local(mc, NY, DX, DY, DT, G, h, hu, hv, h_pred, hu_pred, hv_pred);

        /* Prepare predicted state boundaries for corrector */
        halo_exchange_x(mc, NY, h_pred, hu_pred, hv_pred, left, right);
        apply_bc(rank, size, mc, NY, h_pred, hu_pred, hv_pred);

        /* Corrector */
        update_mc_corrector_local(mc, NY, DX, DY, DT, G, h, hu, hv, h_pred, hu_pred, hv_pred,
                                  h_new, hu_new, hv_new);

        apply_bc(rank, size, mc, NY, h_new, hu_new, hv_new);

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

    const double t1 = now_seconds();
    const double local_elapsed = t1 - t0;

    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

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

    if (rank == 0)
    {
        printf("MPI MacCormack Shallow Water finished in %f s\n", max_elapsed);
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
                        fprintf(fp, "%.10g%s", h_global[IDX(i, j, NY)], (j == NY - 1) ? "" : ",");
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

    free(h);
    free(hu);
    free(hv);
    free(h_pred);
    free(hu_pred);
    free(hv_pred);
    free(h_new);
    free(hu_new);
    free(hv_new);

    MPI_Finalize();
    return 0;
}
