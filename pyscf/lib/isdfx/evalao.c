#include <time.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <omp.h>

static double const PRECISION = 1.e-12; // Determines number of images to sum over.

// Constants for direct expansion of solid harmonics in terms of cartesians.
// These ones are good for L=0..6
static double const sd0 = 1.7320508075688772935;  // = sqrt(3);
static double const sd1 = .5;                     // = 0.5;
static double const sd2 = .86602540378443864676;  // = (0.5)*sqrt(3);
static double const sd3 = 2.371708245126284499;   // = (0.75)*sqrt(10);
static double const sd4 = .790569415042094833;    // = (0.25)*sqrt(10);
static double const sd5 = 3.8729833462074168852;  // = sqrt(15);
static double const sd6 = .61237243569579452455;  // = (0.25)*sqrt(6);
static double const sd7 = 2.4494897427831780982;  // = sqrt(6);
static double const sd8 = 1.5;                    // = 1.5;
static double const sd9 = 1.9364916731037084426;  // = (0.5)*sqrt(15);
static double const sda = 2.9580398915498080213;  // = (0.5)*sqrt(35);
static double const sdb = 6.2749501990055666098;  // = (0.75)*sqrt(70);
static double const sdc = 2.0916500663351888699;  // = (0.25)*sqrt(70);
static double const sdd = 1.1180339887498948482;  // = (0.5)*sqrt(5);
static double const sde = 6.7082039324993690892;  // = 3*sqrt(5);
static double const sdf = 3.162277660168379332;   // = sqrt(10);
static double const sd10 = .375;                  // = 0.375;
static double const sd11 = .75;                   // = 0.75;
static double const sd12 = 3.;                    // = 3;
static double const sd13 = .5590169943749474241;  // = (0.25)*sqrt(5);
static double const sd14 = 3.3541019662496845446; // = (1.5)*sqrt(5);
static double const sd15 = .73950997288745200532; // = (0.125)*sqrt(35);
static double const sd16 = 4.4370598373247120319; // = (0.75)*sqrt(35);
static double const sd17 = 3.507803800100570049;  // = (0.9375)*sqrt(14);
static double const sd18 = 7.015607600201140098;  // = (1.875)*sqrt(14);
static double const sd19 = .7015607600201140098;  // = (0.1875)*sqrt(14);
static double const sd1a = 8.8741196746494240639; // = (1.5)*sqrt(35);
static double const sd1b = 1.5687375497513916525; // = (0.1875)*sqrt(70);
static double const sd1c = 1.045825033167594435;  // = (0.125)*sqrt(70);
static double const sd1d = 12.54990039801113322;  // = (1.5)*sqrt(70);
static double const sd1e = .52291251658379721749; // = (0.0625)*sqrt(70);
static double const sd1f = 4.1833001326703777399; // = (0.5)*sqrt(70);
static double const sd20 = 5.1234753829797991916; // = (0.5)*sqrt(105);
static double const sd21 = 10.246950765959598383; // = sqrt(105);
static double const sd22 = .48412291827592711065; // = (0.125)*sqrt(15);
static double const sd23 = .96824583655185422129; // = (0.25)*sqrt(15);
static double const sd24 = 5.8094750193111253278; // = (1.5)*sqrt(15);
static double const sd25 = 1.875;                 // = 1.875;
static double const sd26 = 3.75;                  // = 3.75;
static double const sd27 = 5.;                    // = 5;
static double const sd28 = 2.5617376914898995958; // = (0.25)*sqrt(105);
static double const sd29 = 2.218529918662356016;  // = (0.375)*sqrt(35);
static double const sd2a = 13.311179511974136096; // = (2.25)*sqrt(35);
static double const sd2b = 4.0301597362883769449; // = (0.1875)*sqrt(462);
static double const sd2c = 13.43386578762792315;  // = (0.625)*sqrt(462);
static double const sd2d = 11.634069043116428059; // = (0.9375)*sqrt(154);
static double const sd2e = 23.268138086232856118; // = (1.875)*sqrt(154);
static double const sd2f = 2.3268138086232856118; // = (0.1875)*sqrt(154);
static double const sd30 = 1.9843134832984429429; // = (0.75)*sqrt(7);
static double const sd31 = 19.843134832984429429; // = (7.5)*sqrt(7);
static double const sd32 = 8.1513994197315591977; // = (0.5625)*sqrt(210);
static double const sd33 = 5.4342662798210394651; // = (0.375)*sqrt(210);
static double const sd34 = 21.737065119284157861; // = (1.5)*sqrt(210);
static double const sd35 = 2.7171331399105197326; // = (0.1875)*sqrt(210);
static double const sd36 = 7.2456883730947192869; // = (0.5)*sqrt(210);
static double const sd37 = .90571104663683991086; // = (0.0625)*sqrt(210);
static double const sd38 = 1.8114220932736798217; // = (0.125)*sqrt(210);
static double const sd39 = 14.491376746189438574; // = sqrt(210);
static double const sd3a = 2.8641098093474000041; // = (0.625)*sqrt(21);
static double const sd3b = 5.7282196186948000082; // = (1.25)*sqrt(21);
static double const sd3c = 11.456439237389600016; // = (2.5)*sqrt(21);
static double const sd3d = 4.5825756949558400066; // = sqrt(21);
static double const sd3e = .3125;                 // = 0.3125;
static double const sd3f = .9375;                 // = 0.9375;
static double const sd40 = 5.625;                 // = 5.625;
static double const sd41 = 11.25;                 // = 11.25;
static double const sd42 = 7.5;                   // = 7.5;
static double const sd43 = .45285552331841995543; // = (0.03125)*sqrt(210);
static double const sd44 = .49607837082461073572; // = (0.1875)*sqrt(7);
static double const sd45 = 2.4803918541230536786; // = (0.9375)*sqrt(7);
static double const sd46 = 4.9607837082461073572; // = (1.875)*sqrt(7);
static double const sd47 = 29.764702249476644143; // = (11.25)*sqrt(7);
static double const sd48 = .67169328938139615748; // = (0.03125)*sqrt(462);
static double const sd49 = 10.075399340720942362; // = (0.46875)*sqrt(462);

static const int _LEN_CART[] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136};
static const int _CUM_LEN_CART[] = {
    0,
    1,
    4,
    10,
    20,
    35,
    56,
    84,
    120,
    165,
    220,
    286,
    364,
    455,
    560,
    680,
    816,
};
static int _MAX_RR_SIZE[] = {
    1,
    4,
    12,
    30,
    60,
    120,
    210,
    350,
    560,
    840,
    1260,
    1800,
    2520,
    3465,
    4620,
    6160,
    8008,
    10296,
    13104,
    16380,
    20475,
};

// Transform cartesians to Slc: map matrix [N] x [nCartY(l)] --> [N] x [(2*l+1)].
static void ShTrC6(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z4 = pIn[N * 4], z5 = pIn[N * 5], z7 = pIn[N * 7], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zc = pIn[N * 12], zd = pIn[N * 13], ze = pIn[N * 14], zf = pIn[N * 15], z10 = pIn[N * 16], z11 = pIn[N * 17], z13 = pIn[N * 19], z14 = pIn[N * 20], z15 = pIn[N * 21], z16 = pIn[N * 22], z17 = pIn[N * 23], z18 = pIn[N * 24], z19 = pIn[N * 25], z1a = pIn[N * 26], z1b = pIn[N * 27];
        pOut[N * 0] = sd2b * z3 + sd2b * z4 - sd2c * zf;
        pOut[N * 1] = sd2d * z14 - sd2e * z19 + sd2f * z7;
        pOut[N * 2] = -sd30 * z3 + sd30 * z4 + sd31 * z15 - sd31 * z16;
        pOut[N * 3] = -sd32 * z14 - sd33 * z19 + sd34 * z1a + sd35 * z7 - sd36 * z11;
        pOut[N * 4] = pIn[N * 18] * sd39 + sd37 * z3 + sd37 * z4 + sd38 * zf - sd39 * z15 - sd39 * z16;
        pOut[N * 5] = pIn[N * 8] * sd3d + sd3a * z14 + sd3a * z7 + sd3b * z19 - sd3c * z11 - sd3c * z1a;
        pOut[N * 6] = pIn[N * 2] - sd3e * z0 - sd3e * z1 - sd3f * z9 - sd3f * zb + sd40 * za + sd40 * zd + sd41 * z1b - sd42 * zc - sd42 * ze;
        pOut[N * 7] = pIn[N * 6] * sd3d + sd3a * z13 + sd3a * z5 + sd3b * z17 - sd3c * z10 - sd3c * z18;
        pOut[N * 8] = -sd36 * za + sd36 * zc + sd36 * zd - sd36 * ze + sd43 * z0 - sd43 * z1 + sd43 * z9 - sd43 * zb;
        pOut[N * 9] = sd32 * z13 + sd33 * z17 - sd34 * z18 - sd35 * z5 + sd36 * z10;
        pOut[N * 10] = -sd44 * z0 - sd44 * z1 + sd45 * z9 + sd45 * zb + sd46 * za + sd46 * zd - sd47 * z1b;
        pOut[N * 11] = sd2d * z13 - sd2e * z17 + sd2f * z5;
        pOut[N * 12] = sd48 * z0 - sd48 * z1 - sd49 * z9 + sd49 * zb;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC5(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z5 = pIn[N * 5], z7 = pIn[N * 7], z8 = pIn[N * 8], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zc = pIn[N * 12], zd = pIn[N * 13], ze = pIn[N * 14], zf = pIn[N * 15], z10 = pIn[N * 16], z12 = pIn[N * 18], z13 = pIn[N * 19], z14 = pIn[N * 20];
        pOut[N * 0] = sd17 * z5 - sd18 * zb + sd19 * z1;
        pOut[N * 1] = -sd1a * z10 + sd1a * zf;
        pOut[N * 2] = -sd1b * z5 - sd1c * zb + sd1d * z13 + sd1e * z1 - sd1f * zc;
        pOut[N * 3] = pIn[N * 17] * sd21 - sd20 * z10 - sd20 * zf;
        pOut[N * 4] = pIn[N * 6] * sd5 + sd22 * z1 + sd22 * z5 + sd23 * zb - sd24 * z13 - sd24 * zc;
        pOut[N * 5] = pIn[N * 2] + sd25 * z7 + sd25 * z8 + sd26 * z14 - sd27 * zd - sd27 * ze;
        pOut[N * 6] = pIn[N * 4] * sd5 + sd22 * z0 + sd22 * z3 + sd23 * z9 - sd24 * z12 - sd24 * za;
        pOut[N * 7] = sd20 * zd - sd20 * ze - sd28 * z7 + sd28 * z8;
        pOut[N * 8] = sd1b * z3 + sd1c * z9 - sd1d * z12 - sd1e * z0 + sd1f * za;
        pOut[N * 9] = sd29 * z7 + sd29 * z8 - sd2a * z14;
        pOut[N * 10] = sd17 * z3 - sd18 * z9 + sd19 * z0;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC4(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z4 = pIn[N * 4], z5 = pIn[N * 5], z7 = pIn[N * 7], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zd = pIn[N * 13], ze = pIn[N * 14];
        pOut[N * 0] = sda * z3 - sda * z4;
        pOut[N * 1] = sdb * ze - sdc * z7;
        pOut[N * 2] = pIn[N * 12] * sde - sdd * z3 - sdd * z4;
        pOut[N * 3] = pIn[N * 8] * sdf - sd3 * z7 - sd3 * ze;
        pOut[N * 4] = pIn[N * 2] + sd10 * z0 + sd10 * z1 + sd11 * z9 - sd12 * za - sd12 * zb;
        pOut[N * 5] = pIn[N * 6] * sdf - sd3 * z5 - sd3 * zd;
        pOut[N * 6] = -sd13 * z0 + sd13 * z1 + sd14 * za - sd14 * zb;
        pOut[N * 7] = -sdb * zd + sdc * z5;
        pOut[N * 8] = sd15 * z0 + sd15 * z1 - sd16 * z9;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC3(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z5 = pIn[N * 5], z7 = pIn[N * 7], z8 = pIn[N * 8];
        pOut[N * 0] = sd3 * z5 - sd4 * z1;
        pOut[N * 1] = pIn[N * 9] * sd5;
        pOut[N * 2] = pIn[N * 6] * sd7 - sd6 * z1 - sd6 * z5;
        pOut[N * 3] = pIn[N * 2] - sd8 * z7 - sd8 * z8;
        pOut[N * 4] = pIn[N * 4] * sd7 - sd6 * z0 - sd6 * z3;
        pOut[N * 5] = sd9 * z7 - sd9 * z8;
        pOut[N * 6] = -sd3 * z3 + sd4 * z0;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC2(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        double z0 = pIn[N * 0], z1 = pIn[N * 1];
        pOut[N * 0] = pIn[N * 3] * sd0;
        pOut[N * 1] = pIn[N * 5] * sd0;
        pOut[N * 2] = pIn[N * 2] - sd1 * z0 - sd1 * z1;
        pOut[N * 3] = pIn[N * 4] * sd0;
        pOut[N * 4] = sd2 * z0 - sd2 * z1;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC1(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        pOut[N * 0] = pIn[N * 0];
        pOut[N * 1] = pIn[N * 1];
        pOut[N * 2] = pIn[N * 2];
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void ShTrC0(double *pOut, double const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        pOut[N * 0] = pIn[N * 0];
        pOut += 1;
        pIn += 1;
    }
    return;
}

// Transform cartesians to Slc: map matrix [N] x [nCartY(l)] --> [N] x [(2*l+1)].
void ShTrN(double *pOut, double const *pIn, size_t N, unsigned l)
{
    switch (l)
    {
    case 0:
        return ShTrC0(pOut, pIn, N);
    case 1:
        return ShTrC1(pOut, pIn, N);
    case 2:
        return ShTrC2(pOut, pIn, N);
    case 3:
        return ShTrC3(pOut, pIn, N);
    case 4:
        return ShTrC4(pOut, pIn, N);
    case 5:
        return ShTrC5(pOut, pIn, N);
    case 6:
        return ShTrC6(pOut, pIn, N);
    }
    assert(0);
}

unsigned char iCartPow[3654][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {1, 0, 2}, {2, 1, 0}, {0, 1, 2}, {2, 0, 1}, {0, 2, 1}, {1, 1, 1}, {4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {1, 3, 0}, {3, 0, 1}, {1, 0, 3}, {0, 3, 1}, {0, 1, 3}, {2, 2, 0}, {2, 0, 2}, {0, 2, 2}, {1, 1, 2}, {1, 2, 1}, {2, 1, 1}, {5, 0, 0}, {0, 5, 0}, {0, 0, 5}, {1, 4, 0}, {1, 0, 4}, {4, 1, 0}, {0, 1, 4}, {4, 0, 1}, {0, 4, 1}, {3, 2, 0}, {3, 0, 2}, {2, 3, 0}, {0, 3, 2}, {2, 0, 3}, {0, 2, 3}, {3, 1, 1}, {1, 3, 1}, {1, 1, 3}, {1, 2, 2}, {2, 1, 2}, {2, 2, 1}, {6, 0, 0}, {0, 6, 0}, {0, 0, 6}, {5, 1, 0}, {1, 5, 0}, {5, 0, 1}, {1, 0, 5}, {0, 5, 1}, {0, 1, 5}, {4, 2, 0}, {4, 0, 2}, {2, 4, 0}, {2, 0, 4}, {0, 4, 2}, {0, 2, 4}, {3, 3, 0}, {3, 0, 3}, {0, 3, 3}, {1, 1, 4}, {1, 4, 1}, {4, 1, 1}, {3, 1, 2}, {1, 3, 2}, {3, 2, 1}, {1, 2, 3}, {2, 3, 1}, {2, 1, 3}, {2, 2, 2}};

int maxi(int *vals, int size)
{
    int maxVal = vals[0];
    for (int i = 0; i < size; i++)
        if (vals[i] > maxVal)
            maxVal = vals[i];
    return maxVal;
}

double max(double *vals, int size)
{
    double maxVal = vals[0];
    for (int i = 0; i < size; i++)
        if (vals[i] > maxVal)
            maxVal = vals[i];
    return maxVal;
}

double min(double *vals, int size)
{
    double maxVal = vals[0];
    for (int i = 0; i < size; i++)
        if (vals[i] < maxVal)
            maxVal = vals[i];
    return maxVal;
}

double absmax(double *vals, int size)
{
    double maxVal = fabs(vals[0]);
    for (int i = 0; i < size; i++)
        if (fabs(vals[i]) > maxVal)
            maxVal = fabs(vals[i]);
    return maxVal;
}

/*
nsub0, nsub1 -> first and the last index
alpha  -> exponent of fun
l  -> angular momentum of fun
A  -> center of function
dh -> grid spacing
vals -> values
*/
void eval_ao_1d(int N, double *R, double alpha, double A, int l, double *vals)
{
    for (int i = 0; i < N; i++)
    {
        double x_A = R[i] - A;

        vals[i * (l + 1)] = exp(-alpha * x_A * x_A);
        if (l <= 5)
        {
            if (l >= 1)
                vals[i * (l + 1) + 1] = vals[i * (l + 1) + 0] * x_A;
            if (l >= 2)
                vals[i * (l + 1) + 2] = vals[i * (l + 1) + 1] * x_A;
            if (l >= 3)
                vals[i * (l + 1) + 3] = vals[i * (l + 1) + 2] * x_A;
            if (l >= 4)
                vals[i * (l + 1) + 4] = vals[i * (l + 1) + 3] * x_A;
            if (l >= 5)
                vals[i * (l + 1) + 5] = vals[i * (l + 1) + 4] * x_A;
        }
        else
        {
            for (int mx = 1; mx < (l + 1); mx++)
            {
                vals[i * (l + 1) + mx] = vals[i * (l + 1) + mx - 1] * x_A;
            }
        }
    }
}

/*
aovals -> array (ao0:ao0+2l+1|R) that we calculate
mesh -> [n, n, n] the full mesh for the cell
nsubx -> [nx0, nx1] the function is non-zero only between these points
nsuby -> [ny0, ny1] the function is non-zero only between these points
nsubz -> [nz0, nz1] the function is non-zero only between these points
alpha  -> exponent of fun
l  -> angular momentum of fun
A  -> center of function
a  -> unit cell lattice vectors
norm -> norm of the gaussian
dh -> grid spacing
cache -> allocated memory
*/
void eval_ao_ortho(double *aoVals, double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                   double alpha, int l, double *A, double norm, double *cache)
{

    size_t N = nx * ny * nz;

    double *cachex = cache;
    double *cachey = cachex + (l + 1) * nx;
    double *cachez = cachey + (l + 1) * ny;

    eval_ao_1d(nx, Rx, alpha, A[0], l, cachex);
    eval_ao_1d(ny, Ry, alpha, A[1], l, cachey);
    eval_ao_1d(nz, Rz, alpha, A[2], l, cachez);

    int startCart = _CUM_LEN_CART[l];

    double maxProd = absmax(cachex, nx * (l + 1)) * absmax(cachey, ny * (l + 1)) * absmax(cachez, nz * (l + 1));
    if (fabs(maxProd) * norm < PRECISION)
    {
        return;
    }

    for (int mx = 0; mx < _LEN_CART[l]; mx++)
    {
        size_t index = 0;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                {
                    aoVals[mx * N + index] += norm *
                                              cachex[i * (l + 1) + iCartPow[startCart + mx][0]] *
                                              cachey[j * (l + 1) + iCartPow[startCart + mx][1]] *
                                              cachez[k * (l + 1) + iCartPow[startCart + mx][2]];

                    index++;
                }
    }
}

/*
aovals -> array (ao0:ao0+2l+1|R) that we calculate
mesh -> [n, n, n] the full mesh for the cell
nsubx -> [nx0, nx1] the function is non-zero only between these points
nsuby -> [ny0, ny1] the function is non-zero only between these points
nsubz -> [nz0, nz1] the function is non-zero only between these points
alpha  -> exponent of fun
l  -> angular momentum of fun
A  -> center of function
a  -> unit cell lattice vectors
n  -> number of exponents
norm -> norm of the gaussian
dh -> grid spacing
nimages -> number of images
images -> the index in Ls for which we have to do the summations
nLS -> total number of unit cell copies in Ls
Ls -> the array of translated unit cell copies
cache -> allocated memory
*/
void eval_ao_sumImages(double *aoVals,
                       double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                       double alpha, int l, double *A, double norm,
                       int nimages, int *images, double *Ls, double *cache)
{
    size_t N = nx * ny * nz * _LEN_CART[l];

    for (int i = 0; i < N; i++)
        aoVals[i] = 0;

    double Atrans[3];
    for (int img = 0; img < nimages; img++)
    {
        Atrans[0] = A[0] + Ls[3 * images[img] + 0];
        Atrans[1] = A[1] + Ls[3 * images[img] + 1];
        Atrans[2] = A[2] + Ls[3 * images[img] + 2];

        eval_ao_ortho(aoVals, Rx, Ry, Rz, nx, ny, nz, alpha, l, Atrans, norm, cache);
    }
}

void eval_all_aos(int nshls, double *aoValsSph,
                  double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                  double *alpha, int *l, double *A, int *atoms, double *norm,
                  int *nimages, int *images, int nLs, double *Ls)
{

    int maxL = maxi(l, nshls);
    size_t N = (size_t)nx * ny * nz;

    // Compute AO index offsets in advance
    int *ao_offsets = (int *)malloc(sizeof(int) * nshls);
    int offset = 0;
    for (int ao = 0; ao < nshls; ++ao)
    {
        ao_offsets[ao] = offset;
        offset += 2 * l[ao] + 1;
    }

#pragma omp parallel
    {
        // Thread-local memory to avoid data races
        double *aoValsCart = (double *)malloc(sizeof(double) * _LEN_CART[maxL] * N);
        double *cache = (double *)malloc(sizeof(double) * (maxL + 1) * (nx + ny + nz + 3));

#pragma omp for schedule(static)
        for (int ao = 0; ao < nshls; ao++)
        {
            eval_ao_sumImages(aoValsCart, Rx, Ry, Rz, nx, ny, nz, alpha[ao], l[ao], A + 3 * atoms[ao], norm[ao],
                              nimages[ao], images + (nLs)*ao, Ls, cache);
            ShTrN(aoValsSph + ((size_t)ao_offsets[ao]) * N, aoValsCart, N, l[ao]);
        }

        free(aoValsCart);
        free(cache);
    }

    free(ao_offsets);
}

double DoubleFactR(int l)
{
    double r = 1.;
    while (l > 1)
    {
        r = r * l;
        l = l - 2;
    }
    return fabs(r);
}

double RawGaussNorm(double fExp, int l)
{
    return pow(3.141592653589793 / (2. * fExp), 0.75) * sqrt(DoubleFactR(2 * l - 1) / (pow(4. * fExp, l)));
}

double getNimg(double A, int l, double *a, int *nimg)
{
    double minExp = A;
    double norm = RawGaussNorm(minExp, l);

    double rcut = pow(-log((PRECISION) / norm) / minExp, 0.5);

    nimg[0] = ceil(rcut / a[0]);
    nimg[1] = ceil(rcut / a[1]);
    nimg[2] = ceil(rcut / a[2]);
    return rcut;
}

void formImages(double *alpha, int *l, int n, double *a, double *Ls, int *nimg, int *nimages, int *images)
{
    int N = (2 * nimg[0] + 1) * (2 * nimg[1] + 1) * (2 * nimg[2] + 1);

    // populate Ls
    int index = 0;
    for (int nx = -nimg[0]; nx < nimg[0] + 1; nx++)
        for (int ny = -nimg[1]; ny < nimg[1] + 1; ny++)
            for (int nz = -nimg[2]; nz < nimg[2] + 1; nz++)
            {
                Ls[index * 3 + 0] = nx * a[0];
                Ls[index * 3 + 1] = ny * a[1];
                Ls[index * 3 + 2] = nz * a[2];
                index++;
            }

    int aoNimages[3];
    for (int ao = 0; ao < n; ao++)
    {
        double rcut = getNimg(alpha[ao], l[ao], a, aoNimages);

        double rcut2 = rcut * rcut;

        nimages[ao] = (2 * aoNimages[0] + 1) * (2 * aoNimages[1] + 1) * (2 * aoNimages[2] + 1);

        int index = 0, locindex = 0;
        for (int nx = -aoNimages[0]; nx < aoNimages[0] + 1; nx++)
            for (int ny = -aoNimages[1]; ny < aoNimages[1] + 1; ny++)
                for (int nz = -aoNimages[2]; nz < aoNimages[2] + 1; nz++)
                {
                    images[N * ao + locindex] = (nx + nimg[0]) * ((2 * nimg[1] + 1) * (2 * nimg[2] + 1)) + (ny + nimg[1]) * ((2 * nimg[2] + 1)) + (nz + nimg[2]);
                    locindex++;
                }
    }
}

static void imShTrC6(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        complex double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z4 = pIn[N * 4], z5 = pIn[N * 5], z7 = pIn[N * 7], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zc = pIn[N * 12], zd = pIn[N * 13], ze = pIn[N * 14], zf = pIn[N * 15], z10 = pIn[N * 16], z11 = pIn[N * 17], z13 = pIn[N * 19], z14 = pIn[N * 20], z15 = pIn[N * 21], z16 = pIn[N * 22], z17 = pIn[N * 23], z18 = pIn[N * 24], z19 = pIn[N * 25], z1a = pIn[N * 26], z1b = pIn[N * 27];
        pOut[N * 0] = sd2b * z3 + sd2b * z4 - sd2c * zf;
        pOut[N * 1] = sd2d * z14 - sd2e * z19 + sd2f * z7;
        pOut[N * 2] = -sd30 * z3 + sd30 * z4 + sd31 * z15 - sd31 * z16;
        pOut[N * 3] = -sd32 * z14 - sd33 * z19 + sd34 * z1a + sd35 * z7 - sd36 * z11;
        pOut[N * 4] = pIn[N * 18] * sd39 + sd37 * z3 + sd37 * z4 + sd38 * zf - sd39 * z15 - sd39 * z16;
        pOut[N * 5] = pIn[N * 8] * sd3d + sd3a * z14 + sd3a * z7 + sd3b * z19 - sd3c * z11 - sd3c * z1a;
        pOut[N * 6] = pIn[N * 2] - sd3e * z0 - sd3e * z1 - sd3f * z9 - sd3f * zb + sd40 * za + sd40 * zd + sd41 * z1b - sd42 * zc - sd42 * ze;
        pOut[N * 7] = pIn[N * 6] * sd3d + sd3a * z13 + sd3a * z5 + sd3b * z17 - sd3c * z10 - sd3c * z18;
        pOut[N * 8] = -sd36 * za + sd36 * zc + sd36 * zd - sd36 * ze + sd43 * z0 - sd43 * z1 + sd43 * z9 - sd43 * zb;
        pOut[N * 9] = sd32 * z13 + sd33 * z17 - sd34 * z18 - sd35 * z5 + sd36 * z10;
        pOut[N * 10] = -sd44 * z0 - sd44 * z1 + sd45 * z9 + sd45 * zb + sd46 * za + sd46 * zd - sd47 * z1b;
        pOut[N * 11] = sd2d * z13 - sd2e * z17 + sd2f * z5;
        pOut[N * 12] = sd48 * z0 - sd48 * z1 - sd49 * z9 + sd49 * zb;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC5(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        complex double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z5 = pIn[N * 5], z7 = pIn[N * 7], z8 = pIn[N * 8], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zc = pIn[N * 12], zd = pIn[N * 13], ze = pIn[N * 14], zf = pIn[N * 15], z10 = pIn[N * 16], z12 = pIn[N * 18], z13 = pIn[N * 19], z14 = pIn[N * 20];
        pOut[N * 0] = sd17 * z5 - sd18 * zb + sd19 * z1;
        pOut[N * 1] = -sd1a * z10 + sd1a * zf;
        pOut[N * 2] = -sd1b * z5 - sd1c * zb + sd1d * z13 + sd1e * z1 - sd1f * zc;
        pOut[N * 3] = pIn[N * 17] * sd21 - sd20 * z10 - sd20 * zf;
        pOut[N * 4] = pIn[N * 6] * sd5 + sd22 * z1 + sd22 * z5 + sd23 * zb - sd24 * z13 - sd24 * zc;
        pOut[N * 5] = pIn[N * 2] + sd25 * z7 + sd25 * z8 + sd26 * z14 - sd27 * zd - sd27 * ze;
        pOut[N * 6] = pIn[N * 4] * sd5 + sd22 * z0 + sd22 * z3 + sd23 * z9 - sd24 * z12 - sd24 * za;
        pOut[N * 7] = sd20 * zd - sd20 * ze - sd28 * z7 + sd28 * z8;
        pOut[N * 8] = sd1b * z3 + sd1c * z9 - sd1d * z12 - sd1e * z0 + sd1f * za;
        pOut[N * 9] = sd29 * z7 + sd29 * z8 - sd2a * z14;
        pOut[N * 10] = sd17 * z3 - sd18 * z9 + sd19 * z0;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC4(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        complex double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z4 = pIn[N * 4], z5 = pIn[N * 5], z7 = pIn[N * 7], z9 = pIn[N * 9], za = pIn[N * 10], zb = pIn[N * 11], zd = pIn[N * 13], ze = pIn[N * 14];
        pOut[N * 0] = sda * z3 - sda * z4;
        pOut[N * 1] = sdb * ze - sdc * z7;
        pOut[N * 2] = pIn[N * 12] * sde - sdd * z3 - sdd * z4;
        pOut[N * 3] = pIn[N * 8] * sdf - sd3 * z7 - sd3 * ze;
        pOut[N * 4] = pIn[N * 2] + sd10 * z0 + sd10 * z1 + sd11 * z9 - sd12 * za - sd12 * zb;
        pOut[N * 5] = pIn[N * 6] * sdf - sd3 * z5 - sd3 * zd;
        pOut[N * 6] = -sd13 * z0 + sd13 * z1 + sd14 * za - sd14 * zb;
        pOut[N * 7] = -sdb * zd + sdc * z5;
        pOut[N * 8] = sd15 * z0 + sd15 * z1 - sd16 * z9;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC3(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        complex double z0 = pIn[N * 0], z1 = pIn[N * 1], z3 = pIn[N * 3], z5 = pIn[N * 5], z7 = pIn[N * 7], z8 = pIn[N * 8];
        pOut[N * 0] = sd3 * z5 - sd4 * z1;
        pOut[N * 1] = pIn[N * 9] * sd5;
        pOut[N * 2] = pIn[N * 6] * sd7 - sd6 * z1 - sd6 * z5;
        pOut[N * 3] = pIn[N * 2] - sd8 * z7 - sd8 * z8;
        pOut[N * 4] = pIn[N * 4] * sd7 - sd6 * z0 - sd6 * z3;
        pOut[N * 5] = sd9 * z7 - sd9 * z8;
        pOut[N * 6] = -sd3 * z3 + sd4 * z0;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC2(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        complex double z0 = pIn[N * 0], z1 = pIn[N * 1];
        pOut[N * 0] = pIn[N * 3] * sd0;
        pOut[N * 1] = pIn[N * 5] * sd0;
        pOut[N * 2] = pIn[N * 2] - sd1 * z0 - sd1 * z1;
        pOut[N * 3] = pIn[N * 4] * sd0;
        pOut[N * 4] = sd2 * z0 - sd2 * z1;
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC1(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        pOut[N * 0] = pIn[N * 0];
        pOut[N * 1] = pIn[N * 1];
        pOut[N * 2] = pIn[N * 2];
        pOut += 1;
        pIn += 1;
    }
    return;
}

static void imShTrC0(double complex *pOut, double complex const *pIn, size_t N)
{
    for (size_t i = 0; i < N; ++i)
    {
        pOut[N * 0] = pIn[N * 0];
        pOut += 1;
        pIn += 1;
    }
    return;
}

void imShTrN(double complex *pOut, double complex const *pIn, size_t N, unsigned l)
{
    switch (l)
    {
    case 0:
        return imShTrC0(pOut, pIn, N);
    case 1:
        return imShTrC1(pOut, pIn, N);
    case 2:
        return imShTrC2(pOut, pIn, N);
    case 3:
        return imShTrC3(pOut, pIn, N);
    case 4:
        return imShTrC4(pOut, pIn, N);
    case 5:
        return imShTrC5(pOut, pIn, N);
    case 6:
        return imShTrC6(pOut, pIn, N);
    }
    assert(0);
}

void pbceval_ao_ortho(double complex *aoVals, double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                      double alpha, int l, double *A, double norm, double *cache,
                      int nkpt, double complex *expLk, int ao, int img)
{

    size_t N = nx * ny * nz;

    double *cachex = cache;
    double *cachey = cachex + (l + 1) * nx;
    double *cachez = cachey + (l + 1) * ny;

    eval_ao_1d(nx, Rx, alpha, A[0], l, cachex);
    eval_ao_1d(ny, Ry, alpha, A[1], l, cachey);
    eval_ao_1d(nz, Rz, alpha, A[2], l, cachez);

    int startCart = _CUM_LEN_CART[l];

    double maxProd = absmax(cachex, nx * (l + 1)) * absmax(cachey, ny * (l + 1)) * absmax(cachez, nz * (l + 1));
    if (fabs(maxProd) * norm < PRECISION)
    {
        return;
    }

    double ijk;
    for (int mx = 0; mx < _LEN_CART[l]; mx++)
    {
        size_t index = 0;
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ny; j++)
                for (int k = 0; k < nz; k++)
                {
                    ijk = norm *
                          cachex[i * (l + 1) + iCartPow[startCart + mx][0]] *
                          cachey[j * (l + 1) + iCartPow[startCart + mx][1]] *
                          cachez[k * (l + 1) + iCartPow[startCart + mx][2]];
                    for (int nk = 0; nk < nkpt; nk++)
                    {
                        aoVals[nk * _LEN_CART[l] * N + mx * N + index] += expLk[nk] * ijk;
                    }

                    index++;
                }
    }
}

void pbceval_ao_sumImages(double complex *aoVals,
                          double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                          double alpha, int l, double *A, double norm,
                          int nimages, int *images, double *Ls, double *cache,
                          int nkpt, double complex *expLk, int ao)
{
    size_t N = nx * ny * nz * _LEN_CART[l] * nkpt;

    for (int i = 0; i < N; i++)
        aoVals[i] = 0;

    double Atrans[3];
    complex double expLkn[nkpt];
    for (int img = 0; img < nimages; img++)
    {
        Atrans[0] = A[0] + Ls[3 * images[img] + 0];
        Atrans[1] = A[1] + Ls[3 * images[img] + 1];
        Atrans[2] = A[2] + Ls[3 * images[img] + 2];

        for (int nk = 0; nk < nkpt; nk++)
        {
            expLkn[nk] = expLk[nkpt * images[img] + nk]; // nimages x nkpt; copy the n-th row
        }

        pbceval_ao_ortho(aoVals, Rx, Ry, Rz, nx, ny, nz, alpha, l, Atrans, norm, cache, nkpt, expLkn, ao, images[img]);
        // each image has a phase given by expLk
        // expLk [ nLs, Nk ]
        // aoVals_k [ Nk, nao, Ng ]
    }
}

// aoValsSph is nkpt x nao x ngrid
void pbceval_all_aos(int nshls, double complex *aoValsSph,
                     double *Rx, double *Ry, double *Rz, int nx, int ny, int nz,
                     double *alpha, int *l, double *A, int *atoms, double *norm,
                     int *nimages, int *images, int nLs, double *Ls,
                     int nkpt, double complex *expLk)
{
    int maxL = maxi(l, nshls);
    size_t N = nx * ny * nz;
    int nao = 0;
    for (int ao = 0; ao < nshls; ao++)
    {
        nao += 2 * l[ao] + 1;
    }

    double complex *aoValsCart = (double complex *)malloc(sizeof(double complex) * _LEN_CART[maxL] * N * nkpt);
    double *cache = (double *)malloc(sizeof(double) * (maxL + 1) * (nx + ny + nz + 3));

    int aoIdx = 0;
    for (int ao = 0; ao < nshls; ao++)
    {
        pbceval_ao_sumImages(aoValsCart, Rx, Ry, Rz, nx, ny, nz, alpha[ao], l[ao], A + 3 * atoms[ao], norm[ao],
                             nimages[ao], images + (nLs)*ao, Ls, cache, nkpt, expLk, ao);

        for (int nk = 0; nk < nkpt; nk++)
        {
            // original: ShTrN(aoValsSph + aoIdx * N, aoValsCart, N, l[ao]);
            // aoValsSph: Nk x Nao x N
            // aoValsCart: Nk x max(num cart) x N
            imShTrN(aoValsSph + (nk * nao * N) + (aoIdx * N), aoValsCart + (nk * _LEN_CART[l[ao]] * N), N, l[ao]);
        }

        aoIdx += 2 * l[ao] + 1;
    }

    free(aoValsCart);
    free(cache);
}

// void read_numpy_array(const char *filename, void **data, int *rows, int *cols, size_t element_size) {
//     FILE *file = fopen(filename, "rb");
//     if (!file) {
//         perror("Error opening file");
//         exit(1);
//     }

//     // Read the shape of the array
//     int shape[2];
//     fread(shape, sizeof(int), 2, file);
//     *rows = shape[0];
//     *cols = shape[1];

//     printf("Read shape: rows=%d, cols=%d, element_size=%zu\n", *rows, *cols, element_size);

//     // Check for invalid sizes
//     if (*rows <= 0 || *cols <= 0) {
//         fprintf(stderr, "Invalid array shape: %d x %d\n", *rows, *cols);
//         exit(1);
//     }

//     // Compute memory size safely
//     size_t total_size;
//     if (__builtin_mul_overflow((size_t)(*rows), (size_t)(*cols), &total_size) ||
//         __builtin_mul_overflow(total_size, element_size, &total_size)) {
//         fprintf(stderr, "Memory allocation size overflow detected!\n");
//         exit(1);
//     }

//     printf("Allocating memory: %zu bytes\n", total_size);

//     // Allocate memory
//     *data = malloc(total_size);
//     if (!(*data)) {
//         perror("Memory allocation failed");
//         exit(1);
//     }

//     // Read the array data
//     fread(*data, element_size, (*rows) * (*cols), file);

//     fclose(file);
// }

// int main() {
//     double *aoValsSph, *Rx, *Ry, *Rz, *alpha, *A, *norm, *Ls;
//     int *l,  *atoms, *nimages, *images;
//     int rows, cols, nLs, nshls, nx, ny, nz;
//     int dsize = sizeof(double);
//     int isize = sizeof(int);

//     read_numpy_array("aoValsSph.bin", (void**)&aoValsSph, &rows, &cols, dsize);
//     printf("Loaded aoValsSph of size %dx%d\n", rows, cols);

//     read_numpy_array("Rx.bin", (void**)&Rx, &rows, &nx, dsize);
//     printf("Loaded Rx of size %dx%d\n", rows, nx);

//     read_numpy_array("Ry.bin",(void**) &Ry, &rows, &ny, dsize);
//     printf("Loaded Ry of size %dx%d\n", rows, ny);

//     read_numpy_array("Rz.bin", (void**)&Rz, &rows, &nz, dsize);
//     printf("Loaded Rz of size %dx%d\n", rows, nz);

//     read_numpy_array("alpha.bin", (void**)&alpha, &rows, &nshls, dsize);
//     printf("Loaded alpha of size %dx%d\n", rows, nshls);

//     read_numpy_array("A.bin", (void**)&A, &rows, &cols, dsize);
//     printf("Loaded A of size %dx%d\n", rows, cols);

//     read_numpy_array("norm.bin", (void**)&norm, &rows, &cols, dsize);
//     printf("Loaded norm of size %dx%d\n", rows, cols);

//     read_numpy_array("Ls.bin",(void**) &Ls, &rows, &nLs, dsize);
//     printf("Loaded Ls of size %dx%d\n", rows, nLs);

//     read_numpy_array("l.bin", (void**)&l, &rows, &cols, isize);
//     printf("Loaded l of size %dx%d\n", rows, cols);

//     read_numpy_array("atoms.bin", (void**)&atoms, &rows, &cols, isize);
//     printf("Loaded atoms of size %dx%d\n", rows, cols);

//     read_numpy_array("nimages.bin", (void**)&nimages, &rows, &cols, isize);
//     printf("Loaded nimages of size %dx%d\n", rows, cols);

//     read_numpy_array("images.bin", (void**)&images, &rows, &cols, isize);
//     printf("Loaded images of size %dx%d\n", rows, cols);

//     double precision = 1.e-12;

//     // Call eval_all_aos (Replace with actual function call)
//     eval_all_aos(nshls, aoValsSph,
//         Rx, Ry, Rz, nx, ny, nz,
//         alpha, l, A, atoms, norm,
//         nimages, images, nLs, Ls, precision);

//     // Free memory
//     free(aoValsSph);
//     free(Rx);
//     free(Ry);
//     free(Rz);
//     free(alpha);
//     free(A);
//     free(norm);
//     free(Ls);
//     free(l);
//     free(atoms);
//     free(nimages);
//     free(images);

//     return 0;
// }

// Compile library:
//  gcc -O3 -fPIC -shared evalao.c -o libevalao.so
// Compile program:
//  gcc -O3 -g -lm evalao.c -o test_evalao
// Run program:
//  valgrind ./test_evalao