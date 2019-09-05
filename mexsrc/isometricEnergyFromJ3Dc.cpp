#include <mex.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

template<class real>
double sqr(real x) { return x*x; };
template<class real>
double pow3(real x) { return x*x*x; };
template<class real>
double pow4(real x) { return sqr( sqr(x) ); };

//#define PRINT_DEBUGGING_OUTPUT
#define USE_SCALAR_IMPLEMENTATION
// #define USE_SSE_IMPLEMENTATION
// #define USE_AVX_IMPLEMENTATION

#define COMPUTE_V_AS_MATRIX
//#define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
//#define COMPUTE_U_AS_QUATERNION

#include "Singular_Value_Decomposition_Preamble.hpp"

double isometricEnergyFromJ3D(double* Eg, double* Eh, const double* Js, const double* X2J, const double* w, int energyType, int nt)
{
    double en = 0;
    
    Map<const Matrix<double, Dynamic, 4> > X2Js(X2J, 3*nt, 4);
#pragma omp parallel for reduction(+:en)
    for (int ii = 0; ii < nt; ii++) {
        const double *curJ = Js + 9 * ii;
        float A11 = (float)curJ[0];
        float A12 = (float)curJ[1];
        float A13 = (float)curJ[2];
        float A21 = (float)curJ[3];
        float A22 = (float)curJ[4];
        float A23 = (float)curJ[5];
        float A31 = (float)curJ[6];
        float A32 = (float)curJ[7];
        float A33 = (float)curJ[8];
        
#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
        ENABLE_SCALAR_IMPLEMENTATION(Sa11.f = A11;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa21.f = A21;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa31.f = A31;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa12.f = A12;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa22.f = A22;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa32.f = A32;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa13.f = A13;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa23.f = A23;)
        ENABLE_SCALAR_IMPLEMENTATION(Sa33.f = A33;)
#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"
        
        float sigma[3] = { Sa11.f, Sa22.f, Sa33.f };
        //Vector3f sig = { Sa11.f, Sa22.f, Sa33.f };
        double e;
        switch(energyType){
            case 1:    //Eiso
                e = sqr(sigma[0]) + 1 / sqr(sigma[0]) + sqr(sigma[1]) + 1 / sqr(sigma[1]) + sqr(sigma[2]) + 1 / sqr(sigma[2]) - 6;
                //e = sig.squaredNorm() + sig.cwiseInverse().squaredNorm();
                break;
            case 5: 
                e = sqr(sigma[0] - 1) + sqr(sigma[1] - 1) + sqr(sigma[2] - 1);
                //e = (sig.array()-1).matrix().squaredNorm();
                break;
        }

        en += w[ii] * e;

        if (Eg) {
            Vector3d partialE;
            double lambda[3], as[3], bs[3];
            switch (energyType) {
            case 1: {    //Eiso
                for (int j = 0; j < 3; j++) {
                    lambda[j] = 2 * (1 + 3 / pow4(sigma[j]));
                    int id1 = (j * 2) % 3, id2 = (id1 + 1) % 3;
                    as[j] = 1 + 1 / sqr(sigma[id1] * sigma[id2]);
                    bs[j] = (sqr(sigma[id1]) + sqr(sigma[id2])) / pow3(sigma[id1] * sigma[id2]);
                }

                partialE << sigma[0] - 1 / pow3(sigma[0]), sigma[1] - 1 / pow3(sigma[1]), sigma[2] - 1 / pow3(sigma[2]);
                break;
            }
            case 5: {
                for (int j = 0; j < 3; j++) {
                    lambda[j] = 2;
                    int id1 = (j * 2) % 3, id2 = (id1 + 1) % 3;
                    as[j] = 1 - 1 / (sigma[id1] + sigma[id2]);
                    bs[j] = 1 / (sigma[id1] + sigma[id2]);
                }

                partialE << sigma[0] - 1, sigma[1] - 1, sigma[2] - 1;
                break;
            }
            }


            Matrix3d U;
            Matrix3d V;
            U << Su11.f, Su12.f, Su13.f, Su21.f, Su22.f, Su23.f, Su31.f, Su32.f, Su33.f;
            V << Sv11.f, Sv12.f, Sv13.f, Sv21.f, Sv22.f, Sv23.f, Sv31.f, Sv32.f, Sv33.f;

            Matrix<double, 4, 3> U2 = X2Js.block<3,4>(ii*3, 0).transpose()*U;
            Matrix<double, 12, 9> p;
            p << U2(0, 0)*V, U2(0, 1)*V, U2(0, 2)*V, U2(1, 0)*V, U2(1, 1)*V, U2(1, 2)*V, U2(2, 0)*V, U2(2, 1)*V, U2(2, 2)*V, U2(3, 0)*V, U2(3, 1)*V, U2(3, 2)*V;

            Matrix<double, 12, 9> M;
            M << p.col(0), p.col(4), p.col(8), p.col(1), p.col(3), p.col(2), p.col(6), p.col(5), p.col(7);

            partialE *= 2;
            Map<MatrixXd>(Eg + ii * 12, 12, 1) = w[ii] * M.block<12, 3>(0, 0)*partialE;

            if (Eh) {
                Matrix<double, 9, 9> K = MatrixXd::Zero(9, 9);
                for (int j = 0; j < 3; j++) {
                    K(j, j) = lambda[j];
                    double a = as[j], b = bs[j];
                    if (a > b)
                        K.block<2, 2>(3 + 2 * j, 3 + 2 * j) << 2 * a, 2 * b, 2 * b, 2 * a;
                    else
                        K.block<2, 2>(3 + 2 * j, 3 + 2 * j) << a + b, a + b, a + b, a + b;
                }

                Map<MatrixXd>(Eh + ii * 12 * 12, 12, 12) = w[ii] * M*K*M.transpose();
            }
        }
    }
    
    return en;
}

// [E, Eg, Eh] = isometricEnergyFromJ3D(Js, X2J, w, et)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    int nt = mxGetM(prhs[2]);

    double *g = nullptr, *h = nullptr;

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(12, nt, mxREAL);
        g = mxGetPr(plhs[1]);
    }
    if (nlhs > 2) {
        plhs[2] = mxCreateDoubleMatrix(12, 12 * nt, mxREAL);
        h = mxGetPr(plhs[2]);
    }

    double en = isometricEnergyFromJ3D(g, h, mxGetPr(prhs[0]), mxGetPr(prhs[1]), mxGetPr(prhs[2]), (int)mxGetScalar(prhs[3]), nt);
    plhs[0] = mxCreateDoubleScalar(en);
}
