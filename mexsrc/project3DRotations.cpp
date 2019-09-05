#include <mex.h>
#include <omp.h>
#include <cmath>
#include <string_view>
#include <Eigen/Core>

//#define PRINT_DEBUGGING_OUTPUT
#define COMPUTE_V_AS_MATRIX
//#define COMPUTE_V_AS_QUATERNION
#define COMPUTE_U_AS_MATRIX
//#define COMPUTE_U_AS_QUATERNION


#define USE_AVX_IMPLEMENTATION
#include "Singular_Value_Decomposition_Preamble.hpp"
void projBlock3DRotationAvx(float *A, size_t n)
{
    const int step = 8;
#pragma omp parallel for
    for (int i = 0; i < n; i += step) {
        float *pv = A + i;
       
#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
        Va11 = _mm256_loadu_ps(pv);
        Va21 = _mm256_loadu_ps(pv+n);
        Va31 = _mm256_loadu_ps(pv+2*n);
        Va12 = _mm256_loadu_ps(pv+3*n);
        Va22 = _mm256_loadu_ps(pv+4*n);
        Va32 = _mm256_loadu_ps(pv+5*n);
        Va13 = _mm256_loadu_ps(pv+6*n);
        Va23 = _mm256_loadu_ps(pv+7*n);
        Va33 = _mm256_loadu_ps(pv+8*n);
        
#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"
        _mm256_storeu_ps(pv,     _mm256_fmadd_ps(Vu11, Vv11, _mm256_fmadd_ps(Vu12, Vv12, _mm256_mul_ps(Vu13, Vv13))) );
        _mm256_storeu_ps(pv+1*n, _mm256_fmadd_ps(Vu21, Vv11, _mm256_fmadd_ps(Vu22, Vv12, _mm256_mul_ps(Vu23, Vv13))) );
        _mm256_storeu_ps(pv+2*n, _mm256_fmadd_ps(Vu31, Vv11, _mm256_fmadd_ps(Vu32, Vv12, _mm256_mul_ps(Vu33, Vv13))) );
        _mm256_storeu_ps(pv+3*n, _mm256_fmadd_ps(Vu11, Vv21, _mm256_fmadd_ps(Vu12, Vv22, _mm256_mul_ps(Vu13, Vv23))) );
        _mm256_storeu_ps(pv+4*n, _mm256_fmadd_ps(Vu21, Vv21, _mm256_fmadd_ps(Vu22, Vv22, _mm256_mul_ps(Vu23, Vv23))) );
        _mm256_storeu_ps(pv+5*n, _mm256_fmadd_ps(Vu31, Vv21, _mm256_fmadd_ps(Vu32, Vv22, _mm256_mul_ps(Vu33, Vv23))) );
        _mm256_storeu_ps(pv+6*n, _mm256_fmadd_ps(Vu11, Vv31, _mm256_fmadd_ps(Vu12, Vv32, _mm256_mul_ps(Vu13, Vv33))) );
        _mm256_storeu_ps(pv+7*n, _mm256_fmadd_ps(Vu21, Vv31, _mm256_fmadd_ps(Vu22, Vv32, _mm256_mul_ps(Vu23, Vv33))) );
        _mm256_storeu_ps(pv+8*n, _mm256_fmadd_ps(Vu31, Vv31, _mm256_fmadd_ps(Vu32, Vv32, _mm256_mul_ps(Vu33, Vv33))) );
    }
}
#undef USE_AVX_IMPLEMENTATION

#define USE_SSE_IMPLEMENTATION
#include "Singular_Value_Decomposition_Preamble.hpp"
void projBlock3DRotationSse(float *A, size_t n)
{
    const int step = 4;
#pragma omp parallel for
    for (int i = 0; i < n; i += step) {
        float *pv = A + i;
       
#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
        Va11 = _mm_loadu_ps(pv);
        Va21 = _mm_loadu_ps(pv+n);
        Va31 = _mm_loadu_ps(pv+2*n);
        Va12 = _mm_loadu_ps(pv+3*n);
        Va22 = _mm_loadu_ps(pv+4*n);
        Va32 = _mm_loadu_ps(pv+5*n);
        Va13 = _mm_loadu_ps(pv+6*n);
        Va23 = _mm_loadu_ps(pv+7*n);
        Va33 = _mm_loadu_ps(pv+8*n);
        
#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"
        _mm_storeu_ps(pv,     _mm_fmadd_ps(Vu11, Vv11, _mm_fmadd_ps(Vu12, Vv12, _mm_mul_ps(Vu13, Vv13))) );
        _mm_storeu_ps(pv+1*n, _mm_fmadd_ps(Vu21, Vv11, _mm_fmadd_ps(Vu22, Vv12, _mm_mul_ps(Vu23, Vv13))) );
        _mm_storeu_ps(pv+2*n, _mm_fmadd_ps(Vu31, Vv11, _mm_fmadd_ps(Vu32, Vv12, _mm_mul_ps(Vu33, Vv13))) );
        _mm_storeu_ps(pv+3*n, _mm_fmadd_ps(Vu11, Vv21, _mm_fmadd_ps(Vu12, Vv22, _mm_mul_ps(Vu13, Vv23))) );
        _mm_storeu_ps(pv+4*n, _mm_fmadd_ps(Vu21, Vv21, _mm_fmadd_ps(Vu22, Vv22, _mm_mul_ps(Vu23, Vv23))) );
        _mm_storeu_ps(pv+5*n, _mm_fmadd_ps(Vu31, Vv21, _mm_fmadd_ps(Vu32, Vv22, _mm_mul_ps(Vu33, Vv23))) );
        _mm_storeu_ps(pv+6*n, _mm_fmadd_ps(Vu11, Vv31, _mm_fmadd_ps(Vu12, Vv32, _mm_mul_ps(Vu13, Vv33))) );
        _mm_storeu_ps(pv+7*n, _mm_fmadd_ps(Vu21, Vv31, _mm_fmadd_ps(Vu22, Vv32, _mm_mul_ps(Vu23, Vv33))) );
        _mm_storeu_ps(pv+8*n, _mm_fmadd_ps(Vu31, Vv31, _mm_fmadd_ps(Vu32, Vv32, _mm_mul_ps(Vu33, Vv33))) );
    }
}
#undef USE_SSE_IMPLEMENTATION


#define USE_SCALAR_IMPLEMENTATION
#include "Singular_Value_Decomposition_Preamble.hpp"
template<typename R>
void projBlock3DRotationScalar(R *A, size_t n)
{
    const size_t rows = n * 3;

#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        R *const M = A + i * 3;

#include "Singular_Value_Decomposition_Kernel_Declarations.hpp"
        Sa11.f = (float)M[0];
        Sa21.f = (float)M[rows];
        Sa31.f = (float)M[rows*2];
        Sa12.f = (float)M[1];
        Sa22.f = (float)M[1 + rows];
        Sa32.f = (float)M[1 + rows*2];
        Sa13.f = (float)M[2];
        Sa23.f = (float)M[2 + rows];
        Sa33.f = (float)M[2 + rows*2];

#include "Singular_Value_Decomposition_Main_Kernel_Body.hpp"   
         float u11 = Su11.f, u21 = Su21.f, u31 = Su31.f;
         float u12 = Su12.f, u22 = Su22.f, u32 = Su32.f;
         float u13 = Su13.f, u23 = Su23.f, u33 = Su33.f;

         float v11 = Sv11.f, v21 = Sv21.f, v31 = Sv31.f;
         float v12 = Sv12.f, v22 = Sv22.f, v32 = Sv32.f;
         float v13 = Sv13.f, v23 = Sv23.f, v33 = Sv33.f;

         //float sigma1 = Sa11.f, sigma2 = Sa22.f, sigma3 = Sa33.f;
         M[0] = u11 * v11 + u12 * v12 + u13 * v13;
         M[1] = u11 * v21 + u12 * v22 + u13 * v23;
         M[2] = u11 * v31 + u12 * v32 + u13 * v33;
         M[    rows] = u21 * v11 + u22 * v12 + u23 * v13;
         M[1 + rows] = u21 * v21 + u22 * v22 + u23 * v23;
         M[2 + rows] = u21 * v31 + u22 * v32 + u23 * v33;
         M[    2 * rows] = u31 * v11 + u32 * v12 + u33 * v13;
         M[1 + 2 * rows] = u31 * v21 + u32 * v22 + u33 * v23;
         M[2 + 2 * rows] = u31 * v31 + u32 * v32 + u33 * v33;
    }
}
#undef USE_SCALAR_IMPLEMENTATION

template<int dim, typename R, typename R2>
void reshapeBlocksToRows(const R* A, size_t blocksA, R2* B, size_t rowsB)
{
    // A : (blocksA*dim) * dim  -> B : rowsB * (dim^2)

    using namespace Eigen;
    using mapmatc = Map<const Matrix<R, Dynamic, dim>, 0, InnerStride<dim>>;
    using mapmat = Map<Matrix<R2, Dynamic, dim>, 0, OuterStride<>>;
    const size_t n = blocksA;
#pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        mapmat(B + rowsB * i*dim, n, dim, OuterStride<>(rowsB)) = mapmatc(A + i, n, dim).cast<R2>();
    }
}

template<int dim, typename R, typename R2>
void reshapeRowsToBlocks(const R* A, size_t rowsA, R2* B, size_t blocksB)
{
    // A : rowsA * (dim^2)  -> B : (blocksB*dim) * dim

    using namespace Eigen;
    using mapmatc = Map<const Matrix<R, Dynamic, dim>, 0, OuterStride<>>;
    using mapmat = Map<Matrix<R2, Dynamic, dim>, 0, InnerStride<dim>>;
    const size_t n = blocksB;
#pragma omp parallel for
    for (int i = 0; i < dim; i++) {
        mapmat(B + i, n, dim) = mapmatc(A + rowsA*i*dim, n, dim, OuterStride<>(rowsA)).cast<R2>();
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    if (nrhs<1) mexErrMsgTxt("R = project3DRotations(A, method); \nmethod can be scalar, sse or avx (default)'");

    if (!mxIsNumeric(prhs[0]) || mxIsComplex(prhs[0])) mexErrMsgTxt("input should be a real matrix");
    if (!mxIsSingle(prhs[0]) && !mxIsDouble(prhs[0])) mexErrMsgTxt("must be floating point matrix, i.e. single or double");

    bool singlePrecision = mxIsSingle(prhs[0]);

    std::u16string_view method = u"avx";
    if (nrhs > 1) {
        if (!mxIsChar(prhs[1])) mexErrMsgTxt("method should be a char array.");

        method = { mxGetChars(prhs[1]), mxGetNumberOfElements(prhs[1]) };
    }

    size_t m = mxGetM(prhs[0]); // # rows of A
    if (mxGetN(prhs[0]) != 3 || m/3*3 != m) mexErrMsgTxt("incorrect matrix dimension, should 3nx3 matrix");
    
    //init output
    plhs[0] = mxDuplicateArray(prhs[0]);

    if (!method.compare(u"scalar")) {
        if (singlePrecision)
            projBlock3DRotationScalar(mxGetSingles(plhs[0]), m / 3);
        else
            projBlock3DRotationScalar(mxGetDoubles(plhs[0]), m / 3);
    }
    else {
        bool avx = !method.compare(u"avx");

        // AVX step = 8
        const int vecstep = avx ? 8 : 4;
        size_t paddrows = (m / 3 + vecstep - 1) / vecstep * vecstep;

        Eigen::MatrixXf A2(paddrows, 9);  // pad A2 with 0s

        // reshape to rows
        if (singlePrecision)
            reshapeBlocksToRows<3>(mxGetSingles(prhs[0]), m / 3, A2.data(), A2.rows());
        else
            reshapeBlocksToRows<3>(mxGetDoubles(prhs[0]), m / 3, A2.data(), A2.rows());

        // project
        if (avx)
            projBlock3DRotationAvx(A2.data(), A2.rows());
        else
            projBlock3DRotationSse(A2.data(), A2.rows());

        // reshape back to blocks
        if (singlePrecision)
            reshapeRowsToBlocks<3>(A2.data(), A2.rows(), mxGetSingles(plhs[0]), m / 3);
        else
            reshapeRowsToBlocks<3>(A2.data(), A2.rows(), mxGetDoubles(plhs[0]), m / 3);
    }
}
