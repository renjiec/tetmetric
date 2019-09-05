#include <mex.h>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/LU>

template<class mat, class matc>
void block_inverse(matc &A, mat &B)
{
    const size_t dim = A.cols();
    const size_t n = A.rows() / dim;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (dim == 2)
            B.block<2, 2>(i * 2, 0).noalias() = A.block<2, 2>(i * 2, 0).inverse();
        else if (dim == 3)
            B.block<3, 3>(i * 3, 0).noalias() = A.block<3, 3>(i * 3, 0).inverse();
        else if (dim == 4)
            B.block<4, 4>(i * 4, 0).noalias() = A.block<4, 4>(i * 4, 0).inverse();
        else if (dim == 6)
            B.block<6, 6>(i * 6, 0).noalias() = A.block<6, 6>(i * 6, 0).inverse();
        else
            B.middleRows(i*dim, dim).noalias() = A.middleRows(i*dim, dim).inverse();
    }
}

template<class mat, class matc>
void block_transpose(matc &A, mat &B)
{
    const size_t dim = A.cols();
    const size_t n = A.rows() / dim;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        B.middleRows(i*dim, dim).noalias() = A.middleRows(i*dim, dim).transpose();
    }
}

template<class mat, class matc>
void block_gemm(matc &A, matc &B, mat &C)
{
    const size_t dim = A.cols();
    const size_t n = A.rows() / dim;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (dim == 2)
            C.block<2, 2>(i * 2, 0).noalias() = A.block<2, 2>(i * 2, 0)*B.block<2, 2>(i * 2, 0);
        else if (dim == 3)
            C.block<3, 3>(i * 3, 0).noalias() = A.block<3, 3>(i * 3, 0)*B.block<3, 3>(i * 3, 0);
        else if (dim == 4)
            C.block<4, 4>(i * 4, 0).noalias() = A.block<4, 4>(i * 4, 0)*B.block<4, 4>(i * 4, 0);
        else
            C.middleRows(i*dim, dim).noalias() = A.middleRows(i*dim, dim) * B.middleRows(i*dim, dim);
    }
}

template<class mat, class matc>
void block_gemm_at(matc &A, matc &B, mat &C)
{
    const size_t dim = A.cols();
    const size_t n = A.rows() / dim;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        C.middleRows(i*dim, dim).noalias() = A.middleRows(i*dim, dim).transpose() * B.middleRows(i*dim, dim);
    }
}

template<class mat, class matc>
void block_solve(matc &A, matc &B, mat &C)
{
    const size_t dim = A.cols();
    const size_t n = A.rows() / dim;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (dim == 2)
            C.block<2, 2>(i * 2, 0).noalias() = A.block<2, 2>(i * 2, 0).inverse()*B.block<2, 2>(i * 2, 0);
        else if (dim == 3)
            C.block<3, 3>(i * 3, 0).noalias() = A.block<3, 3>(i * 3, 0).inverse()*B.block<3, 3>(i * 3, 0);
        else if (dim == 4)
            C.block<4, 4>(i * 4, 0).noalias() = A.block<4, 4>(i * 4, 0).inverse()*B.block<4, 4>(i * 4, 0);
        else
            C.middleRows(i*dim, dim).noalias() = A.middleRows(i*dim, dim).inverse() * B.middleRows(i*dim, dim);
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    // block blas operation for square matrices
    if (nrhs<1) mexErrMsgTxt("syntax: C = blockblas(blasop, A[, B]);");

    if (nrhs<2) mexErrMsgTxt("not enough inputs!");

    if (!mxIsChar(prhs[0])) mexErrMsgTxt("blasop should be a string specifies the blas operation.");

    std::basic_string_view<mxChar> blasop(mxGetChars(prhs[0]), mxGetNumberOfElements(prhs[0]));

    if (!mxIsNumeric(prhs[1])) mexErrMsgTxt("input A should be numerical matrix.");

    const mxArray *matA = prhs[1];
    const size_t dim = mxGetN(matA);
    const size_t m = mxGetM(matA);
    const size_t n = m / dim;

    if (n * dim != m) mexErrMsgTxt("input matrix(s) must be blocks of square matrices.");
    const mxArray *matB = nrhs>2?prhs[2]:nullptr;

    plhs[0] = mxDuplicateArray(matA);

    if (mxIsDouble(matA)) {
        using mapmat = Eigen::Map<Eigen::MatrixXd>;
        using mapmatc = Eigen::Map<const Eigen::MatrixXd>;

        mapmatc A(mxGetDoubles(matA), m, dim);
        mapmat C(mxGetDoubles(plhs[0]), m, dim);

        // using u literal for char16_t that Matlab uses
        if (!blasop.compare(u"transpose")) 
            block_transpose(A, C);
        else if (!blasop.compare(u"inverse")) 
            block_inverse(A, C);
        else {
            if (!matB) mexErrMsgTxt("input matrix B is missing for the specified blas operation.");

            if (!mxIsDouble(matB) || mxGetM(matB) != m || mxGetN(matB) != dim) mexErrMsgTxt("matrix B should be of the same size and type as matrix A.");

            mapmatc B(mxGetDoubles(matB), m, dim);
            if (!blasop.compare(u"multiply")) 
                block_gemm(A, B, C);
            else if (!blasop.compare(u"solve")) 
                block_solve(A, B, C);
            else if (!blasop.compare(u"transpose_A_and_multiply")) 
                block_gemm_at(A, B, C);
            else 
                mexErrMsgTxt("unsupported blas op");
        }
    }
    else if (mxIsSingle(matA)) {
        using mapmat = Eigen::Map<Eigen::MatrixXf>;
        using mapmatc = Eigen::Map<const Eigen::MatrixXf>;

        mapmatc A(mxGetSingles(matA), m, dim);
        mapmat C(mxGetSingles(plhs[0]), m, dim);

        // using u literal for char16_t that Matlab uses
        if (!blasop.compare(u"transpose")) 
            block_transpose(A, C);
        else if (!blasop.compare(u"inverse")) 
            block_inverse(A, C);
        else {
            if (!matB) mexErrMsgTxt("input matrix B is missing for the specified blas operation.");

            if (!mxIsSingle(matB) || mxGetM(matB) != m || mxGetN(matB) != dim) mexErrMsgTxt("matrix B should be of the same size and type as matrix A.");

            mapmatc B(mxGetSingles(matB), m, dim);
            if (!blasop.compare(u"multiply")) 
                block_gemm(A, B, C);
            else if (!blasop.compare(u"solve")) 
                block_solve(A, B, C);
            else if (!blasop.compare(u"transpose_A_and_multiply")) 
                block_gemm_at(A, B, C);
            else 
                mexErrMsgTxt("unsupported blas op");
        }
    }
    else 
        mexErrMsgTxt("check numeric type of matrix A, only single and double are supported.");
}
