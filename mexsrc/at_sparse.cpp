#define MATLAB_DEFAULT_RELEASE R2017b

#include <mex.h>
#include <algorithm>
#include <limits>

void mexFunction(int nlhs, mxArray *plhs[],	int nrhs, const mxArray*prhs[])
{
    if (nrhs < 3) 
        mexErrMsgTxt("Invalid input: not enough input, idx = at_sparse(A, i, j);");

    const mxArray* mat_A = prhs[0];
    const mxArray* mat_I = prhs[1];
    const mxArray* mat_J = prhs[2];

    if( !mxIsSparse(mat_A) )
        mexErrMsgTxt("Invalid input: Matrix A should be sparse!");

    if( mxIsSparse(mat_I) || mxIsSparse(mat_J) )
        mexErrMsgTxt("Invalid input: matrix I and J should be dense!");

    //if ( !mxIsUint64(mat_I) || !mxIsUint64(mat_J) )        // for 64bit
    if (mxGetClassID(mat_I) != mxINDEX_CLASS || mxGetClassID(mat_J) != mxINDEX_CLASS)        // for 64bit
        mexErrMsgTxt("Invalid input: matrix I and J should be of index type (uint64 on x64)!");

    const size_t k = mxGetNumberOfElements(mat_I);
    
    if (k != mxGetNumberOfElements(mat_J))
        mexErrMsgTxt("Invalid input: matrix I and J should have the same number of entries!");


    const size_t *A_Ir = mxGetIr(mat_A);
    const size_t *A_Jc = mxGetJc(mat_A);
    const double *A_nonzeros = mxGetPr(mat_A);
    const size_t m = mxGetM(mat_A);
    const size_t n = mxGetN(mat_A);

    const size_t *pI = (size_t*)mxGetData(mat_I);
    const size_t *pJ = (size_t*)mxGetData(mat_J);

    plhs[0] = mxCreateNumericArray(mxGetNumberOfDimensions(mat_I), mxGetDimensions(mat_I), mxDOUBLE_CLASS, mxREAL); // todo: fix for complex matrix
    //double *vals = mxGetDoubles(plhs[0]);  // 2018a api
    double *vals = mxGetPr(plhs[0]);

    const auto intmax = std::numeric_limits<size_t>::max();

    const int offset = 1; // for matlab 1-based index
#pragma omp parallel for shared(A_Ir, A_Jc, pI, pJ, vals)
    for (int i = 0; i < k; i++) {
        const size_t row = pI[i] - offset;
        const size_t col = pJ[i] - offset;

        if (row >= m || col >= n)
            mexErrMsgTxt("index out of bound!");

        const auto pstart = A_Ir+A_Jc[col];
        const auto pend   = A_Ir+A_Jc[col + 1];
        const auto it = std::find(pstart, pend, row);

        vals[i] = (it == pend) ? 0 : A_nonzeros[it - A_Ir];
    }
}
