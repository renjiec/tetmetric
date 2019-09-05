#include <mex.h>
#include <algorithm>

void mexFunction(int nlhs, mxArray *plhs[],	int nrhs, const mxArray*prhs[])
{
    if (nrhs < 2) 
        mexErrMsgTxt("Invalid input: not enough input, x = replaceNonzeros(A, nzvals);");

    const mxArray* mat_A = prhs[0];
    if( !mxIsSparse(mat_A) )
        mexErrMsgTxt("Invalid input: Matrix A should be sparse!");

    const mxArray* mat_nzvals = prhs[1];

    if( mxIsSparse(mat_nzvals) )
        mexErrMsgTxt("Invalid input: vector nonzeros should be dense!");

    if (mxGetClassID(mat_nzvals) != mxDOUBLE_CLASS || mxIsComplex(mat_nzvals))
        mexErrMsgTxt("Invalid input: vector nzvals should be of real & double type!");

    const size_t n = mxGetN(mat_A);
    const size_t nnz = *(mxGetJc(mat_A)+n); 

    if (nnz != mxGetNumberOfElements(mat_nzvals))
        mexErrMsgTxt("Invalid input: matrix A does not have matching number of nonzeros with the size of nzvals");

    plhs[0] = mxDuplicateArray(mat_A);
    std::copy_n(mxGetDoubles(mat_nzvals), nnz, mxGetDoubles(plhs[0]));
}