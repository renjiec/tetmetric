#include <mex.h>
#include <algorithm>
#include <execution>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    if (nrhs < 2) 
        mexErrMsgTxt("Invalid input: not enough input, x = myaccumarray(idx, val, initArray);");

    const mxArray* mat_idx = prhs[0];
    const mxArray* mat_val = prhs[1];

    if (mxGetClassID(mat_idx) != mxINDEX_CLASS || mxGetClassID(mat_val) != mxDOUBLE_CLASS)        // for 64bit
        mexErrMsgTxt("Invalid input: matrix idx should be of index type (uint64 on x64), and val should be of double type!");

    const size_t nval = mxGetNumberOfElements(mat_val);

    if (mxGetNumberOfElements(mat_idx)!=nval)        // for 64bit
        mexErrMsgTxt("Invalid input: matrix idx and val should have the same length!");


    const double *pvals = mxGetPr(mat_val);
    const size_t* pidx = (size_t*)mxGetData(mat_idx);

    if (nrhs > 2) {
        if (mxGetClassID(prhs[2]) == mxDOUBLE_CLASS)
            plhs[0] = mxDuplicateArray(prhs[2]); // given initial array
        else if (mxGetClassID(prhs[2]) == mxUINT64_CLASS && mxGetNumberOfElements(prhs[2])==1 )
            plhs[0] = mxCreateDoubleMatrix(*mxGetUint64s(prhs[2]), 1, mxREAL); // automatically initialized with 0
        else
            mexErrMsgTxt("Invalid input: initial array should be of double type, or single element uint64 to specify size!");
    }
    else {
        size_t n = *std::max_element(std::execution::par, pidx, pidx + nval);
        plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL); // automatically initialized with 0
        //std::fill_n(std::execution::par, mxGetPr(plhs[0]), n, 0);
    }

    double* r = mxGetDoubles(plhs[0]);
#pragma omp parallel for shared(r, pidx, pvals)
    for (ptrdiff_t i = 0; i < nval; i++) {
        size_t j = pidx[i] - 1;
#pragma omp atomic
        r[j] += pvals[i];                // notice the -1: convert matlab index to c
    }
}
