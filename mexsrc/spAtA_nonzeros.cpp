#ifndef MATLAB_DEFAULT_RELEASE
#define MATLAB_DEFAULT_RELEASE R2018a
#endif

#include <mex.h>

namespace {              
using index_t = size_t;
// https://stackoverflow.com/questions/32753310/how-to-get-dot-product-of-two-sparsevectors-in-omn-where-m-and-n-are-the-nu
double spdot_sorted(size_t nnz1, const index_t *i1, const double *v1, size_t nnz2, const index_t* i2, const double *v2)
{
    double dotP = 0;

    for (int i = 0, j = 0; i < nnz1; i++) {
        while (j<nnz2 && i2[j] < i1[i]) j++;
        if (j >= nnz2) break;

        if (i2[j] == i1[i]) dotP += v1[i] * v2[j];
    }
    return dotP;
}

void sparse_sypr_nonzeros(size_t rows, size_t cols, const index_t* Ajc, const index_t* Air, const double *Aval, const index_t* Mjc, const index_t* Mir, double *Mval)
{
    // M = At*A, A in 0-based csc format (column major as in Matlab)

#pragma omp parallel for //collapse(2)  // shared(Ajc, Air, Aval, Mjc, Mir)  
    for (ptrdiff_t c2 = 0; c2 < cols; c2++) {
        index_t c2start = Ajc[c2];
        size_t n2 = Ajc[c2 + 1] - c2start;
        for (index_t j = Mjc[c2]; j < Mjc[c2 + 1]; j++) {
            index_t c1 = Mir[j];
            //if (c2 < c1) continue; // compute only upper triangle
            index_t c1start = Ajc[c1];
            size_t n1 = Ajc[c1 + 1] - c1start;
            Mval[j] = spdot_sorted(n1, Air + c1start, Aval + c1start, n2, Air + c2start, Aval + c2start);
        }}
}


// replace mxGetNzmax with nnz, as mxGetNzmax has over-allocated elements (zeros)
size_t nnz(const mxArray* m) { return *(mxGetJc(m) + mxGetN(m)); }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2) mexErrMsgTxt("Invalid input: not enough input, C = spAtA_nonzeros(A, AtA0 [, returnSpMat=false] );");

    const mxArray *A = prhs[0];
    const mxArray *AtA = prhs[1];

    size_t cols = mxGetN(A);
    if (cols != mxGetM(AtA) || cols != mxGetN(AtA) )
        mexErrMsgTxt("template matrix (AtA0) does not have matching dimension as A.");

    plhs[0] = (nrhs>2 && mxIsLogicalScalarTrue(prhs[2]))?mxDuplicateArray(AtA):mxCreateDoubleMatrix(nnz(AtA), 1, mxREAL);
    sparse_sypr_nonzeros(mxGetM(A), cols, mxGetJc(A), mxGetIr(A), mxGetDoubles(A), mxGetJc(AtA), mxGetIr(AtA), mxGetDoubles(plhs[0]));
}

