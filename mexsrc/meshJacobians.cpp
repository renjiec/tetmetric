#ifndef MATLAB_DEFAULT_RELEASE
#define MATLAB_DEFAULT_RELEASE R2018a
#endif

#include <mex.h>
#include <Eigen/Core>
#include <Eigen/LU>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    // block blas operation for square matrices
    if (nrhs<1) mexErrMsgTxt("syntax: J = meshJacobians(x, y, t);");

    if (nrhs<3) mexErrMsgTxt("not enough inputs! syntax: J = meshJacobians(x, y, t);");

    if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]))
        mexErrMsgTxt("input x and y should be of type double.");

    if ( mxGetClassID(prhs[2])!=mxINDEX_CLASS ) 
        mexErrMsgTxt("input t should be of index type (uint64 on x64).");

    const mxArray *matx = prhs[0], *maty = prhs[1], *matt = prhs[2];

    const size_t nv = mxGetM(matx);
    const size_t dim = mxGetN(matx);
    if (dim != 3) mexErrMsgTxt("x should have 3 columns.");

    if (mxGetN(maty) != dim || mxGetM(maty) != nv)  
        mexErrMsgTxt("y should have matching dimensions as x.");

    if ( mxGetN(matt)!=dim+1 ) mexErrMsgTxt("t should have 4 columns.");

    const size_t nt = mxGetM(matt);

    plhs[0] = mxCreateDoubleMatrix(nt * dim, dim, mxREAL);

    using MatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3>;
    using MatX4i = Eigen::Matrix<mwIndex, Eigen::Dynamic, 4>;
    using mapmat = Eigen::Map<MatX3d>;
    using mapmatc = Eigen::Map<const MatX3d>;

    mapmatc x(mxGetDoubles(matx), nv, dim);
    mapmatc y(mxGetDoubles(maty), nv, dim);
    Eigen::Map<const MatX4i> t((const mwIndex*)mxGetData(matt), nt, dim+1);
    //Eigen::Map<const MatX4i> t(mxGetUint64s(matt), nt, dim+1);
    mapmat J(mxGetDoubles(plhs[0]), nt*dim, dim);

    mxAssert(t.minCoeff() >= 0 && t.maxCoeff() < nv, "error, some vertex(s) has index out of bounds");

#pragma omp parallel for
    for (int i = 0; i < nt; i++) {
        Eigen::Matrix3d M1, M2;
        M1 << x.row(t(i, 1)) - x.row(t(i, 0)), x.row(t(i, 2)) - x.row(t(i, 0)), x.row(t(i, 3)) - x.row(t(i, 0));
        M2 << y.row(t(i, 1)) - y.row(t(i, 0)), y.row(t(i, 2)) - y.row(t(i, 0)), y.row(t(i, 3)) - y.row(t(i, 0));
        J.block<3, 3>(i * 3, 0) = M1.inverse()*M2;
    }
}
