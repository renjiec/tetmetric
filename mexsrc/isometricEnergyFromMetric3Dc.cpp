#include "mex.h"
#include <Eigen/Core>
#include <omp.h>  

using namespace Eigen;

template<class real>
double sqr(real x) { return x*x; };

Matrix<double, 6, 6> computeHess(double a, double b, double c, double d, double e, double f)
{
	double x0 = 2 * b;
	double x1 = 2 * c;
	double x2 = -2 * e;
	double x3 = 2 * e;
	double x4 = d * f - e * e;
	double x5 = b * b;
	double x6 = c * c;
	double x7 = c * b;
	double x8 = a * x4 - d * x6 - f * x5 + x7 * x3;
	double x9 = 2 * x8;
	double x10 = d + f;
	double x11 = a * x10 - x5 + x4 - x6;
	double x12 = x8 * x8;
	double x13 = 1 / x12;
	double x14 = x11 * x13 * x13;
	double x15 = -x4 * x13;
	double x16 = c * x2 + f * x0;
	double x17 = x16 * x13;
	double x18 = b * x2 + d * x1;
	double x19 = x18 * x13;
	double x20 = -a * f + x6;
	double x21 = x20 * x13;
	double x22 = -a * x2 - 2.0 * x7;
	double x23 = x22 * x13;
	double x24 = -a * d + x5;
	double x25 = x24 * x13;
	double x26 = 1 / x8;
	double x27 = -2 * x26;
	double x28 = x11 * x13;
	double x29 = x28 * d;
	double x30 = x28 * f;
	double x31 = -a * x28 + x26;
	double x41 = x14 * x9;
	double x32 = x4 * x41 - x10 * x13;
	double x33 = x32 * x3;
	double x34 = a * x32 - x28 + x15;
	double x35 = a * x15 + x26;
	double x36 = x0 * x13 - x16 * x41;
	double x37 = x36 * x3;
	double x38 = a * x36 + x17;
	double x39 = a * x17;
	double x40 = x1 * x13 + -x18 * x41;
	double x42 = x40 * x3;
	double x43 = a * x40 + x19;
	double x44 = a * x19;
	double x45 = -x13 * (a + f) - x20 * x41;
	double x46 = x45 * x3;
	double x47 = a * x45 + x21;
	double x48 = a * x21;
	double x49 = -x2 * x13 - x22 * x41;
	double x50 = 2.0 * e * x49 - 2.0 * x28;
	double x51 = a * x49 + x23;
	double x52 = a * x23;
	double x53 = -(a + d) * x13 + -x24 * x41;
	double x54 = x53 * x3;
	double x55 = a * x53 + x25;
	double x56 = a * x25;
	double x57 = x28 * x3;

    Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
	H(0, 0) = x4 * x32 + x10 * x15;
	H(0, 3) = f * x34 - x32 * x6 + x35;
	H(0, 4) = 2.0 * e * -x34 + 2.0 * x7 * x32;
	H(0, 5) = d * x34 - x5 * x32 + x35;

	H(1, 0) = x4 * x36 + x10 * x17;
	H(1, 1) = c * x37 + x0 * (-f * x36 - x17) + 2.0 * x30 + x27;
	H(1, 2) = b * x37 + x1 * (-d * x36 - x17) - x57;
	H(1, 3) = f * x38 - x36 * x6 + x39;

	H(2, 0) = x4 * x40 + x10 * x19;
	H(2, 2) = b * x42 + x1 * (-d * x40 - x19) + 2.0 * x29 + x27;
	H(2, 5) = d * x43 - x5 * x40 + x44;

	H(3, 2) = b * x46 + x1 * (-d * x45 + x28 - x21);
	H(3, 3) = f * x47 - x6 * x45 + x48;
	H(3, 4) = 2.0 * (e * -x47 + x7 * x45);
	H(3, 5) = d * x47 - x5 * x45 + x48 + x31;

	H(4, 1) = c * x50 + x0 * (-f * x49 - x23);
	H(4, 2) = b * x50 + x1 * (-d * x49 - x23);
	H(4, 4) = 2.0 * e * -x51 + 2.0 * x49 * x7 - 2.0 * x31;

	H(5, 1) = c * x54 + x0 * (-f * x53 - x25 + x28);
	H(5, 4) = 2.0 * (e * -x55 + x53 * x7);
	H(5, 5) = d * x55 - x5 * x53 + x56;

	H(0, 1) = H(1, 0);		H(0, 2) = H(2, 0);		H(1, 4) = H(4, 1);		H(1, 5) = H(5, 1);		H(2, 1) = H(1, 2);
	H(2, 3) = H(3, 2);		H(2, 4) = H(4, 2);		H(3, 0) = H(0, 3);		H(3, 1) = H(1, 3);		H(4, 0) = H(0, 4);
	H(4, 3) = H(3, 4);		H(4, 5) = H(5, 4);		H(5, 0) = H(0, 5);		H(5, 2) = H(2, 5);		H(5, 3) = H(3, 5);
    return H;
}

Matrix<double, 6, 1> computeGrad(double a, double b, double c, double d, double e, double f)
{
	double x0 = d + f;
	double x1 = d * f - e * e;
	double x2 = c * c;
	double x3 = b * b;
	double x4 = c * b;
	double x5 = a * x1 - d * x2 + 2 * e * x4 - f * x3;
	double x6 = 1 / x5;
	double x7 = (a * x0 + x1 - x2 - x3) * x6 * x6;
	double x8 = -a * x7 + x6;
	double x9 = a * x6 + 1;

    Matrix<double, 6, 1> G = Matrix<double, 6, 1>::Zero();;
	G(0) = x0 * x6 - x1 * x7 + 1;
	G(1) = -2 * c * e * x7 + 2 * (f * x7 - x6) * b;
	G(2) = -2 * b * e * x7 + 2 * (d * x7 - x6) * c;
	G(3) = f * x8 + x2 * x7 + x9;
	G(4) = -2 * e * x8 - 2 * x4 * x7;
	G(5) = d * x8 + x3 * x7 + x9;
    return G;
}


const double minsd = 6;

double symmetricDirichletEnergyFromMetric3D(double* Eg, double* Eh, const double* pMetric, const double* invM, const double* weights, int nt)
{
    double en = 0;

    Map<const Matrix<double, Dynamic, 6> > M(pMetric, nt, 6);
    Map<const Matrix<double, Dynamic, 6> > invA(invM, nt*6, 6);

	#pragma omp parallel for reduction(+:en)
	for (int i = 0; i < nt; i++){
		double tet_weight = weights[i];
        
		double a =  M(i, 0);
		double b =  M(i, 1);
		double c =  M(i, 2);
		double d =  M(i, 3);
		double e =  M(i, 4);
		double f =  M(i, 5);

        double sd = a + d + f + (sqr(b) + sqr(c) + sqr(e) - d * f - a * d - a * f) / (f*sqr(b) + d * sqr(c) + a * sqr(e) - a * d*f - 2 * b*c*e) - minsd;
        en += tet_weight*sd;
		
		auto single_trans_invC = invA.block<6,6>(i*6, 0); // single_trans_invC is whats in our paper called T'
        if(Eg)
            Map<Matrix<double, 6, 1> >(Eg + i * 6) = tet_weight * single_trans_invC.transpose() * computeGrad(a, b, c, d, e, f);

        if(Eh)
            Map<Matrix<double, 6, 6> >(Eh + i * 6 * 6) = tet_weight * single_trans_invC.transpose() * computeHess(a, b, c, d, e, f)* single_trans_invC;
	}

    return en;
}

double expSDEnergyFromMetric3D(double* Eg, double* Eh, const double* pMetric, const double* invM, const double* weights, int nt, double param)
{
    double en = 0;

    Map<const Matrix<double, Dynamic, 6> > M(pMetric, nt, 6);
    Map<const Matrix<double, Dynamic, 6> > invA(invM, nt*6, 6);

	#pragma omp parallel for reduction(+:en)
	for (int i = 0; i < nt; i++){
		double tet_weight = weights[i];
        
		double a =  M(i, 0);
		double b =  M(i, 1);
		double c =  M(i, 2);
		double d =  M(i, 3);
		double e =  M(i, 4);
		double f =  M(i, 5);

        double sd = a + d + f + (sqr(b) + sqr(c) + sqr(e) - d * f - a * d - a * f) / (f*sqr(b) + d * sqr(c) + a * sqr(e) - a * d*f - 2 * b*c*e) - minsd;
        double esd = exp(param*sd);
        en += tet_weight*(esd-1); // shift towards 0

        if (!Eg && !Eh) continue;
		
		auto invA_i = invA.block<6,6>(i*6, 0); // invA_i is whats in our paper called T
        const auto grad = computeGrad(a, b, c, d, e, f);
        double w1 = tet_weight * param*esd;

        if (Eg) 
            Map<Matrix<double, 6, 1> >(Eg + i * 6) = w1 * invA_i.transpose() * grad;

        if (Eh) {
            Map<Matrix<double, 6, 6> >(Eh + i * 6 * 6) = w1 * invA_i.transpose() * (computeHess(a, b, c, d, e, f) + param*grad*grad.transpose())* invA_i ;
        }
	}

    return en;
}

// [E, Eg, Eh] = isometricEnergyFromMetric3D(M, invA, w)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 3) mexErrMsgTxt("[E, Eg, Eh] = isometricEnergyFromMetric3D(M, invA, wt [, param for expSD]);");

    int nt = mxGetM(prhs[2]);

    double *g = nullptr, *h = nullptr;

    if (nlhs > 1) {
        plhs[1] = mxCreateDoubleMatrix(6, nt, mxREAL);
        g = mxGetPr(plhs[1]);
    }
    if (nlhs > 2) {
        plhs[2] = mxCreateDoubleMatrix(6, 6 * nt, mxREAL);
        h = mxGetPr(plhs[2]);
    }

    double en;
    
    if (nrhs>3)
        en = expSDEnergyFromMetric3D(g, h, mxGetPr(prhs[0]), mxGetPr(prhs[1]), mxGetPr(prhs[2]), nt, mxGetScalar(prhs[3]));
    else
        en = symmetricDirichletEnergyFromMetric3D(g, h, mxGetPr(prhs[0]), mxGetPr(prhs[1]), mxGetPr(prhs[2]), nt);

    plhs[0] = mxCreateDoubleScalar(en);
}
