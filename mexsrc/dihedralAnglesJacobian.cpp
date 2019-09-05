#include "mex.h"
#include <math.h>   

template<class real>
double sqr(real x, int) { return x*x; };

template<class real>
double cubic(real x, int) { return x*x*x; };

/* The computational routine */
void dihedralAnglesDerivatives(const double *els, int numTetras, double* value_arr) 
{
	#pragma omp parallel for
	for (int i = 0; i < numTetras; i++)
	{
        // a,b,...,f = els
		// '-1' due to C indexing
        const double *cur_els = els + i * 6;
        double a = cur_els[0]; 
        double b = cur_els[1];
        double c = cur_els[2];
        double d = cur_els[3];
        double e = cur_els[4];
        double f = cur_els[5];
      
        //derivatives of dihedral angle around edge a (AO)
		value_arr[i * 36] = -((-2 * d - 2 * a + b + c + e + f) / sqrt((4 * a*c - sqr(a + c - e, 2))*(4 * f*b - sqr(f + b - a, 2))) - 0.5*(-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f))*((2 * c - 2 * a + 2 * e)*(4 * f*b - sqr((f + b - a), 2)) + (4 * a*c - sqr((a + c - e), 2))*(2 * f + 2 * b - 2 * a)) / sqrt(cubic(((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))) / sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))));
		value_arr[i * 36 + 1] = (2 * (a*a + (-2 * c - 2 * e)*a + sqr((c - e), 2)))*((f + d - e)*a - f*f + (b - 2 * c + d + e)*f - b*(d - e))*a / (sqrt(cubic(((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 2] = (2 * ((d + e - f)*a - e*e + (-2 * b + c + d + f)*e - c*(d - f)))*a*(a*a + (-2 * f - 2 * b)*a + sqr((b - f), 2)) / (sqrt(cubic(((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 3]  = 2 * a / (sqrt((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))*sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 4] = -(2 * ((b - c - d)*a + c*c + (-b - d - e + 2 * f)*c - e*(b - d)))*a*(a*a + (-2 * f - 2 * b)*a + sqr((b - f), 2)) / (sqrt(cubic(((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 5] = (2 * ((b + d - c)*a - b*b + (c + d - 2 * e + f)*b + f*(c - d)))*(a*a + (-2 * c - 2 * e)*a + sqr((c - e), 2))*a / (sqrt(cubic(((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-a*a + (b + c - 2 * d + e + f)*a - (c - e)*(b - f)), 2) / ((4 * a*c - sqr((a + c - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		
		//derivatives of dihedral angle around edge b (BO)
		value_arr[i * 36 + 6] = 2 * b*(b*b + (-2 * c - 2 * d)*b + sqr((c - d), 2))*((-d + e + f)*b - f*f + (a - 2 * c + d + e)*f + a*(d - e)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 7] = -((-2 * b + a + c + d - 2 * e + f) / sqrt((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))) - 0.5*(-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f))*((2 * d - 2 * b + 2 * c)*(4 * f*b - sqr((f + b - a), 2)) + (4 * b*d - sqr((b + d - c), 2))*(2 * f + 2 * a - 2 * b)) / sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))) / sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))));
		value_arr[i * 36 + 8] = -4 * b*((-0.5*d - 0.5*e + 0.5*f)*b + d*a + (-0.5*c - 0.5*d)*f - (0.5*(d - e))*(c - d))*(b*b + (-2 * a - 2 * f)*b + sqr((a - f), 2)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 9] = -2 * b*((a - c - e)*b + c*c + (-a - d - e + 2 * f)*c - d*(a - e))*(b*b + (-2 * a - 2 * f)*b + sqr((a - f), 2)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 10] = 2 * b / (sqrt((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))*sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 11] = -2 * b*((-a + c - e)*b + a*a + (-c + 2 * d - e - f)*a - f*(c - e))*(b*b + (-2 * c - 2 * d)*b + sqr((c - d), 2)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * f*b - sqr((f + b - a), 2)))));

		//derivatives of dihedral angle around edge c (CO)  
		value_arr[i * 36 + 12] = (2 * (c*c + (-2 * b - 2 * d)*c + sqr((b - d), 2)))*c*((-d + e + f)*c - 2 * b*e + (a + e)*d + (e - f)*(a - e)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))), 3))*sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))));
		value_arr[i * 36 + 13] = -4 * c*(c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*((-0.5*d + 0.5*e - 0.5*f)*c + d*a + (-0.5*b - 0.5*d)*e - (0.5*(d - f))*(b - d)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))), 3))*sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))));
		value_arr[i * 36 + 14] = -((-2 * c + a + b + d + e - 2 * f) / sqrt((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))) - 0.5*(-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e))*((2 * b + 2 * d - 2 * c)*(4 * a*c - sqr((a + c - e), 2)) + (4 * b*d - sqr((b + d - c), 2))*(2 * a - 2 * c + 2 * e)) / sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))), 3))) / sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))));
		value_arr[i * 36 + 15] = (2 * (-b*b + (a + c + d - 2 * e + f)*b - (c - d)*(a - f)))*c*(c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2)) / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))), 3))*sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))));
		value_arr[i * 36 + 16] = -(2 * ((-a + b - f)*c + a*a + (-b + 2 * d - e - f)*a - e*(b - f)))*(c*c + (-2 * b - 2 * d)*c + sqr((b - d), 2))*c / (sqrt(cubic(((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2))), 3))*sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))));
		value_arr[i * 36 + 17] = 2 * c / (sqrt((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))*sqrt(1 - sqr((-c*c + (a + b + d + e - 2 * f)*c - (b - d)*(a - e)), 2) / ((4 * b*d - sqr((b + d - c), 2))*(4 * a*c - sqr((a + c - e), 2)))));

		//derivatives of dihedral angle around edge d (BC)
		value_arr[i * 36 + 18] = 2 * d / (sqrt(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2)))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		value_arr[i * 36 + 19] = -(2 * ((-a - c + e)*d + c*c + (-a - b - e + 2 * f)*c + b*(a - e)))*(d*d + (-2 * e - 2 * f)*d + sqr((e - f), 2))*d / (sqrt((cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))), 3))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		value_arr[i * 36 + 20] = (2 * ((a + b - f)*d - b*b + (a + c - 2 * e + f)*b - c*(a - f)))*(d*d + (-2 * e - 2 * f)*d + sqr((e - f), 2))*d / (sqrt((cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))), 3))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		value_arr[i * 36 + 21] = (2 * d + 2 * a - b - c - e - f) / (sqrt(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2)))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2)))))) + 0.5*(-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c))*((2 * f - 2 * d + 2 * e)*(4 * b*d - sqr((b + d - c), 2)) + (4 * f*d - sqr((f + d - e), 2))*(2 * b - 2 * d + 2 * c)) / (sqrt((cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))), 3))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		value_arr[i * 36 + 22] = (2 * (d*d + (-2 * b - 2 * c)*d + sqr((b - c), 2)))*((a - b + f)*d - f*f + (a + b - 2 * c + e)*f - e*(a - b))*d / (sqrt((cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))), 3))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		value_arr[i * 36 + 23] = (2 * (d*d + (-2 * b - 2 * c)*d + sqr((b - c), 2)))*((a - c + e)*d - e*e + (a - 2 * b + c + f)*e - f*(a - c))*d / (sqrt((cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))), 3))*(1 - sqr((-d*d + (-2 * a + b + c + e + f)*d + (e - f)*(b - c)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * b*d - sqr((b + d - c), 2))))));
		
		//derivatives of dihedral angle around edge e (AC)
		value_arr[i * 36 + 24] = -(2 * ((-b - c + d)*e + c*c + (-a - b - d + 2 * f)*c + a*(b - d)))*(e*e + (-2 * d - 2 * f)*e + sqr((d - f), 2))*e / (sqrt(cubic(((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))), 3))*sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))));
		value_arr[i * 36 + 25] = 2 * e / (sqrt((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))*sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))));
		value_arr[i * 36 + 26] = -(2 * ((-a - b + f)*e + a*a + (-b - c + 2 * d - f)*a + c*(b - f)))*(e*e + (-2 * d - 2 * f)*e + sqr((d - f), 2))*e / (sqrt(cubic(((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))), 3))*sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))));
		value_arr[i * 36 + 27] = (2 * (e*e + (-2 * a - 2 * c)*e + sqr((a - c), 2)))*((-a + b + f)*e - f*f + (a + b - 2 * c + d)*f + d*(a - b))*e / (sqrt(cubic(((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))), 3))*sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))));
		value_arr[i * 36 + 28] = -((-2 * e + a - 2 * b + c + d + f) / sqrt((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))) - 0.5*(-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c))*((-2 * a - 2 * c + 2 * e)*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)) + (c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(-2 * d - 2 * f + 2 * e)) / sqrt(cubic(((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))), 3))) / sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))));
		value_arr[i * 36 + 29] = -(4 * ((-0.5*b + 0.5*c - 0.5*d)*e + d*a + (-0.5*d - 0.5*f)*c - (0.5*(d - f))*(b - d)))*(e*e + (-2 * a - 2 * c)*e + sqr((a - c), 2))*e / (sqrt(cubic(((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2))), 3))*sqrt(1 - sqr((-e*e + (a - 2 * b + c + d + f)*e + (d - f)*(a - c)), 2) / ((c*c + (-2 * a - 2 * e)*c + sqr((a - e), 2))*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)))));
		
		//derivatives of dihedral angle around edge f (AB)
		value_arr[i * 36 + 30] = (2 * ((b + c - d)*f - b*b + (a + c + d - 2 * e)*b - a*(c - d)))*f*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)) / (sqrt(cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 31] = -(2 * ((-a - c + e)*f + a*a + (-b - c + 2 * d - e)*a + b*(c - e)))*f*(f*f + (-2 * d - 2 * e)*f + sqr((d - e), 2)) / (sqrt(cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 32] = 2 * f / (sqrt((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))*sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 33] = 2 * f*((-a + c + e)*f - e*e + (a - 2 * b + c + d)*e + d*(a - c))*(f*f + (-2 * a - 2 * b)*f + sqr((a - b), 2)) / (sqrt(cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 34] = -(4 * ((0.5*b - 0.5*c - 0.5*d)*f + d*a + (-0.5*d - 0.5*e)*b - (0.5*(d - e))*(c - d)))*f*(f*f + (-2 * a - 2 * b)*f + sqr((a - b), 2)) / (sqrt(cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))*sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2)))));
		value_arr[i * 36 + 35] = -((-2 * f + a + b - 2 * c + d + e) / sqrt((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))) - 0.5*(-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b))*((2 * d - 2 * f + 2 * e)*(4 * f*b - sqr((f + b - a), 2)) + (4 * f*d - sqr((f + d - e), 2))*(2 * b - 2 * f + 2 * a)) / sqrt(cubic(((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))), 3))) / sqrt(1 - sqr((-f*f + (a + b - 2 * c + d + e)*f + (d - e)*(a - b)), 2) / ((4 * f*d - sqr((f + d - e), 2))*(4 * f*b - sqr((f + b - a), 2))));
	}

}

/* The gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/* explanation */
    /* input: update_els - all els (those we develop taylor series around)
     dihedral_input_indices - matrix which create in C++ during preprocessing
     outside the function, approximation curvature will be:
     approximate curvature = dihedral_angles_around_all_edges + J*update_els */
    
    /* important note */
    /* J includes derivatives around all edegs. hence before using J, 
    we should extract from J, only the rows which belong to interior edges*/
    
    /* call example */
	//value_arr = dihedralAnglesDerivatives(els_per_tet_transposed); 
    
    int numTetras = (int)mxGetN(prhs[0]);
    
	plhs[0] = mxCreateDoubleMatrix(6, 6 * numTetras, mxREAL);

	/* call the computational routine */
	dihedralAnglesDerivatives(mxGetPr(prhs[0]), numTetras, mxGetPr(plhs[0]));
}

