#include <mex.h>
#include <string>
#include "matlabmatrix.h"
#include "sparsematrix.h"
#include "pardisoinfo.h"


// -----------------------------------------------------------------
// Return true if the MATLAB array is a scalar in double precision.
inline bool mxIsDoubleScalar (const mxArray* ptr) {
  return mxIsDouble(ptr) && mxGetNumberOfElements(ptr);
}
// Function definitions.
// -----------------------------------------------------------------
template <class Type> void copymemory (const Type* source, Type* dest, int n) {
  memcpy(dest,source,sizeof(Type) * n);
}


void mexFunction (int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) 
{
    if (!mxIsChar(prhs[0])) mexErrMsgTxt("Invalid input: first argument should be method name in char array.");
    const std::u16string_view mode(mxGetChars(prhs[0]), mxGetNumberOfElements(prhs[0]));

    if(mode==u"init"){
        int            mtype;   // The matrix type.
        int            solver;  // Which solver to use.
        const mxArray* ptr;     // Pointer to an input argument.

        // Check to see if we have the correct number of input and output arguments.
        if (nrhs != 3)
            mexErrMsgTxt("Incorrect number of input arguments");
        if (nlhs != 1)
            mexErrMsgTxt("Incorrect number of output arguments");

        // The first input specifies the matrix type.
        ptr = prhs[1];
        if (mxIsDoubleScalar(ptr))
            mtype = (int) mxGetScalar(ptr);
        else
            mexErrMsgTxt("The first input must be a number specifying the matrix type (see the PARDISO manual for more details).");

        // The second input specifies which solver to use.
        ptr = prhs[2];
        if (mxIsDoubleScalar(ptr))
            solver = (int) mxGetScalar(ptr);
        else
            mexErrMsgTxt("The second input must be a number specifying which solver to use (see the PARDISO manual for more details).");

        // Create the single output containing PARDISO's internal data structures.
        plhs[0] = PardisoInfo(mtype,solver);
    }
    else if(mode==u"solve"){
        const mxArray* ptr;

        // Check to see if we have the correct number of input and output arguments.
        if (nrhs != 5)
            mexErrMsgTxt("Incorrect number of input arguments");
        if (nlhs != 2)
            mexErrMsgTxt("Incorrect number of output arguments");

        // Get the third input, which contains PARDISO's internal data structures.
        ptr = prhs[3];
        if (!PardisoInfo::isValid(ptr))
            mexErrMsgTxt("The third input must a STRUCT initialized with PARDISOINIT");
        PardisoInfo info(ptr);

        // Convert the first input to PARDISO's sparse matrix format.
        ptr = prhs[1];
        if (!SparseMatrix::isValid(ptr))
            mexErrMsgTxt("The first input must be a sparse, square matrix");
        SparseMatrix A(ptr,useComplexNumbers(info),useConjTranspose(info));

        // Process the second input, the matrix of right-hand sides B. Each column of B corresponds to a single right-hand side vector.
        ptr = prhs[2];
        if (!MatlabMatrix::isValid(ptr))
            mexErrMsgTxt("The second input must be a matrix in DOUBLE precision");
        MatlabMatrix B(ptr,useComplexNumbers(info));

        // Report an error if we are using the iterative solver, and there is more than one right-hand side 
        // (i.e. the width of the matrix B is greater than 1).
        if (useIterativeSolver(info) && width(B) > 1)
            mexErrMsgTxt("The iterative solver can only compute the solution for a single right-hand side");

        // Get the fourth input, the level of verbosity.
        ptr = prhs[4];
        if (!mxIsLogicalScalar(ptr))
            mexErrMsgTxt("The fourth input must be either TRUE or FALSE");
        bool verbose = mxIsLogicalScalarTrue(ptr);

        // Ask PARDISO to solve the system(s) of equations.
        MatlabMatrix X(height(B),width(B),useComplexNumbers(info));
        info.solve(A,B,X,verbose);

        // The first ouput is the matrix of solutions X, and the second output is the updated PARDISO internal data structures.
        plhs[0] = X;
        plhs[1] = info;
    }
    else if(mode==u"factor"){
        const mxArray* ptr;

        // Check to see if we have the correct number of input and output arguments.
        if (nrhs != 4)
            mexErrMsgTxt("Incorrect number of input arguments");
        if (nlhs != 1)
            mexErrMsgTxt("Incorrect number of output arguments");

        // Get the second input, which contains PARDISO's internal data structures.
        ptr = prhs[2];
        if (!PardisoInfo::isValid(ptr))
            mexErrMsgTxt("The second input must a STRUCT initialized with PARDISOINIT");
        PardisoInfo info(ptr);

        // Convert the first input, a matrix from MATLAB's sparse matrix format, to PARDISO's sparse matrix format.
        ptr = prhs[1];
        if (!SparseMatrix::isValid(ptr))
            mexErrMsgTxt("The first input must be a sparse, square matrix");
        SparseMatrix A(ptr,useComplexNumbers(info),useConjTranspose(info));

        // Get the third input, the level of verbosity.
        ptr = prhs[3];
        if (!mxIsLogicalScalar(ptr))
            mexErrMsgTxt("The third input must be either TRUE or FALSE");
        bool verbose = mxIsLogicalScalarTrue(ptr);

        // Ask PARDISO to factorize the matrix.
        info.factor(A,verbose);

        // Return the modified PARDISO internal data structures to the user.
        plhs[0] = info;
    }
    else if(mode==u"reorder"){
        // Check to see if we have the correct number of input and output arguments.
        if (nrhs < 4 || nrhs > 5)
            mexErrMsgTxt("Incorrect number of input arguments");
        if (nlhs != 1)
            mexErrMsgTxt("Incorrect number of output arguments");

        // Get the second input, which contains PARDISO's internal data structures.
        const mxArray* ptr = prhs[2];
        if (!PardisoInfo::isValid(ptr))
            mexErrMsgTxt("The second input must a STRUCT initialized with PARDISOINIT");
        PardisoInfo info(ptr);

        // Convert the first input to PARDISO's sparse matrix format.
        ptr = prhs[1];
        if (!SparseMatrix::isValid(ptr))
            mexErrMsgTxt("The first input must be a sparse, square matrix");
        SparseMatrix A(ptr,useComplexNumbers(info),useConjTranspose(info));

        // Get the size of the matrix.
        int n = size(A);

        // Get the third input, the level of verbosity.
        ptr = prhs[3];
        if (!mxIsLogicalScalar(ptr))
            mexErrMsgTxt("The third input must be either TRUE or FALSE");
        bool verbose = mxIsLogicalScalarTrue(ptr);

        // Get the fourth (optional) input, a user-supplied reordering of the
        // rows and columns of the system.
        int* perm = 0;
        if (nrhs == 5) {
            ptr = prhs[4];
            if (!mxIsEmpty(ptr)) {
                if (!mxIsDouble(ptr) || mxGetNumberOfElements(ptr) != n)
                    mexErrMsgTxt("The permutation must be a double-precision array of length equal to the size of the sparse matrix");

                // Copy the elements of the array.
                double* p = mxGetPr(ptr);
                perm = new int[n];
                for (int i = 0; i < n; i++)
                    perm[i] = (int) p[i];
            }
        }

        // Ask PARDISO to analyze the matrix A.
        info.reorder(A,perm,verbose);

        // Return the modified PARDISO internal data structures to the user.
        plhs[0] = info;

        // Free the dynamically allocated memory.
        delete[] perm;
    }
    else if(mode==u"free"){
        // Check to see if we have the correct number of input and output arguments.
        if (nrhs != 2)
            mexErrMsgTxt("Incorrect number of input arguments");
        if (nlhs != 0)
            mexErrMsgTxt("Incorrect number of output arguments");

        // Get the input argument containing PARDISO's internal data structures.
        const mxArray* ptr = prhs[1];
        if (!PardisoInfo::isValid(ptr))
            mexErrMsgTxt("The input must a STRUCT initialized with PARDISOINIT");
        PardisoInfo info(ptr);

        // Ask PARDISO release memory associated with all internal structures.
        info.free();
    }
}

