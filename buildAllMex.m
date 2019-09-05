%% make sure Eigen is installed. EIGEN_DIR environment variable should point to installation dir
mexsrcDirName = 'mexsrc';

outdir = fileparts( mfilename('fullpath') ); % output mex file to the same folder as this matlab function
eigen_path = getenv('EIGEN_DIR');
Ieigen = ['-I' eigen_path];
Isvd3 = ['-I' mexsrcDirName '\svd3'];

mex('COMPFLAGS="$COMPFLAGS /openmp"', '-R2017b', [mexsrcDirName '\at_sparse.cpp'], '-outdir', outdir)
mex('-R2018a', [mexsrcDirName '\replaceNonzeros.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /std:c++17 /openmp"', '-R2018a', [mexsrcDirName '\myaccumarray.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /openmp"', [mexsrcDirName '\ij2nzIdxs.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /std:c++17 /openmp"', '-R2018a', Ieigen, [mexsrcDirName '\blockblas.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /std:c++17 /openmp"', Ieigen, Isvd3, [mexsrcDirName '\isometricEnergyFromJ3Dc.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /openmp"', '-R2018a', Ieigen, [mexsrcDirName '\meshJacobians.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /openmp"', '-R2018a', [mexsrcDirName '\spAtA_nonzeros.cpp'], '-outdir', outdir);

%% for SQP_interp
mex(Ieigen, [mexsrcDirName '\tetMeshReconstructionGreedy.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /openmp"', Ieigen, [mexsrcDirName '\isometricEnergyFromMetric3Dc.cpp'], '-outdir', outdir)
mex('COMPFLAGS="$COMPFLAGS /openmp"', Ieigen, [mexsrcDirName '\dihedralAnglesJacobian.cpp'], '-outdir', outdir)

%% build pardiso_imp
% mex 'COMPFLAGS=$COMPFLAGS /std:c++17' pardiso_imp.cpp matlabmatrix.cpp pardisoinfo.cpp sparsematrix.cpp -lpardiso
