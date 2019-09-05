This package contains the code that implements the following paper,

Ido Aharon, Renjie Chen, Denis Zorin, and Ofir Weber.
Bounded Distortion Tetrahedral Metric Interpolation.
ACM Transactions on Graphics, 38(6) (SIGGRAPH Asia 2019)

In addition, we provide implementations for interpolation methods developed in the following papers,
* [ARAP] Alexa et al. 2000. As-rigid-as-possible Shape Interpolation.
* [FFMP] Kircher and Garland. 2008. Free-form motion processing.
* [GE] Chao et al. 2010. A simple geometric model for elastic deformations.  \
       Smith et al. 2019. Analytic Eigensystems for Isotropic Distortion Energies. (for SPD Hessian projection).
* [ABF (3D)] Paillé et al. 2015. Dihedral Angle-based Maps of Tetrahedral Meshes.

# To run the code:
1. Start MATLAB;
2. cd to the code folder;
3. run tetmetric_demo.m within MATLAB;
4. More models can be downloaded from the release page of the Github project. 

# Requirements:
- MATLAB (>2018a)
- Pardiso V6. (https://pardiso-project.org). Make sure that a valid license is installed and has not expired, and the dll needs to be placed in search path, e.g. the code folder.

# What does the package contain?
The code is implemented in MATLAB, some heavy computation is offloaded through C++ (mex). Precompiled mex binary for Win64 are provided with the package.

# Compile the mex binaries
Precompiled mex binaries are provided for 64 bit Windows. For any other OS, the mex/C++ code needs to be compiled before running the demo script, which can be done by calling buildAllMex.m, the build script, but make sure the following before calling the build script,
* The mex compiler is working. See [https://www.mathworks.com/help/matlab/matlab_external/changing-default-compiler.html].
* The Eigen [http://eigen.tuxfamily.org] library is installed and the environment variable "EIGEN_DIR" is set to the installation path.
* abfflatten.mexw64 provides mex interface to the volumetric ABF algorithm of Paillé et al. [2015]. To compile the mex, the volumnABF code from is needed, which can be obtained from the release page.

# ACKNOWLEDGMENTS
We thank Gilles-Philippe Paillé for providing the code for volumetric ABF.
