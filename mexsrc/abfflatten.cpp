#include <mex.h>
#include <locale>
#include <codecvt> 
#include "volumeABF.h"

// see https://stackoverflow.com/questions/32055357/visual-studio-c-2015-stdcodecvt-with-char16-t-or-char32-t
#if _MSC_VER >= 1900
std::string utf16_to_utf8(std::u16string utf16_string)
{
    std::wstring_convert<std::codecvt_utf8_utf16<int16_t>, int16_t> convert;
    auto p = reinterpret_cast<const int16_t *>(utf16_string.data());
    return convert.to_bytes(p, p + utf16_string.size());
}
#else
std::string utf16_to_utf8(std::u16string utf16_string)
{
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    return convert.to_bytes(utf16_string);
}
#endif

inline std::string ws2s(const std::u16string& wstr)
{
    return { wstr.cbegin(), wstr.cend() };
}

template<typename R = double>
R getFieldValueWithDefault(const mxArray* mat, const char* name, R defaultvalue)
{
    const mxArray *f = mat?mxGetField(mat, 0, name):nullptr;

    if (!f) return defaultvalue;

    if constexpr (std::is_same_v<R, std::u16string>) {
        if (!mxIsChar(f)) mexErrMsgTxt( (std::string("expecting a char array for field: ") +name).data() );
        return R(mxGetChars(f), mxGetNumberOfElements(f));
    }
    else if constexpr (std::is_same_v < R, bool>){
        if (!mxIsLogicalScalar(f)) mexErrMsgTxt( (std::string("expecting a boolean scalar for field: ") +name).data() );
        return mxIsLogicalScalarTrue(f);
    }
    else {
        if (!mxIsScalar(f)) mexErrMsgTxt((std::string("expecting a scalar for field: ") + name).data());
        return R(mxGetScalar(f));
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
    if (nrhs == 0) mexErrMsgTxt("usage: z = abfflatten(diangles, x, tets, tetneighbors[, options]);");
    if (nrhs < 4) mexErrMsgTxt("not enough inputs!");

    if (!(mxIsDouble(prhs[0]) && mxIsDouble(prhs[1]) && mxIsInt32(prhs[2]) && mxIsInt32(prhs[3])))
        mexErrMsgTxt("check input, incorrect types");

    const double *diangles = mxGetDoubles(prhs[0]);
    const double *x = mxGetDoubles(prhs[1]);
    const int *tets = mxGetInt32s(prhs[2]);
    const int *tetneighbors = mxGetInt32s(prhs[3]);

    size_t nv = mxGetM(prhs[1]);
    size_t nt = mxGetM(prhs[2]);


    const mxArray *params = nrhs>4?prhs[4]:nullptr;
    bool useFFMP = getFieldValueWithDefault<bool>(params, "ffmp", false);
    std::u16string reconstructMethod = getFieldValueWithDefault<std::u16string>(params, "reconstruction", u"leastsquare");
    double min_angle = getFieldValueWithDefault(params, "min_angle", 0.1);;
    double regularization = getFieldValueWithDefault(params, "regularization", 1e-6);
    unsigned int nb_outer_iterations = getFieldValueWithDefault<unsigned>(params, "nOuterIter", 2);
    unsigned int nb_inner_iterations = getFieldValueWithDefault<unsigned>(params, "nInnerIter", 2000);
    double threshold = getFieldValueWithDefault(params, "threshold", 1e-9);

    OGF::ReconstructionMethod reconstruction_method = OGF::LeastSquares;

    if(reconstructMethod==u"leastsquare") reconstruction_method = OGF::LeastSquares;
    else if(reconstructMethod==u"eigen") reconstruction_method = OGF::Eigen;
    else if(reconstructMethod==u"greedy") reconstruction_method = OGF::Greedy;
    else 
        mexErrMsgTxt(("unsupported reconstructed method: " + ws2s(reconstructMethod)).data());

    using mapmatc = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor> >;
    using mapmat6c = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::ColMajor> >;
    using mapmat = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor> >;
    using imapmat = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::ColMajor> >;

    plhs[0] = mxDuplicateArray(prhs[1]);
    mapmat z(mxGetDoubles(plhs[0]), nv, 3);

    abfflatten(mapmat6c(diangles, nt, 6), mapmatc(x, nv, 3), imapmat(tets, nt, 4), imapmat(tetneighbors, nt, 4), z, min_angle, regularization, nb_outer_iterations, nb_inner_iterations, threshold, reconstruction_method);
}
