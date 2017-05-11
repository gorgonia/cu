
typedef  enum CUBLAS_STATUS {
    CUBLAS_STATUS_SUCCESS =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED =3,
    CUBLAS_STATUS_INVALID_VALUE =7,
    CUBLAS_STATUS_ARCH_MISMATCH =8,
    CUBLAS_STATUS_MAPPING_ERROR =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR =14,
    CUBLAS_STATUS_NOT_SUPPORTED =15,
    CUBLAS_STATUS_LICENSE_ERROR =16
} cublasStatus_t;


typedef enum CUBLAS_UPLO {
    CUBLAS_FILL_MODE_LOWER=0,
    CUBLAS_FILL_MODE_UPPER=1
} cublasFillMode_t;

typedef enum CUBLAS_DIAG {
    CUBLAS_DIAG_NON_UNIT=0,
    CUBLAS_DIAG_UNIT=1
} cublasDiagType_t;

typedef enum CUBLAS_SIDE {
    CUBLAS_SIDE_LEFT =0,
    CUBLAS_SIDE_RIGHT=1
} cublasSideMode_t;


typedef enum CUBLAS_TRANSPOSE {
    CUBLAS_OP_N=0,
    CUBLAS_OP_T=1,
    CUBLAS_OP_C=2
} cublasOperation_t;


typedef enum {
    CUBLAS_POINTER_MODE_HOST = 0,
    CUBLAS_POINTER_MODE_DEVICE = 1
} cublasPointerMode_t;

typedef enum {
    CUBLAS_ATOMICS_NOT_ALLOWED = 0,
    CUBLAS_ATOMICS_ALLOWED = 1
} cublasAtomicsMode_t;


typedef enum{
    CUBLAS_GEMM_DFALT = -1,
    CUBLAS_GEMM_ALGO0 = 0,
    CUBLAS_GEMM_ALGO1 = 1,
    CUBLAS_GEMM_ALGO2 = 2,
    CUBLAS_GEMM_ALGO3 = 3,
    CUBLAS_GEMM_ALGO4 = 4,
    CUBLAS_GEMM_ALGO5 = 5,
    CUBLAS_GEMM_ALGO6 = 6,
    CUBLAS_GEMM_ALGO7 = 7
} cublasGemmAlgo_t;

// DUMMY types so that this file can be used to generate the BLAS file
typedef int cudaDataType;
typedef int cublasDataType_t;
typedef int libraryPropertyType;
typedef int cudaStream_t;
typedef float _Complex cuComplex;
typedef double _Complex cuDoubleComplex;
typedef int __half;
typedef int cublasHandle_t;

/*

 cublasStatus_t cublasCreate(cublasHandle_t *handle);
 cublasStatus_t cublasDestroy(cublasHandle_t handle);

 cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version);
 cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value);

 cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
 cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId);

 cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t *mode);
 cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode);

 cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t *mode);
 cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle, cublasAtomicsMode_t mode);
# 184 "cublas_api.h"
cublasStatus_t cublasSetVector (int n, int elemSize, const void *x,
                                             int incX, void *devicePtr, int incY);
# 210 "cublas_api.h"
cublasStatus_t cublasGetVector (int n, int elemSize, const void *x,
                                             int incX, void *y, int incY);
# 234 "cublas_api.h"
cublasStatus_t cublasSetMatrix (int rows, int cols, int elemSize,
                                             const void *A, int lda, void *B,
                                             int ldb);
# 258 "cublas_api.h"
cublasStatus_t cublasGetMatrix (int rows, int cols, int elemSize,
                                             const void *A, int lda, void *B,
                                             int ldb);
# 278 "cublas_api.h"
cublasStatus_t cublasSetVectorAsync (int n, int elemSize,
                                                  const void *hostPtr, int incX,
                                                  void *devicePtr, int incY,
                                                  cudaStream_t stream);
# 298 "cublas_api.h"
cublasStatus_t cublasGetVectorAsync (int n, int elemSize,
                                                  const void *devicePtr, int incX,
                                                  void *hostPtr, int incY,
                                                  cudaStream_t stream);
# 320 "cublas_api.h"
cublasStatus_t cublasSetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream);
# 340 "cublas_api.h"
cublasStatus_t cublasGetMatrixAsync (int rows, int cols, int elemSize,
                                                  const void *A, int lda, void *B,
                                                  int ldb, cudaStream_t stream);


 void cublasXerbla (const char *srName, int info);

*/


/*
 cublasStatus_t cublasNrm2Ex(cublasHandle_t handle,
                                                     int n,
                                                     const void *x,
                                                     cudaDataType xType,
                                                     int incX,
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);
*/
 cublasStatus_t cublasSnrm2(cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incX,
                                                     float *result);

 cublasStatus_t cublasDnrm2(cublasHandle_t handle,
                                                     int n,
                                                     const double *x,
                                                     int incX,
                                                     double *result);

 cublasStatus_t cublasScnrm2(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      float *result);

 cublasStatus_t cublasDznrm2(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      double *result);
/*
 cublasStatus_t cublasDotEx (cublasHandle_t handle,
                                                     int n,
                                                     const void *x,
                                                     cudaDataType xType,
                                                     int incX,
                                                     const void *y,
                                                     cudaDataType yType,
                                                     int incY,
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

 cublasStatus_t cublasDotcEx (cublasHandle_t handle,
                                                     int n,
                                                     const void *x,
                                                     cudaDataType xType,
                                                     int incX,
                                                     const void *y,
                                                     cudaDataType yType,
                                                     int incY,
                                                     void *result,
                                                     cudaDataType resultType,
                                                     cudaDataType executionType);

*/
 cublasStatus_t cublasSdot(cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incX,
                                                     const float *y,
                                                     int incY,
                                                     float *result);

 cublasStatus_t cublasDdot(cublasHandle_t handle,
                                                     int n,
                                                     const double *x,

                                                     int incX,
                                                     const double *y,
                                                     int incY,
                                                     double *result);
/*
 cublasStatus_t cublasCdotu(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *result);

 cublasStatus_t cublasCdotc(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *result);

 cublasStatus_t cublasZdotu(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *result);

 cublasStatus_t cublasZdotc(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *result);

 cublasStatus_t cublasScalEx(cublasHandle_t handle,
                                                     int n,
                                                     const void *alpha,
                                                     cudaDataType alphaType,
                                                     void *x,
                                                     cudaDataType xType,
                                                     int incX,
                                                     cudaDataType executionType);

*/
 cublasStatus_t cublasSscal(cublasHandle_t handle,
                                                     int n,
                                                     const float *alpha,
                                                     float *x,
                                                     int incX);

 cublasStatus_t cublasDscal(cublasHandle_t handle,
                                                     int n,
                                                     const double *alpha,
                                                     double *x,
                                                     int incX);

 cublasStatus_t cublasCscal(cublasHandle_t handle,
                                                     int n,
                                                     const cuComplex *alpha,
                                                     cuComplex *x,
                                                     int incX);

 cublasStatus_t cublasCsscal(cublasHandle_t handle,
                                                      int n,
                                                      const float *alpha,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZscal(cublasHandle_t handle,
                                                     int n,
                                                     const cuDoubleComplex *alpha,
                                                     cuDoubleComplex *x,
                                                     int incX);

 cublasStatus_t cublasZdscal(cublasHandle_t handle,
                                                      int n,
                                                      const double *alpha,
                                                      cuDoubleComplex *x,
                                                      int incX);


/*
 cublasStatus_t cublasAxpyEx (cublasHandle_t handle,
                                                      int n,
                                                      const void *alpha,
                                                      cudaDataType alphaType,
                                                      const void *x,
                                                      cudaDataType xType,
                                                      int incX,
                                                      void *y,
                                                      cudaDataType yType,
                                                      int incY,
                                                      cudaDataType executiontype);
*/

 cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                                                      int n,
                                                      const float *alpha,
                                                      const float *x,
                                                      int incX,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                                                      int n,
                                                      const double *alpha,
                                                      const double *x,
                                                      int incX,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCaxpy(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZaxpy(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      cuDoubleComplex *y,
                                                      int incY);

 cublasStatus_t cublasScopy(cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incX,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDcopy(cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incX,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCcopy(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZcopy(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      cuDoubleComplex *y,
                                                      int incY);

 cublasStatus_t cublasSswap(cublasHandle_t handle,
                                                      int n,
                                                      float *x,
                                                      int incX,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDswap(cublasHandle_t handle,
                                                      int n,
                                                      double *x,
                                                      int incX,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCswap(cublasHandle_t handle,
                                                      int n,
                                                      cuComplex *x,
                                                      int incX,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZswap(cublasHandle_t handle,
                                                      int n,
                                                      cuDoubleComplex *x,
                                                      int incX,
                                                      cuDoubleComplex *y,
                                                      int incY);

 cublasStatus_t cublasIsamax(cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIdamax(cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIcamax(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIzamax(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIsamin(cublasHandle_t handle,
                                                      int n,
                                                      const float *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIdamin(cublasHandle_t handle,
                                                      int n,
                                                      const double *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIcamin(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasIzamin(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      int *result);

 cublasStatus_t cublasSasum(cublasHandle_t handle,
                                                     int n,
                                                     const float *x,
                                                     int incX,
                                                     float *result);

 cublasStatus_t cublasDasum(cublasHandle_t handle,
                                                     int n,
                                                     const double *x,
                                                     int incX,
                                                     double *result);

 cublasStatus_t cublasScasum(cublasHandle_t handle,
                                                      int n,
                                                      const cuComplex *x,
                                                      int incX,
                                                      float *result);

 cublasStatus_t cublasDzasum(cublasHandle_t handle,
                                                      int n,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      double *result);

 cublasStatus_t cublasSrot(cublasHandle_t handle,
                                                     int n,
                                                     float *x,
                                                     int incX,
                                                     float *y,
                                                     int incY,
                                                     const float *cScalar,
                                                     const float *sScalar);

 cublasStatus_t cublasDrot(cublasHandle_t handle,
                                                     int n,
                                                     double *x,
                                                     int incX,
                                                     double *y,
                                                     int incY,
                                                     const double *cScalar,
                                                     const double *sScalar);

 cublasStatus_t cublasCrot(cublasHandle_t handle,
                                                     int n,
                                                     cuComplex *x,
                                                     int incX,
                                                     cuComplex *y,
                                                     int incY,
                                                     const float *cScalar,
                                                     const cuComplex *sScalar);

 cublasStatus_t cublasCsrot(cublasHandle_t handle,
                                                     int n,
                                                     cuComplex *x,
                                                     int incX,
                                                     cuComplex *y,
                                                     int incY,
                                                     const float *cScalar,
                                                     const float *sScalar);

 cublasStatus_t cublasZrot(cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex *x,
                                                     int incX,
                                                     cuDoubleComplex *y,
                                                     int incY,
                                                     const double *cScalar,
                                                     const cuDoubleComplex *sScalar);

 cublasStatus_t cublasZdrot(cublasHandle_t handle,
                                                     int n,
                                                     cuDoubleComplex *x,
                                                     int incX,
                                                     cuDoubleComplex *y,
                                                     int incY,
                                                     const double *cScalar,
                                                     const double *sScalar);

 cublasStatus_t cublasSrotg(cublasHandle_t handle,
                                                     float *a,
                                                     float *b,
                                                     float *c,
                                                     float *s);

 cublasStatus_t cublasDrotg(cublasHandle_t handle,
                                                     double *a,
                                                     double *b,
                                                     double *c,
                                                     double *s);

 cublasStatus_t cublasCrotg(cublasHandle_t handle,
                                                     cuComplex *a,
                                                     cuComplex *b,
                                                     float *cScalar,
                                                     cuComplex *sScalar);

 cublasStatus_t cublasZrotg(cublasHandle_t handle,
                                                     cuDoubleComplex *a,
                                                     cuDoubleComplex *b,
                                                     double *cScalar,
                                                     cuDoubleComplex *sScalar);

 cublasStatus_t cublasSrotm(cublasHandle_t handle,
                                                     int n,
                                                     float *x,
                                                     int incX,
                                                     float *y,
                                                     int incY,
                                                     const float* param);

 cublasStatus_t cublasDrotm(cublasHandle_t handle,
                                                     int n,
                                                     double *x,
                                                     int incX,
                                                     double *y,
                                                     int incY,
                                                     const double* param);

 cublasStatus_t cublasSrotmg(cublasHandle_t handle,
                                                      float *d1,
                                                      float *d2,
                                                      float *x1,
                                                      const float *y1,
                                                      float *param);

 cublasStatus_t cublasDrotmg(cublasHandle_t handle,
                                                      double *d1,
                                                      double *d2,
                                                      double *x1,
                                                      const double *y1,
                                                      double *param);




 cublasStatus_t cublasSgemv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incX,
                                                      const float *beta,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDgemv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incX,
                                                      const double *beta,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCgemv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZgemv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);

 cublasStatus_t cublasSgbmv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incX,
                                                      const float *beta,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDgbmv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incX,
                                                      const double *beta,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCgbmv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZgbmv(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      int m,
                                                      int n,
                                                      int kl,
                                                      int ku,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);


 cublasStatus_t cublasStrmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtrmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtrmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtrmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incX);


 cublasStatus_t cublasStbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incX);


 cublasStatus_t cublasStpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *AP,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *AP,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *AP,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *AP,
                                                      cuDoubleComplex *x,
                                                      int incX);


 cublasStatus_t cublasStrsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtrsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtrsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtrsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incX);


 cublasStatus_t cublasStpsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const float *AP,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtpsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const double *AP,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtpsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuComplex *AP,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtpsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      const cuDoubleComplex *AP,
                                                      cuDoubleComplex *x,
                                                      int incX);

 cublasStatus_t cublasStbsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const float *A,
                                                      int lda,
                                                      float *x,
                                                      int incX);

 cublasStatus_t cublasDtbsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const double *A,
                                                      int lda,
                                                      double *x,
                                                      int incX);

 cublasStatus_t cublasCtbsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuComplex *A,
                                                      int lda,
                                                      cuComplex *x,
                                                      int incX);

 cublasStatus_t cublasZtbsv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      cuDoubleComplex *x,
                                                      int incX);


 cublasStatus_t cublasSsymv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incX,
                                                      const float *beta,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDsymv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incX,
                                                      const double *beta,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasCsymv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZsymv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);

 cublasStatus_t cublasChemv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZhemv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);


 cublasStatus_t cublasSsbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *x,
                                                      int incX,
                                                      const float *beta,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDsbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *x,
                                                      int incX,
                                                      const double *beta,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasChbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZhbmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);


 cublasStatus_t cublasSspmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,
                                                      const float *AP,
                                                      const float *x,
                                                      int incX,
                                                      const float *beta,
                                                      float *y,
                                                      int incY);

 cublasStatus_t cublasDspmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,
                                                      const double *AP,
                                                      const double *x,
                                                      int incX,
                                                      const double *beta,
                                                      double *y,
                                                      int incY);

 cublasStatus_t cublasChpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *AP,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *beta,
                                                      cuComplex *y,
                                                      int incY);

 cublasStatus_t cublasZhpmv(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *AP,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *y,
                                                      int incY);


 cublasStatus_t cublasSger(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const float *alpha,
                                                     const float *x,
                                                     int incX,
                                                     const float *y,
                                                     int incY,
                                                     float *A,
                                                     int lda);

 cublasStatus_t cublasDger(cublasHandle_t handle,
                                                     int m,
                                                     int n,
                                                     const double *alpha,
                                                     const double *x,
                                                     int incX,
                                                     const double *y,
                                                     int incY,
                                                     double *A,
                                                     int lda);

 cublasStatus_t cublasCgeru(cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *A,
                                                      int lda);

 cublasStatus_t cublasCgerc(cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *A,
                                                      int lda);

 cublasStatus_t cublasZgeru(cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *A,
                                                      int lda);

 cublasStatus_t cublasZgerc(cublasHandle_t handle,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *A,
                                                      int lda);


 cublasStatus_t cublasSsyr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha,
                                                     const float *x,
                                                     int incX,
                                                     float *A,
                                                     int lda);

 cublasStatus_t cublasDsyr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha,
                                                     const double *x,
                                                     int incX,
                                                     double *A,
                                                     int lda);

 cublasStatus_t cublasCsyr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *alpha,
                                                     const cuComplex *x,
                                                     int incX,
                                                     cuComplex *A,
                                                     int lda);

 cublasStatus_t cublasZsyr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *alpha,
                                                     const cuDoubleComplex *x,
                                                     int incX,
                                                     cuDoubleComplex *A,
                                                     int lda);

 cublasStatus_t cublasCher(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha,
                                                     const cuComplex *x,
                                                     int incX,
                                                     cuComplex *A,
                                                     int lda);

 cublasStatus_t cublasZher(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha,
                                                     const cuDoubleComplex *x,
                                                     int incX,
                                                     cuDoubleComplex *A,
                                                     int lda);


 cublasStatus_t cublasSspr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha,
                                                     const float *x,
                                                     int incX,
                                                     float *AP);

 cublasStatus_t cublasDspr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha,
                                                     const double *x,
                                                     int incX,
                                                     double *AP);

 cublasStatus_t cublasChpr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *alpha,
                                                     const cuComplex *x,
                                                     int incX,
                                                     cuComplex *AP);

 cublasStatus_t cublasZhpr(cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *alpha,
                                                     const cuDoubleComplex *x,
                                                     int incX,
                                                     cuDoubleComplex *AP);


 cublasStatus_t cublasSsyr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,
                                                      const float *x,
                                                      int incX,
                                                      const float *y,
                                                      int incY,
                                                      float *A,
                                                      int lda);

 cublasStatus_t cublasDsyr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,
                                                      const double *x,
                                                      int incX,
                                                      const double *y,
                                                      int incY,
                                                      double *A,
                                                      int lda);

 cublasStatus_t cublasCsyr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *A,
                                                      int lda);

 cublasStatus_t cublasZsyr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *A,
                                                      int lda);


 cublasStatus_t cublasCher2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo, int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *A,
                                                      int lda);

 cublasStatus_t cublasZher2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *A,
                                                      int lda);


 cublasStatus_t cublasSspr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const float *alpha,
                                                      const float *x,
                                                      int incX,
                                                      const float *y,
                                                      int incY,
                                                      float *AP);

 cublasStatus_t cublasDspr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const double *alpha,
                                                      const double *x,
                                                      int incX,
                                                      const double *y,
                                                      int incY,
                                                      double *AP);


 cublasStatus_t cublasChpr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *x,
                                                      int incX,
                                                      const cuComplex *y,
                                                      int incY,
                                                      cuComplex *AP);

 cublasStatus_t cublasZhpr2(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *x,
                                                      int incX,
                                                      const cuDoubleComplex *y,
                                                      int incY,
                                                      cuDoubleComplex *AP);




 cublasStatus_t cublasSgemm(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta,
                                                      float *C,
                                                      int ldc);

 cublasStatus_t cublasDgemm(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta,
                                                      double *C,
                                                      int ldc);

 cublasStatus_t cublasCgemm(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta,
                                                      cuComplex *C,
                                                      int ldc);

 cublasStatus_t cublasCgemm3m (cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta,
                                                      cuComplex *C,
                                                      int ldc);
 /*
 cublasStatus_t cublasCgemm3mEx (cublasHandle_t handle,
                                                     cublasOperation_t TransA, cublasOperation_t TransB,
                                                     int m, int n, int k,
                                                     const cuComplex *alpha,
                                                     const void *A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     const void *B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     const cuComplex *beta,
                                                     void *C,
                                                     cudaDataType Ctype,
                                                     int ldc);
*/

 cublasStatus_t cublasZgemm(cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);

 cublasStatus_t cublasZgemm3m (cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);

/*
 cublasStatus_t cublasHgemm (cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const __half *alpha,
                                                      const __half *A,
                                                      int lda,
                                                      const __half *B,
                                                      int ldb,
                                                      const __half *beta,
                                                      __half *C,
                                                      int ldc);
 cublasStatus_t cublasSgemmEx (cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const float *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc);


 cublasStatus_t cublasGemmEx (cublasHandle_t handle,
                                                      cublasOperation_t TransA,
                                                      cublasOperation_t TransB,
                                                      int m,
                                                      int n,
                                                      int k,
                                                      const void *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const void *B,
                                                      cudaDataType Btype,
                                                      int ldb,
                                                      const void *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc,
                                                      cudaDataType computeType,
                                                      cublasGemmAlgo_t algo);


 cublasStatus_t cublasCgemmEx (cublasHandle_t handle,
                                                     cublasOperation_t TransA, cublasOperation_t TransB,
                                                     int m, int n, int k,
                                                     const cuComplex *alpha,
                                                     const void *A,
                                                     cudaDataType Atype,
                                                     int lda,
                                                     const void *B,
                                                     cudaDataType Btype,
                                                     int ldb,
                                                     const cuComplex *beta,
                                                     void *C,
                                                     cudaDataType Ctype,
                                                     int ldc);
*/

/*
 cublasStatus_t cublasUint8gemmBias (cublasHandle_t handle,
                                                           cublasOperation_t TransA, cublasOperation_t TransB, cublasOperation_t TransAc,
                                                           int m, int n, int k,
                                                           const unsigned char *A, int A_bias, int lda,
                                                           const unsigned char *B, int B_bias, int ldb,
                                                                 unsigned char *C, int C_bias, int ldc,
                                                           int C_mult, int C_shift);
*/

 cublasStatus_t cublasSsyrk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *beta,
                                                      float *C,
                                                      int ldc);

 cublasStatus_t cublasDsyrk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *beta,
                                                      double *C,
                                                      int ldc);

 cublasStatus_t cublasCsyrk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *beta,
                                                      cuComplex *C,
                                                      int ldc);

 cublasStatus_t cublasZsyrk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);
/*
 cublasStatus_t cublasCsyrkEx ( cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc);


 cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      int n,
                                                      int k,
                                                      const cuComplex *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const cuComplex *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc);
*/

 cublasStatus_t cublasCherk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const float *beta,
                                                      cuComplex *C,
                                                      int ldc);

 cublasStatus_t cublasZherk(cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t Trans,
                                                      int n,
                                                      int k,
                                                      const double *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const double *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);

/*
 cublasStatus_t cublasCherkEx (cublasHandle_t handle,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      int n,
                                                      int k,
                                                      const float *alpha,
                                                      const void *A,
                                                      cudaDataType Atype,
                                                      int lda,
                                                      const float *beta,
                                                      void *C,
                                                      cudaDataType Ctype,
                                                      int ldc);


 cublasStatus_t cublasCherk3mEx (cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t TransA,
                                                       int n,
                                                       int k,
                                                       const float *alpha,
                                                       const void *A, cudaDataType Atype,
                                                       int lda,
                                                       const float *beta,
                                                       void *C,
                                                       cudaDataType Ctype,
                                                       int ldc);

*/


 cublasStatus_t cublasSsyr2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const float *alpha,
                                                       const float *A,
                                                       int lda,
                                                       const float *B,
                                                       int ldb,
                                                       const float *beta,
                                                       float *C,
                                                       int ldc);

 cublasStatus_t cublasDsyr2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const double *alpha,
                                                       const double *A,
                                                       int lda,
                                                       const double *B,
                                                       int ldb,
                                                       const double *beta,
                                                       double *C,
                                                       int ldc);

 cublasStatus_t cublasCsyr2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha,
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const cuComplex *beta,
                                                       cuComplex *C,
                                                       int ldc);

 cublasStatus_t cublasZsyr2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha,
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const cuDoubleComplex *beta,
                                                       cuDoubleComplex *C,
                                                       int ldc);

 cublasStatus_t cublasCher2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const cuComplex *alpha,
                                                       const cuComplex *A,
                                                       int lda,
                                                       const cuComplex *B,
                                                       int ldb,
                                                       const float *beta,
                                                       cuComplex *C,
                                                       int ldc);

 cublasStatus_t cublasZher2k(cublasHandle_t handle,
                                                       cublasFillMode_t uplo,
                                                       cublasOperation_t Trans,
                                                       int n,
                                                       int k,
                                                       const cuDoubleComplex *alpha,
                                                       const cuDoubleComplex *A,
                                                       int lda,
                                                       const cuDoubleComplex *B,
                                                       int ldb,
                                                       const double *beta,
                                                       cuDoubleComplex *C,
                                                       int ldc);

 cublasStatus_t cublasSsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const float *alpha,
                                                    const float *A,
                                                    int lda,
                                                    const float *B,
                                                    int ldb,
                                                    const float *beta,
                                                    float *C,
                                                    int ldc);

 cublasStatus_t cublasDsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const double *alpha,
                                                    const double *A,
                                                    int lda,
                                                    const double *B,
                                                    int ldb,
                                                    const double *beta,
                                                    double *C,
                                                    int ldc);

 cublasStatus_t cublasCsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha,
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const cuComplex *beta,
                                                    cuComplex *C,
                                                    int ldc);

 cublasStatus_t cublasZsyrkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha,
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const cuDoubleComplex *beta,
                                                    cuDoubleComplex *C,
                                                    int ldc);

 cublasStatus_t cublasCherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const cuComplex *alpha,
                                                    const cuComplex *A,
                                                    int lda,
                                                    const cuComplex *B,
                                                    int ldb,
                                                    const float *beta,
                                                    cuComplex *C,
                                                    int ldc);

 cublasStatus_t cublasZherkx (cublasHandle_t handle,
                                                    cublasFillMode_t uplo,
                                                    cublasOperation_t Trans,
                                                    int n,
                                                    int k,
                                                    const cuDoubleComplex *alpha,
                                                    const cuDoubleComplex *A,
                                                    int lda,
                                                    const cuDoubleComplex *B,
                                                    int ldb,
                                                    const double *beta,
                                                    cuDoubleComplex *C,
                                                    int ldc);

 cublasStatus_t cublasSsymm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      const float *beta,
                                                      float *C,
                                                      int ldc);

 cublasStatus_t cublasDsymm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      const double *beta,
                                                      double *C,
                                                      int ldc);

 cublasStatus_t cublasCsymm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta,
                                                      cuComplex *C,
                                                      int ldc);

 cublasStatus_t cublasZsymm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);


 cublasStatus_t cublasChemm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuComplex *alpha,
                                                      const cuComplex *A,
                                                      int lda,
                                                      const cuComplex *B,
                                                      int ldb,
                                                      const cuComplex *beta,
                                                      cuComplex *C,
                                                      int ldc);

 cublasStatus_t cublasZhemm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      int m,
                                                      int n,
                                                      const cuDoubleComplex *alpha,
                                                      const cuDoubleComplex *A,
                                                      int lda,
                                                      const cuDoubleComplex *B,
                                                      int ldb,
                                                      const cuDoubleComplex *beta,
                                                      cuDoubleComplex *C,
                                                      int ldc);


 cublasStatus_t cublasStrsm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      float *B,
                                                      int ldb);


 cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      double *B,
                                                      int ldb);

 cublasStatus_t cublasCtrsm(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t TransA,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha,
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *B,
                                                     int ldb);

 cublasStatus_t cublasZtrsm(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t TransA,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha,
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *B,
                                                     int ldb);


 cublasStatus_t cublasStrmm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const float *alpha,
                                                      const float *A,
                                                      int lda,
                                                      const float *B,
                                                      int ldb,
                                                      float *C,
                                                      int ldc);

 cublasStatus_t cublasDtrmm(cublasHandle_t handle,
                                                      cublasSideMode_t side,
                                                      cublasFillMode_t uplo,
                                                      cublasOperation_t TransA,
                                                      cublasDiagType_t diag,
                                                      int m,
                                                      int n,
                                                      const double *alpha,
                                                      const double *A,
                                                      int lda,
                                                      const double *B,
                                                      int ldb,
                                                      double *C,
                                                      int ldc);

 cublasStatus_t cublasCtrmm(cublasHandle_t handle,
                                                     cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t TransA,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuComplex *alpha,
                                                     const cuComplex *A,
                                                     int lda,
                                                     const cuComplex *B,
                                                     int ldb,
                                                     cuComplex *C,
                                                     int ldc);

 cublasStatus_t cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side,
                                                     cublasFillMode_t uplo,
                                                     cublasOperation_t TransA,
                                                     cublasDiagType_t diag,
                                                     int m,
                                                     int n,
                                                     const cuDoubleComplex *alpha,
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     const cuDoubleComplex *B,
                                                     int ldb,
                                                     cuDoubleComplex *C,
                                                     int ldc);
 /*

 cublasStatus_t cublasSgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t TransA,
                                                          cublasOperation_t TransB,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const float *alpha,
                                                          const float *Aarray[],
                                                          int lda,
                                                          const float *Barray[],
                                                          int ldb,
                                                          const float *beta,
                                                          float *Carray[],
                                                          int ldc,
                                                          int batchCount);

 cublasStatus_t cublasDgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t TransA,
                                                          cublasOperation_t TransB,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const double *alpha,
                                                          const double *Aarray[],
                                                          int lda,
                                                          const double *Barray[],
                                                          int ldb,
                                                          const double *beta,
                                                          double *Carray[],
                                                          int ldc,
                                                          int batchCount);

 cublasStatus_t cublasCgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t TransA,
                                                          cublasOperation_t TransB,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha,
                                                          const cuComplex *Aarray[],
                                                          int lda,
                                                          const cuComplex *Barray[],
                                                          int ldb,
                                                          const cuComplex *beta,
                                                          cuComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

 cublasStatus_t cublasCgemm3mBatched (cublasHandle_t handle,
                                                          cublasOperation_t TransA,
                                                          cublasOperation_t TransB,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuComplex *alpha,
                                                          const cuComplex *Aarray[],
                                                          int lda,
                                                          const cuComplex *Barray[],
                                                          int ldb,
                                                          const cuComplex *beta,
                                                          cuComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

 cublasStatus_t cublasZgemmBatched (cublasHandle_t handle,
                                                          cublasOperation_t TransA,
                                                          cublasOperation_t TransB,
                                                          int m,
                                                          int n,
                                                          int k,
                                                          const cuDoubleComplex *alpha,
                                                          const cuDoubleComplex *Aarray[],
                                                          int lda,
                                                          const cuDoubleComplex *Barray[],
                                                          int ldb,
                                                          const cuDoubleComplex *beta,
                                                          cuDoubleComplex *Carray[],
                                                          int ldc,
                                                          int batchCount);

 cublasStatus_t cublasSgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const float *alpha,
                                                                 const float *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const float *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const float *beta,
                                                                 float *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

 cublasStatus_t cublasDgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const double *alpha,
                                                                 const double *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const double *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const double *beta,
                                                                 double *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

 cublasStatus_t cublasCgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,
                                                                 const cuComplex *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const cuComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuComplex *beta,
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

 cublasStatus_t cublasCgemm3mStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuComplex *alpha,
                                                                 const cuComplex *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const cuComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuComplex *beta,
                                                                 cuComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);


 cublasStatus_t cublasZgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const cuDoubleComplex *alpha,
                                                                 const cuDoubleComplex *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const cuDoubleComplex *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const cuDoubleComplex *beta,
                                                                 cuDoubleComplex *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);

 cublasStatus_t cublasHgemmStridedBatched (cublasHandle_t handle,
                                                                 cublasOperation_t TransA,
                                                                 cublasOperation_t TransB,
                                                                 int m,
                                                                 int n,
                                                                 int k,
                                                                 const __half *alpha,
                                                                 const __half *A,
                                                                 int lda,
                                                                 long long int strideA,
                                                                 const __half *B,
                                                                 int ldb,
                                                                 long long int strideB,
                                                                 const __half *beta,
                                                                 __half *C,
                                                                 int ldc,
                                                                 long long int strideC,
                                                                 int batchCount);
*/


 cublasStatus_t cublasSgeam(cublasHandle_t handle,
                                                  cublasOperation_t TransA,
                                                  cublasOperation_t TransB,
                                                  int m,
                                                  int n,
                                                  const float *alpha,
                                                  const float *A,
                                                  int lda,
                                                  const float *beta ,
                                                  const float *B,
                                                  int ldb,
                                                  float *C,
                                                  int ldc);

 cublasStatus_t cublasDgeam(cublasHandle_t handle,
                                                  cublasOperation_t TransA,
                                                  cublasOperation_t TransB,
                                                  int m,
                                                  int n,
                                                  const double *alpha,
                                                  const double *A,
                                                  int lda,
                                                  const double *beta,
                                                  const double *B,
                                                  int ldb,
                                                  double *C,
                                                  int ldc);

 cublasStatus_t cublasCgeam(cublasHandle_t handle,
                                                  cublasOperation_t TransA,
                                                  cublasOperation_t TransB,
                                                  int m,
                                                  int n,
                                                  const cuComplex *alpha,
                                                  const cuComplex *A,
                                                  int lda,
                                                  const cuComplex *beta,
                                                  const cuComplex *B,
                                                  int ldb,
                                                  cuComplex *C,
                                                  int ldc);

 cublasStatus_t cublasZgeam(cublasHandle_t handle,
                                                  cublasOperation_t TransA,
                                                  cublasOperation_t TransB,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex *alpha,
                                                  const cuDoubleComplex *A,
                                                  int lda,
                                                  const cuDoubleComplex *beta,
                                                  const cuDoubleComplex *B,
                                                  int ldb,
                                                  cuDoubleComplex *C,
                                                  int ldc);

/*
 cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  float *A[],
                                                  int lda,
                                                  int *P,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  double *A[],
                                                  int lda,
                                                  int *P,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  cuComplex *A[],
                                                  int lda,
                                                  int *P,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle,
                                                  int n,
                                                  cuDoubleComplex *A[],
                                                  int lda,
                                                  int *P,
                                                  int *info,
                                                  int batchSize);


 cublasStatus_t cublasSgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const float *A[],
                                                  int lda,
                                                  const int *P,
                                                  float *C[],
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasDgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const double *A[],
                                                  int lda,
                                                  const int *P,
                                                  double *C[],
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasCgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuComplex *A[],
                                                  int lda,
                                                  const int *P,
                                                  cuComplex *C[],
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);

 cublasStatus_t cublasZgetriBatched(cublasHandle_t handle,
                                                  int n,
                                                  const cuDoubleComplex *A[],
                                                  int lda,
                                                  const int *P,
                                                  cuDoubleComplex *C[],
                                                  int ldc,
                                                  int *info,
                                                  int batchSize);



 cublasStatus_t cublasSgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t TransA,
                                                            int n,
                                                            int nrhs,
                                                            const float *Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            float *Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize);

 cublasStatus_t cublasDgetrsBatched( cublasHandle_t handle,
                                                           cublasOperation_t TransA,
                                                           int n,
                                                           int nrhs,
                                                           const double *Aarray[],
                                                           int lda,
                                                           const int *devIpiv,
                                                           double *Barray[],
                                                           int ldb,
                                                           int *info,
                                                           int batchSize);

 cublasStatus_t cublasCgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t TransA,
                                                            int n,
                                                            int nrhs,
                                                            const cuComplex *Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            cuComplex *Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize);


 cublasStatus_t cublasZgetrsBatched( cublasHandle_t handle,
                                                            cublasOperation_t TransA,
                                                            int n,
                                                            int nrhs,
                                                            const cuDoubleComplex *Aarray[],
                                                            int lda,
                                                            const int *devIpiv,
                                                            cuDoubleComplex *Barray[],
                                                            int ldb,
                                                            int *info,
                                                            int batchSize);




 cublasStatus_t cublasStrsmBatched( cublasHandle_t handle,
                                                          cublasSideMode_t side,
                                                          cublasFillMode_t uplo,
                                                          cublasOperation_t TransA,
                                                          cublasDiagType_t diag,
                                                          int m,
                                                          int n,
                                                          const float *alpha,
                                                          const float *A[],
                                                          int lda,
                                                          float *B[],
                                                          int ldb,
                                                          int batchCount);

 cublasStatus_t cublasDtrsmBatched( cublasHandle_t handle,
                                                          cublasSideMode_t side,
                                                          cublasFillMode_t uplo,
                                                          cublasOperation_t TransA,
                                                          cublasDiagType_t diag,
                                                          int m,
                                                          int n,
                                                          const double *alpha,
                                                          const double *A[],
                                                          int lda,
                                                          double *B[],
                                                          int ldb,
                                                          int batchCount);

 cublasStatus_t cublasCtrsmBatched( cublasHandle_t handle,
                                                          cublasSideMode_t side,
                                                          cublasFillMode_t uplo,
                                                          cublasOperation_t TransA,
                                                          cublasDiagType_t diag,
                                                          int m,
                                                          int n,
                                                          const cuComplex *alpha,
                                                          const cuComplex *A[],
                                                          int lda,
                                                          cuComplex *B[],
                                                          int ldb,
                                                          int batchCount);

 cublasStatus_t cublasZtrsmBatched( cublasHandle_t handle,
                                                          cublasSideMode_t side,
                                                          cublasFillMode_t uplo,
                                                          cublasOperation_t TransA,
                                                          cublasDiagType_t diag,
                                                          int m,
                                                          int n,
                                                          const cuDoubleComplex *alpha,
                                                          const cuDoubleComplex *A[],
                                                          int lda,
                                                          cuDoubleComplex *B[],
                                                          int ldb,
                                                          int batchCount);


 cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const float *A[],
                                                          int lda,
                                                          float *Ainv[],
                                                          int lda_inv,
                                                          int *info,
                                                          int batchSize);

 cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const double *A[],
                                                          int lda,
                                                          double *Ainv[],
                                                          int lda_inv,
                                                          int *info,
                                                          int batchSize);

 cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuComplex *A[],
                                                          int lda,
                                                          cuComplex *Ainv[],
                                                          int lda_inv,
                                                          int *info,
                                                          int batchSize);

 cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle,
                                                          int n,
                                                          const cuDoubleComplex *A[],
                                                          int lda,
                                                          cuDoubleComplex *Ainv[],
                                                          int lda_inv,
                                                          int *info,
                                                          int batchSize);


 cublasStatus_t cublasSgeqrfBatched( cublasHandle_t handle,
                                                           int m,
                                                           int n,
                                                           float *Aarray[],
                                                           int lda,
                                                           float *TauArray[],
                                                           int *info,
                                                           int batchSize);

 cublasStatus_t cublasDgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            double *Aarray[],
                                                            int lda,
                                                            double *TauArray[],
                                                            int *info,
                                                            int batchSize);

 cublasStatus_t cublasCgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            cuComplex *Aarray[],
                                                            int lda,
                                                            cuComplex *TauArray[],
                                                            int *info,
                                                            int batchSize);

 cublasStatus_t cublasZgeqrfBatched( cublasHandle_t handle,
                                                            int m,
                                                            int n,
                                                            cuDoubleComplex *Aarray[],
                                                            int lda,
                                                            cuDoubleComplex *TauArray[],
                                                            int *info,
                                                            int batchSize);

 cublasStatus_t cublasSgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t TransA,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           float *Aarray[],
                                                           int lda,
                                                           float *Carray[],
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize );

 cublasStatus_t cublasDgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t TransA,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           double *Aarray[],
                                                           int lda,
                                                           double *Carray[],
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize);

 cublasStatus_t cublasCgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t TransA,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuComplex *Aarray[],
                                                           int lda,
                                                           cuComplex *Carray[],
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize);

 cublasStatus_t cublasZgelsBatched( cublasHandle_t handle,
                                                           cublasOperation_t TransA,
                                                           int m,
                                                           int n,
                                                           int nrhs,
                                                           cuDoubleComplex *Aarray[],
                                                           int lda,
                                                           cuDoubleComplex *Carray[],
                                                           int ldc,
                                                           int *info,
                                                           int *devInfoArray,
                                                           int batchSize);
*/

 cublasStatus_t cublasSdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const float *A,
                                                  int lda,
                                                  const float *x,
                                                  int incX,
                                                  float *C,
                                                  int ldc);

 cublasStatus_t cublasDdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const double *A,
                                                  int lda,
                                                  const double *x,
                                                  int incX,
                                                  double *C,
                                                  int ldc);

 cublasStatus_t cublasCdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuComplex *A,
                                                  int lda,
                                                  const cuComplex *x,
                                                  int incX,
                                                  cuComplex *C,
                                                  int ldc);

 cublasStatus_t cublasZdgmm(cublasHandle_t handle,
                                                  cublasSideMode_t mode,
                                                  int m,
                                                  int n,
                                                  const cuDoubleComplex *A,
                                                  int lda,
                                                  const cuDoubleComplex *x,
                                                  int incX,
                                                  cuDoubleComplex *C,
                                                  int ldc);


 cublasStatus_t cublasStpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *AP,
                                                     float *A,
                                                     int lda );

 cublasStatus_t cublasDtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *AP,
                                                     double *A,
                                                     int lda );

 cublasStatus_t cublasCtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *AP,
                                                     cuComplex *A,
                                                     int lda );

 cublasStatus_t cublasZtpttr ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *AP,
                                                     cuDoubleComplex *A,
                                                     int lda );

 cublasStatus_t cublasStrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const float *A,
                                                     int lda,
                                                     float *AP );

 cublasStatus_t cublasDtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const double *A,
                                                     int lda,
                                                     double *AP );

 cublasStatus_t cublasCtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuComplex *A,
                                                     int lda,
                                                     cuComplex *AP );

 cublasStatus_t cublasZtrttp ( cublasHandle_t handle,
                                                     cublasFillMode_t uplo,
                                                     int n,
                                                     const cuDoubleComplex *A,
                                                     int lda,
                                                     cuDoubleComplex *AP );
