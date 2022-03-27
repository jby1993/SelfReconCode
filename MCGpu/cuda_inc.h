#ifndef CUDA_INC_H
#define CUDA_INC_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
//#include <cblas.h>
#include <cusolverDn.h>

inline const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << cublasGetErrorString(status); \
  } while (0)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(int N) {
    int to_return = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    if (to_return>65535){ to_return = 65535;}
  return to_return;
}

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if(error!=cudaSuccess) \
    { \
      std::cerr<<"Error: "<<cudaGetErrorString(error)<<std::endl; \
      exit(1); \
    } \
  } while (0)

#define CUDA_POST_KERNEL_CHECK cudaDeviceSynchronize();CUDA_CHECK(cudaPeekAtLastError())

struct pixel
{
    int x, y;
    int v1index, v2index, v3index;
    float coord[3];
};


#endif // CUDA_INC_H