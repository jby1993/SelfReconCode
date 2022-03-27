#include <cuda_runtime.h>

#include <iostream>
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

__inline__ __device__ int index(int r, int c)
{
	return 3*r+c;
}
template <typename scalar_t>
__global__ void cu3x3MInv(const scalar_t* ms, scalar_t* invs, bool* checks, int N)
{
	int mid=threadIdx.x+blockIdx.x*blockDim.x;
	if(mid>=N)
		return;
	scalar_t* inv=invs+9*mid;
	const scalar_t* m=ms+9*mid;
	scalar_t cof00=m[index(1,1)]*m[index(2,2)]-m[index(1,2)]*m[index(2,1)];
	scalar_t cof01=-m[index(1,0)]*m[index(2,2)]+m[index(1,2)]*m[index(2,0)];
	scalar_t cof02=m[index(1,0)]*m[index(2,1)]-m[index(1,1)]*m[index(2,0)];
	scalar_t cof10=-m[index(0,1)]*m[index(2,2)]+m[index(0,2)]*m[index(2,1)];
	scalar_t cof11=m[index(0,0)]*m[index(2,2)]-m[index(0,2)]*m[index(2,0)];
	scalar_t cof12=-m[index(0,0)]*m[index(2,1)]+m[index(0,1)]*m[index(2,0)];
	scalar_t cof20=m[index(0,1)]*m[index(1,2)]-m[index(0,2)]*m[index(1,1)];
	scalar_t cof21=-m[index(0,0)]*m[index(1,2)]+m[index(0,2)]*m[index(1,0)];
	scalar_t cof22=m[index(0,0)]*m[index(1,1)]-m[index(0,1)]*m[index(1,0)];

	scalar_t det=m[index(0,0)]*cof00+m[index(0,1)]*cof01+m[index(0,2)]*cof02;
	if(fabs(det)<0.0001)
	{
		#pragma unroll 9 //const time number seemly default unroll? Anyway, here I write down explicitly.
		for(int i=0;i<9;i++)
			// for now, seting inv as zeros
			inv[i]=0.;
		checks[mid]=false;
	}
	else
	{
		inv[index(0,0)]=cof00/det;
		inv[index(0,1)]=cof10/det;
		inv[index(0,2)]=cof20/det;
		inv[index(1,0)]=cof01/det;
		inv[index(1,1)]=cof11/det;
		inv[index(1,2)]=cof21/det;
		inv[index(2,0)]=cof02/det;
		inv[index(2,1)]=cof12/det;
		inv[index(2,2)]=cof22/det;
		checks[mid]=true;
	}
}

template <typename scalar_t>
__global__ void cu3x3MInv_backward(const scalar_t* grads, const scalar_t* invs, scalar_t* outs, int N)
{
	int mid=threadIdx.x+blockIdx.x*blockDim.x;
	if(mid>=N)
		return;
	const scalar_t* inv=invs+9*mid;
	const scalar_t* g=grads+9*mid;
	scalar_t* out=outs+9*mid;
	scalar_t c00=inv[index(0,0)];
	scalar_t c01=inv[index(0,1)];
	scalar_t c02=inv[index(0,2)];
	scalar_t c10=inv[index(1,0)];
	scalar_t c11=inv[index(1,1)];
	scalar_t c12=inv[index(1,2)];
	scalar_t c20=inv[index(2,0)];
	scalar_t c21=inv[index(2,1)];
	scalar_t c22=inv[index(2,2)];

	scalar_t g00=g[index(0,0)];
	scalar_t g01=g[index(0,1)];
	scalar_t g02=g[index(0,2)];
	scalar_t g10=g[index(1,0)];
	scalar_t g11=g[index(1,1)];
	scalar_t g12=g[index(1,2)];
	scalar_t g20=g[index(2,0)];
	scalar_t g21=g[index(2,1)];
	scalar_t g22=g[index(2,2)];

	out[index(0,0)] = -(g00*c00*c00+g01*c00*c01+g02*c00*c02+g10*c10*c00+g11*c10*c01+g12*c10*c02+g20*c20*c00+g21*c20*c01+g22*c20*c02);
	out[index(0,1)] = -(g00*c00*c10+g01*c00*c11+g02*c00*c12+g10*c10*c10+g11*c10*c11+g12*c10*c12+g20*c20*c10+g21*c20*c11+g22*c20*c12);
	out[index(0,2)] = -(g00*c00*c20+g01*c00*c21+g02*c00*c22+g10*c10*c20+g11*c10*c21+g12*c10*c22+g20*c20*c20+g21*c20*c21+g22*c20*c22);

	out[index(1,0)] = -(g00*c01*c00+g01*c01*c01+g02*c01*c02+g10*c11*c00+g11*c11*c01+g12*c11*c02+g20*c21*c00+g21*c21*c01+g22*c21*c02);
	out[index(1,1)] = -(g00*c01*c10+g01*c01*c11+g02*c01*c12+g10*c11*c10+g11*c11*c11+g12*c11*c12+g20*c21*c10+g21*c21*c11+g22*c21*c12);
	out[index(1,2)] = -(g00*c01*c20+g01*c01*c21+g02*c01*c22+g10*c11*c20+g11*c11*c21+g12*c11*c22+g20*c21*c20+g21*c21*c21+g22*c21*c22);

	out[index(2,0)] = -(g00*c02*c00+g01*c02*c01+g02*c02*c02+g10*c12*c00+g11*c12*c01+g12*c12*c02+g20*c22*c00+g21*c22*c01+g22*c22*c02);
	out[index(2,1)] = -(g00*c02*c10+g01*c02*c11+g02*c02*c12+g10*c12*c10+g11*c12*c11+g12*c12*c12+g20*c22*c10+g21*c22*c11+g22*c22*c12);
	out[index(2,2)] = -(g00*c02*c20+g01*c02*c21+g02*c02*c22+g10*c12*c20+g11*c12*c21+g12*c12*c22+g20*c22*c20+g21*c22*c21+g22*c22*c22);

}

void M3x3Inv_float(const float* ms, float* invs, bool* checks,int N)
{	
	int threads=1024;
	int blocks=N/threads;
	if(blocks*threads<N)
		blocks+=1;
	cu3x3MInv<float><<<blocks,threads>>>(ms,invs,checks,N);
}


void M3x3Inv_double(const double* ms, double* invs, bool* checks,int N)
{	
	int threads=1024;
	int blocks=N/threads;
	if(blocks*threads<N)
		blocks+=1;
	cu3x3MInv<double><<<blocks,threads>>>(ms,invs,checks,N);
}

void M3x3Inv_backward_float(const float* grads, const float* invs, float* outs,int N)
{	
	int threads=1024;
	int blocks=N/threads;
	if(blocks*threads<N)
		blocks+=1;
	cu3x3MInv_backward<float><<<blocks,threads>>>(grads,invs,outs,N);
}


void M3x3Inv_backward_double(const double* grads, const double* invs, double* outs,int N)
{	
	int threads=1024;
	int blocks=N/threads;
	if(blocks*threads<N)
		blocks+=1;
	cu3x3MInv_backward<double><<<blocks,threads>>>(grads,invs,outs,N);
}