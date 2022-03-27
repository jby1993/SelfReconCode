#include <torch/extension.h>
#include <vector>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#include "CudaKernels.h"
#include <cuda_runtime.h>
//batch_pointcloud(batchsize,ndim,maxPnum), not all pointcloud has same vertex number, but not greater than maxPnum
//batch_query(ndim,all_query_size), splice all querys into one
//query_to_batchIds(all_query_size), indicate each query's batch id
//query_to_pc_sizes(all_query_size), indicate each query corresponding pc vertex number
//knn_dists(all_query_size,k), each query knearst's distances
//knn_indexs(all_query_size,k), each query knearst's indexs relate to corresponding to pointcloud
void mc_init(int device_id)
{
	MCGpu::Get(device_id);
}
//if sdfs volume increase beyond ever lagest, we need to reallocate gpu memory in gpu, this will lead slow run
//better to use with same volume, like during training
std::vector<at::Tensor> mc_gpu(at::Tensor sdfs,
                     	float xstep=1.0,
                     	float ystep=1.0,
                     	float zstep=1.0,
                     	float xmin=0.0,
                     	float ymin=0.0,
                     	float zmin=0.0,
						float fTargetValue=0.0
                     )
{
	CHECK_INPUT(sdfs);
	auto options_float_nograd = torch::TensorOptions()
                                    .dtype(sdfs.dtype())
                                    .layout(sdfs.layout())
                                    .device(sdfs.device())
                                    .requires_grad(false);
    auto options_long_nograd = torch::TensorOptions()
                                    .dtype(torch::kInt64)
                                    .layout(sdfs.layout())
                                    .device(sdfs.device())
                                    .requires_grad(false);
	if(sdfs.dtype()!=options_float_nograd.dtype())
		return std::vector<at::Tensor>();
	int device_id=sdfs.get_device();
	if(device_id<0||device_id>=8)
		return std::vector<at::Tensor>();
	cudaSetDevice(device_id);
	if(!MCGpu::Get(device_id).init(sdfs.size(0),sdfs.size(1),sdfs.size(2)))
		return std::vector<at::Tensor>();
	MCGpu::Get(device_id).MC(sdfs.data<float>(),fTargetValue);
	MCGpu::Get(device_id).scaleVertices(xstep,ystep,zstep,xmin,ymin,zmin);	
    auto vertices=torch::zeros({MCGpu::Get(device_id).number_record_[0],3},options_float_nograd);
    auto faces=torch::zeros({MCGpu::Get(device_id).number_record_[1],3},options_long_nograd);
    cudaMemcpy(vertices.data<float>(),MCGpu::Get(device_id).d_points_coor_,sizeof(float)*3*MCGpu::Get(device_id).number_record_[0],cudaMemcpyDeviceToDevice);
    cudaMemcpy(faces.data<long int>(),MCGpu::Get(device_id).d_faces_index_,sizeof(long int)*3*MCGpu::Get(device_id).number_record_[1],cudaMemcpyDeviceToDevice);	
	return {vertices,faces};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mc_gpu", &mc_gpu, "marching cube cuda global (CUDA)");
  m.def("mc_init", &mc_init, "marching cube init a gpu device, not necessary for mc_gpu");
}