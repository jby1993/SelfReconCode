#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void M3x3Inv_float(const float* ms, float* invs, bool* checks,int N);
void M3x3Inv_double(const double* ms, double* invs, bool* checks,int N);
void M3x3Inv_backward_float(const float* grads, const float* invs, float* outs,int N);
void M3x3Inv_backward_double(const double* grads, const double* invs, double* outs,int N);
std::vector<at::Tensor> Fast3x3Minv(at::Tensor ms)
{
	CHECK_INPUT(ms);
	AT_ASSERTM(ms.type().scalarType()==at::ScalarType::Float||ms.type().scalarType()==at::ScalarType::Double, "rs must be a float/double tensor");
	int N=ms.size(0);
	cudaSetDevice(ms.get_device());
	auto options_float_nograd = torch::TensorOptions()
	                            .dtype(ms.dtype())
	                            .layout(ms.layout())
	                            .device(ms.device())
	                            .requires_grad(false);
	auto options_bool_nograd = torch::TensorOptions()
	                            .dtype(torch::Dtype::Bool)
	                            .layout(ms.layout())
	                            .device(ms.device())
	                            .requires_grad(false);
	auto invs=torch::zeros({N,3,3},options_float_nograd);
	auto checks=torch::zeros({N},options_bool_nograd);
	if(ms.type().scalarType()==at::ScalarType::Float)
		M3x3Inv_float(ms.data<float>(), invs.data<float>(), checks.data<bool>(),N);
	else
		M3x3Inv_double(ms.data<double>(), invs.data<double>(), checks.data<bool>(),N);
	return {invs,checks};

}

at::Tensor Fast3x3Minv_backward(at::Tensor grads, at::Tensor invs)
{
	CHECK_INPUT(grads);
	CHECK_INPUT(invs);
	AT_ASSERTM(grads.type().scalarType()==at::ScalarType::Float||grads.type().scalarType()==at::ScalarType::Double, "grads must be a float/double tensor");
	AT_ASSERTM(invs.type().scalarType()==at::ScalarType::Float||invs.type().scalarType()==at::ScalarType::Double, "invs must be a float/double tensor");
	AT_ASSERTM(invs.type().scalarType()==grads.type().scalarType(), "invs must have same type with grads");
	int N=invs.size(0);
	cudaSetDevice(invs.get_device());
	auto options_float_nograd = torch::TensorOptions()
	                            .dtype(invs.dtype())
	                            .layout(invs.layout())
	                            .device(invs.device())
	                            .requires_grad(false);
	auto outs=torch::zeros({N,3,3},options_float_nograd);
	if(invs.type().scalarType()==at::ScalarType::Float)
		M3x3Inv_backward_float(grads.data<float>(), invs.data<float>(), outs.data<float>(),N);
	else
		M3x3Inv_backward_double(grads.data<double>(), invs.data<double>(), outs.data<double>(),N);
	return outs;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("Fast3x3Minv", &Fast3x3Minv, "fast batch 3x3 matrix inversion (CUDA)");
  m.def("Fast3x3Minv_backward", &Fast3x3Minv_backward, "fast batch 3x3 matrix inversion backward (CUDA)");
}