#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> interp2x_boundary3d_cuda_forward(
    const torch::Tensor& input, 
    const float balance_value);

torch::Tensor interp2x_boundary3d_cuda_backward(
    const torch::Tensor& grad_output);


#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> interp2x_boundary3d_forward(
    const torch::Tensor& input, 
    const float balance_value) {
      CHECK_INPUT(input);

      return interp2x_boundary3d_cuda_forward(input, balance_value);
}


torch::Tensor interp2x_boundary3d_backward(
    const torch::Tensor& grad_output) {
      CHECK_INPUT(grad_output);

      return interp2x_boundary3d_cuda_backward(grad_output);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &interp2x_boundary3d_forward, "forward (CUDA)");
  m.def("backward", &interp2x_boundary3d_backward, "backward (CUDA)");
}