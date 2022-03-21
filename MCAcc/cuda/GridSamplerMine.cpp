#include <ATen/native/GridSampler.h>
#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

//note: if torch version>1.7.0, use AT_CHECK to replace TORCH_CHECK

namespace at { namespace native {
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;
Tensor grid_sampler_3d_cuda_mine(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode);

std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda_mine(const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                              int64_t interpolation_mode, int64_t padding_mode);

std::tuple<Tensor, Tensor, Tensor>
grid_sampler_3d_backward_backward_cuda_mine(const Tensor& grad_output_input, const Tensor& grad_output_grid, const Tensor& grad_output, const Tensor& input, const Tensor& grid,
                              int64_t interpolation_mode, int64_t padding_mode);

void check(const Tensor& input, const Tensor& grid,
                    int64_t interpolation_mode, int64_t padding_mode)
{
  TORCH_CHECK(
    input.defined() && grid.defined(),
    "grid_sampler(): expected input and grid to not be undefined, but input "
    "is ", input, " and grid is ", grid);
  auto input_opt = input.options();
  auto grid_opt = grid.options();
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  TORCH_CHECK(
    input_opt.dtype() == grid_opt.dtype(),
    "grid_sampler(): expected input and grid to have same dtype, but input "
    "has ", input_opt.dtype(), " and grid has ", grid_opt.dtype());
  TORCH_CHECK(
    input_opt.layout() == kStrided && grid_opt.layout() == kStrided,
    "grid_sampler(): expected input and grid to have torch.strided layout, but "
    "input has ", input_opt.layout(), " and grid has ", grid_opt.layout());
  TORCH_CHECK(
    (input.dim() == 5) && input.dim() == grid.dim(),
    "grid_sampler(): expected 5D input and grid with same number of "
    "dimensions, but got input with sizes ", input.sizes(),
    " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    input.size(0) == grid.size(0),
    "grid_sampler(): expected grid and input to have same batch size, but got "
    "input with sizes ", input.sizes(), " and grid with sizes ", grid.sizes());
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());
  TORCH_CHECK(
    static_cast<GridSamplerInterpolation>(interpolation_mode) == GridSamplerInterpolation::Bilinear,
    "grid_sampler(): only support Bilinear now");
  TORCH_CHECK(
    static_cast<GridSamplerPadding>(padding_mode) == GridSamplerPadding::Border,
    "grid_sampler(): only support Border Padding now");
  for (int64_t i = 2; i < input.dim(); i++) {
    TORCH_CHECK(input.size(i) > 0,
      "grid_sampler(): expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }
  cudaSetDevice(input.get_device());
}

Tensor grid_sampler_forward_mine(const Tensor& input, const Tensor& grid,
                    int64_t interpolation_mode, int64_t padding_mode) {
  
  check(input,grid,interpolation_mode,padding_mode);

  return grid_sampler_3d_cuda_mine(input, grid, interpolation_mode, padding_mode);
}


std::tuple<Tensor, Tensor> grid_sampler_3d_backward_mine(const Tensor& input, const Tensor& grid,const Tensor& grad_output,
                              int64_t interpolation_mode, int64_t padding_mode) {
  
  check(input,grid,interpolation_mode,padding_mode);

  return grid_sampler_3d_backward_cuda_mine(input, grid, grad_output, interpolation_mode, padding_mode);
}

std::tuple<Tensor, Tensor, Tensor> grid_sampler_3d_backward_backward_mine(const Tensor& grad_output_input, const Tensor& grad_output_grid, const Tensor& input, const Tensor& grid, const Tensor& grad_output,
                              int64_t interpolation_mode, int64_t padding_mode) {
  
  check(input,grid,interpolation_mode,padding_mode);

  return grid_sampler_3d_backward_backward_cuda_mine(grad_output_input, grad_output_grid, input, grid, grad_output, interpolation_mode, padding_mode);
}

}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &at::native::grid_sampler_forward_mine, "forward (CUDA)");
  m.def("backward", &at::native::grid_sampler_3d_backward_mine, "backward (CUDA)");
  m.def("dbackward", &at::native::grid_sampler_3d_backward_backward_mine, "double backward (CUDA)");
}