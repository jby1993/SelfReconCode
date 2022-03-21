#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void interp2x_boundary3d_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5> input,
    torch::PackedTensorAccessor32<scalar_t,5> output,
    torch::PackedTensorAccessor32<bool,5> is_boundary,
    const float balance_value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    const int bn = output.size(0);
    const int c = output.size(1);
    const int d = output.size(2);
    const int h = output.size(3);
    const int w = output.size(4);

    if (i >= bn * c * d * h * w) {
        return;
    }
    
    const int x = i % w;
    const int y = (i / w) % h;
    const int z = (i / (h * w)) % d;
    const int ci = (i / (d * h * w)) % c;
    const int bi = i / (c * d * h * w);

    const bool skip_x = x % 2 == 0;
    const bool skip_y = y % 2 == 0;
    const bool skip_z = z % 2 == 0; 

    if (skip_x && skip_y && skip_z){
        output[bi][ci][z][y][x] = input[bi][ci][z/2][y/2][x/2];
        is_boundary[bi][ci][z][y][x] = false;
        return;

    }else if (skip_x && skip_z){
        auto v1 = input[bi][ci][z/2][(y-1)/2][x/2];
        auto v2 = input[bi][ci][z/2][(y+1)/2][x/2];
        output[bi][ci][z][y][x] = (v1 + v2) / 2.;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        if (flag1 == flag2){is_boundary[bi][ci][z][y][x] = false;}
        else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }else if (skip_y && skip_z){
        auto v1 = input[bi][ci][z/2][y/2][(x-1)/2];
        auto v2 = input[bi][ci][z/2][y/2][(x+1)/2];
        output[bi][ci][z][y][x] = (v1 + v2) / 2.;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        if (flag1 == flag2){is_boundary[bi][ci][z][y][x] = false;}
        else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }else if (skip_x && skip_y){
        auto v1 = input[bi][ci][(z-1)/2][y/2][x/2];
        auto v2 = input[bi][ci][(z+1)/2][y/2][x/2];
        output[bi][ci][z][y][x] = (v1 + v2) / 2.;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        if (flag1 == flag2){is_boundary[bi][ci][z][y][x] = false;}
        else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }else if (skip_z){
        auto v1 = input[bi][ci][z/2][(y-1)/2][(x-1)/2];
        auto v2 = input[bi][ci][z/2][(y-1)/2][(x+1)/2]; 
        auto v3 = input[bi][ci][z/2][(y+1)/2][(x-1)/2]; 
        auto v4 = input[bi][ci][z/2][(y+1)/2][(x+1)/2];
        output[bi][ci][z][y][x] = (v1 + v2 + v3 + v4) / 4.0;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        bool flag3 = v3 > balance_value;
        bool flag4 = v4 > balance_value;
        if (flag1 == flag2 && flag2 == flag3 && flag3 == flag4){
            is_boundary[bi][ci][z][y][x] = false;
        }else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }else if (skip_x){
        auto v1 = input[bi][ci][(z-1)/2][(y-1)/2][x/2];
        auto v2 = input[bi][ci][(z+1)/2][(y-1)/2][x/2]; 
        auto v3 = input[bi][ci][(z-1)/2][(y+1)/2][x/2]; 
        auto v4 = input[bi][ci][(z+1)/2][(y+1)/2][x/2];
        output[bi][ci][z][y][x] = (v1 + v2 + v3 + v4) / 4.0;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        bool flag3 = v3 > balance_value;
        bool flag4 = v4 > balance_value;
        if (flag1 == flag2 && flag2 == flag3 && flag3 == flag4){
            is_boundary[bi][ci][z][y][x] = false;
        }else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }else if (skip_y){
        auto v1 = input[bi][ci][(z-1)/2][y/2][(x-1)/2];
        auto v2 = input[bi][ci][(z+1)/2][y/2][(x-1)/2]; 
        auto v3 = input[bi][ci][(z-1)/2][y/2][(x+1)/2]; 
        auto v4 = input[bi][ci][(z+1)/2][y/2][(x+1)/2];
        output[bi][ci][z][y][x] = (v1 + v2 + v3 + v4) / 4.0;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        bool flag3 = v3 > balance_value;
        bool flag4 = v4 > balance_value;
        if (flag1 == flag2 && flag2 == flag3 && flag3 == flag4){
            is_boundary[bi][ci][z][y][x] = false;
        }else{is_boundary[bi][ci][z][y][x] = true;}
        return;
    
    }else{
        auto v1 = input[bi][ci][(z-1)/2][(y-1)/2][(x-1)/2];
        auto v2 = input[bi][ci][(z-1)/2][(y-1)/2][(x+1)/2];
        auto v3 = input[bi][ci][(z-1)/2][(y+1)/2][(x-1)/2];
        auto v4 = input[bi][ci][(z-1)/2][(y+1)/2][(x+1)/2];
        auto v5 = input[bi][ci][(z+1)/2][(y-1)/2][(x-1)/2];
        auto v6 = input[bi][ci][(z+1)/2][(y-1)/2][(x+1)/2];
        auto v7 = input[bi][ci][(z+1)/2][(y+1)/2][(x-1)/2];
        auto v8 = input[bi][ci][(z+1)/2][(y+1)/2][(x+1)/2];
        output[bi][ci][z][y][x] = (v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8) / 8.0;

        bool flag1 = v1 > balance_value;
        bool flag2 = v2 > balance_value;
        bool flag3 = v3 > balance_value;
        bool flag4 = v4 > balance_value;
        bool flag5 = v5 > balance_value;
        bool flag6 = v6 > balance_value;
        bool flag7 = v7 > balance_value;
        bool flag8 = v8 > balance_value;
        if (flag1 == flag2 && flag2 == flag3 && flag3 == flag4 && 
            flag4 == flag5 && flag5 == flag6 && flag6 == flag7 && 
            flag7 == flag8){
            is_boundary[bi][ci][z][y][x] = false;
        }else{is_boundary[bi][ci][z][y][x] = true;}
        return;

    }
}


template <typename scalar_t>
__global__ void interp2x_boundary3d_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5> grad_output,
    torch::PackedTensorAccessor32<scalar_t,5> grad_input) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const int bn = grad_input.size(0);
    const int c = grad_input.size(1);
    const int d = grad_input.size(2);
    const int h = grad_input.size(3);
    const int w = grad_input.size(4);

    if (i >= bn * c * d * h * w) {
        return;
    }
    
    const int x = i % w;
    const int y = (i / w) % h;
    const int z = (i / (h * w)) % d;
    const int ci = (i / (d * h * w)) % c;
    const int bi = i / (c * d * h * w);

    auto grad = grad_output[bi][ci][z*2][y*2][x*2];

    // 6 edges
    if (x > 0){
        grad += grad_output[bi][ci][z*2][y*2][x*2 - 1] / 2.0;}
    if (x < w - 1){
        grad += grad_output[bi][ci][z*2][y*2][x*2 + 1] / 2.0;}
    if (y > 0){
        grad += grad_output[bi][ci][z*2][y*2 - 1][x*2] / 2.0;}
    if (y < h - 1){
        grad += grad_output[bi][ci][z*2][y*2 + 1][x*2] / 2.0;}
    if (z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2][x*2] / 2.0;}
    if (z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2][x*2] / 2.0;}
    
    // 12 faces: xy
    if (x > 0 && y > 0){
        grad += grad_output[bi][ci][z*2][y*2 - 1][x*2 - 1] / 4.0;}
    if (x < w - 1 && y > 0){
        grad += grad_output[bi][ci][z*2][y*2 - 1][x*2 + 1] / 4.0;}
    if (x > 0 && y < h - 1){
        grad += grad_output[bi][ci][z*2][y*2 + 1][x*2 - 1] / 4.0;}
    if (x < w - 1 && y < h - 1){
        grad += grad_output[bi][ci][z*2][y*2 + 1][x*2 + 1] / 4.0;}
    // 12 faces: xz
    if (x > 0 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2][x*2 - 1] / 4.0;}
    if (x < w - 1 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2][x*2 + 1] / 4.0;}
    if (x > 0 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2][x*2 - 1] / 4.0;}
    if (x < w - 1 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2][x*2 + 1] / 4.0;}
    // 12 faces: yz
    if (y > 0 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 - 1][x*2] / 4.0;}
    if (y < h - 1 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 + 1][x*2] / 4.0;}
    if (y > 0 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 - 1][x*2] / 4.0;}
    if (y < h - 1 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 + 1][x*2] / 4.0;}

    // 8 corners
    if (x > 0 && y > 0 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 - 1][x*2 - 1] / 8.0;}
    if (x < w - 1 && y > 0 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 - 1][x*2 + 1] / 8.0;}
    if (x > 0 && y < h - 1 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 + 1][x*2 - 1] / 8.0;}
    if (x < w - 1 && y < h - 1 && z > 0){
        grad += grad_output[bi][ci][z*2 - 1][y*2 + 1][x*2 + 1] / 8.0;}
    if (x > 0 && y > 0 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 - 1][x*2 - 1] / 8.0;}
    if (x < w - 1 && y > 0 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 - 1][x*2 + 1] / 8.0;}
    if (x > 0 && y < h - 1 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 + 1][x*2 - 1] / 8.0;}
    if (x < w - 1 && y < h - 1 && z < d - 1){
        grad += grad_output[bi][ci][z*2 + 1][y*2 + 1][x*2 + 1] / 8.0;}
    
    grad_input[bi][ci][z][y][x] = grad;
    }
} // namespace


std::vector<torch::Tensor> interp2x_boundary3d_cuda_forward(
    const torch::Tensor& input, 
    const float balance_value) {
    
    torch::Device device = input.device();
    int bn = input.size(0);
    int c = input.size(1);
    int d = input.size(2) * 2 - 1;
    int h = input.size(3) * 2 - 1;
    int w = input.size(4) * 2 - 1;
    
    auto option1 = torch::TensorOptions().dtype(input.scalar_type()).device(device);
    auto output = torch::empty({bn, c, d, h, w}, option1);

    auto option2 = torch::TensorOptions().dtype(torch::ScalarType::Bool).device(device);
    auto is_boundary = torch::empty({bn, c, d, h, w}, option2);

    const int num_kernels = bn * c * d * h * w;
    const int num_threads = 1024;
    const dim3 blocks((num_kernels + num_threads - 1) / num_threads);

    AT_DISPATCH_FLOATING_TYPES(
        output.scalar_type(), "interp2x_boundary3d_cuda_forward", ([&] {
            interp2x_boundary3d_cuda_forward_kernel<scalar_t>
                <<<blocks, num_threads>>>(
                    input.packed_accessor32<scalar_t, 5>(), 
                    output.packed_accessor32<scalar_t, 5>(),
                    is_boundary.packed_accessor32<bool, 5>(),
                    balance_value);
    }));

    return {output, is_boundary};
}


torch::Tensor interp2x_boundary3d_cuda_backward(
    const torch::Tensor& grad_output) {
    
    torch::Device device = grad_output.device();
    int bn = grad_output.size(0);
    int c = grad_output.size(1);
    int d = (grad_output.size(2) + 1) / 2;
    int h = (grad_output.size(3) + 1) / 2;
    int w = (grad_output.size(4) + 1) / 2;

    auto option = torch::TensorOptions().dtype(grad_output.scalar_type()).device(device);
    auto grad_input = torch::empty({bn, c, d, h, w}, option);
    
    const int num_kernels = bn * c * d * h * w;
    const int num_threads = 1024;
    const dim3 blocks((num_kernels + num_threads - 1) / num_threads);

    AT_DISPATCH_FLOATING_TYPES(
        grad_output.scalar_type(), "interp2x_boundary3d_cuda_backward", ([&] {
            interp2x_boundary3d_cuda_backward_kernel<scalar_t>
                <<<blocks, num_threads>>>(
                    grad_output.packed_accessor32<scalar_t, 5>(), 
                    grad_input.packed_accessor32<scalar_t, 5>());
    }));

    return grad_input;
}
