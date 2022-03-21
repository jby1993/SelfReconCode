#include <ATen/ATen.h>
#include <ATen/native/GridSampler.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

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

namespace at { namespace native {
using namespace at::cuda::detail;

using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;


namespace {
  // Clips coordinates to between 0 and clip_limit - 1
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t clip_coordinates(scalar_t in, int clip_limit) {
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
  }



  // clip_coordinates_set_grad works similarly to clip_coordinates except that
  // it also returns the `d output / d input` via pointer argument `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t *grad_in) {
    // Note that it is important for the gradient calculation that borders
    // are considered out of bounds.
    if (in <= static_cast<scalar_t>(0)) {
      *grad_in = static_cast<scalar_t>(0);
      return static_cast<scalar_t>(0);
    } else {
      scalar_t max = static_cast<scalar_t>(clip_limit - 1);
      if (in >= max) {
        *grad_in = static_cast<scalar_t>(0);
        return max;
      } else {
        *grad_in = static_cast<scalar_t>(1);
        return in;
      }
    }
  }

  // Reflects coordinates until they fall between low and high (inclusive).
  // The bounds are passed as twice their value so that half-integer values
  // can be represented as ints.
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
    if (twice_low == twice_high) {
      return static_cast<scalar_t>(0);
    }
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = ::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 == 0) {
      return extra + min;
    } else {
      return span - extra + min;
    }
  }

  // reflect_coordinates_set_grad works similarly to reflect_coordinates except
  // that it also returns the `d output / d input` via pointer argument
  // `grad_in`.
  // This is useful in the backward pass of grid_sampler.
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                                        scalar_t *grad_in) {
    if (twice_low == twice_high) {
      *grad_in = static_cast<scalar_t>(0);
      return static_cast<scalar_t>(0);
    }
    int grad_in_mult_;
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = in - min;
    if (in < static_cast<scalar_t>(0)) {
      grad_in_mult_ = -1;
      in = -in;
    } else {
      grad_in_mult_ = 1;
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 == 0) {
      *grad_in = static_cast<scalar_t>(grad_in_mult_);
      return extra + min;
    } else {
      *grad_in = static_cast<scalar_t>(-grad_in_mult_);
      return span - extra + min;
    }
  }

template<typename scalar_t> 
static __forceinline__ __device__ 
scalar_t safe_downgrade_to_int_range(scalar_t x){
  // -100.0 does not have special meaning. This is just to make sure 
  // it's not within_bounds_2d or within_bounds_3d, and does not cause 
  // undefined behavior. See #35506.  
  if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x))) 
    return static_cast<scalar_t>(-100.0); 
  return x;
}

  static __forceinline__ __device__
  bool within_bounds_2d(int h, int w, int H, int W) {
    return h >= 0 && h < H && w >= 0 && w < W;
  }

  static __forceinline__ __device__
  bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }

  template<typename scalar_t>
  static __forceinline__ __device__
  void safe_add_2d(scalar_t *data, int h, int w,
                   int sH, int sW, int H, int W,
                   scalar_t delta) {
    if (within_bounds_2d(h, w, H, W)) {
      atomicAdd(data + h * sH + w * sW, delta);
    }
  }

  template<typename scalar_t>
  static __forceinline__ __device__
  void safe_add_3d(scalar_t *data, int d, int h, int w,
                   int sD, int sH, int sW, int D, int H, int W,
                   scalar_t delta) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
      atomicAdd(data + d * sD + h * sH + w * sW, delta);
    }
  }


  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_3d_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> output,
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_D = input.sizes[2];
    int inp_H = input.sizes[3];
    int inp_W = input.sizes[4];
    int out_D = grid.sizes[1];
    int out_H = grid.sizes[2];
    int out_W = grid.sizes[3];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sD = input.strides[2];
    int inp_sH = input.strides[3];
    int inp_sW = input.strides[4];
    int grid_sN = grid.strides[0];
    int grid_sD = grid.strides[1];
    int grid_sH = grid.strides[2];
    int grid_sW = grid.strides[3];
    int grid_sCoor = grid.strides[4];
    int out_sN = output.strides[0];
    int out_sC = output.strides[1];
    int out_sD = output.strides[2];
    int out_sH = output.strides[3];
    int out_sW = output.strides[4];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];
      // // align_corners=True
      // // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
      // float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      // float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      // float izf = ((iz + 1.f) / 2) * (inp_D - 1);
      //align_corners=False
      ix = ((ix + 1.f) *inp_W - 1.)/2.;
      iy = ((iy + 1.f) *inp_H - 1.)/2.;
      iz = ((iz + 1.f) *inp_D - 1.)/2.;

      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix = clip_coordinates(ix, inp_W);
        iy = clip_coordinates(iy, inp_H);
        iz = clip_coordinates(iz, inp_D);
      } else if (padding_mode == GridSamplerPadding::Reflection) {
        // reflect coordinates by image borders
        ix = reflect_coordinates(ix, -1, 2*inp_W-1);
        iy = reflect_coordinates(iy, -1, 2*inp_H-1);
        iz = reflect_coordinates(iz, -1, 2*inp_D-1);
      }
      ix=safe_downgrade_to_int_range(ix);
      iy=safe_downgrade_to_int_range(iy);
      iz=safe_downgrade_to_int_range(iz);
      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // ix = static_cast<scalar_t>(ix);
        // iy = static_cast<scalar_t>(iy);
        // iz = static_cast<scalar_t>(iz);

        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        int ix_tnw = static_cast<int>(::floor(ix));
        int iy_tnw = static_cast<int>(::floor(iy));
        int iz_tnw = static_cast<int>(::floor(iz));

        int ix_tne = ix_tnw + 1;
        int iy_tne = iy_tnw;
        int iz_tne = iz_tnw;

        int ix_tsw = ix_tnw;
        int iy_tsw = iy_tnw + 1;
        int iz_tsw = iz_tnw;

        int ix_tse = ix_tnw + 1;
        int iy_tse = iy_tnw + 1;
        int iz_tse = iz_tnw;

        int ix_bnw = ix_tnw;
        int iy_bnw = iy_tnw;
        int iz_bnw = iz_tnw + 1;

        int ix_bne = ix_tnw + 1;
        int iy_bne = iy_tnw;
        int iz_bne = iz_tnw + 1;

        int ix_bsw = ix_tnw;
        int iy_bsw = iy_tnw + 1;
        int iz_bsw = iz_tnw + 1;

        int ix_bse = ix_tnw + 1;
        int iy_bse = iy_tnw + 1;
        int iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
          // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
          // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
          // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
          }
        }
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));
        int iz_nearest = static_cast<int>(::round(iz));

        // assign nearest neighor pixel value to output pixel
        auto inp_ptr_NC = input.data + n * inp_sN;
        auto out_ptr_NCDHW = output.data + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
        for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
          if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
            *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
          } else {
            *out_ptr_NCDHW = static_cast<scalar_t>(0);
          }
        }
      }
    }
  }


  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_3d_backward_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> grad_output,
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_D = input.sizes[2];
    int inp_H = input.sizes[3];
    int inp_W = input.sizes[4];
    int out_D = grid.sizes[1];
    int out_H = grid.sizes[2];
    int out_W = grid.sizes[3];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sD = input.strides[2];
    int inp_sH = input.strides[3];
    int inp_sW = input.strides[4];
    int grid_sN = grid.strides[0];
    int grid_sD = grid.strides[1];
    int grid_sH = grid.strides[2];
    int grid_sW = grid.strides[3];
    int grid_sCoor = grid.strides[4];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sD = grad_output.strides[2];
    int gOut_sH = grad_output.strides[3];
    int gOut_sW = grad_output.strides[4];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sD = grad_input.strides[2];
    int gInp_sH = grad_input.strides[3];
    int gInp_sW = grad_input.strides[4];
    int gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
      // float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      // float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      // float izf = ((iz + 1.f) / 2) * (inp_D - 1);
      // align_corners=False
      ix = ((ix + 1.f) *inp_W - 1.)/2.;
      iy = ((iy + 1.f) *inp_H - 1.)/2.;
      iz = ((iz + 1.f) *inp_D - 1.)/2.;

      // multipliers for gradients on ix, iy, and iz
      // E.g.,  0 for out-of-bound indices when GridSamplerPadding::Border
      scalar_t gix_mult, giy_mult, giz_mult;
      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
        iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
        iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
      } else if (padding_mode == GridSamplerPadding::Reflection) {
        // reflect coordinates by image borders
        ix = reflect_coordinates_set_grad(ix, -1, 2*inp_W-1, &gix_mult);
        iy = reflect_coordinates_set_grad(iy, -1, 2*inp_H-1, &giy_mult);
        iz = reflect_coordinates_set_grad(iz, -1, 2*inp_D-1, &giz_mult);
      } else {  // padding_mode == GridSamplerPadding::Zeros
        gix_mult = static_cast<scalar_t>(1);
        giy_mult = static_cast<scalar_t>(1);
        giz_mult = static_cast<scalar_t>(1);
      }
      ix=safe_downgrade_to_int_range(ix);
      iy=safe_downgrade_to_int_range(iy);
      iz=safe_downgrade_to_int_range(iz);
      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // ix = static_cast<scalar_t>(ixf);
        // iy = static_cast<scalar_t>(iyf);
        // iz = static_cast<scalar_t>(izf);

        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        int ix_tnw = static_cast<int>(::floor(ix));
        int iy_tnw = static_cast<int>(::floor(iy));
        int iz_tnw = static_cast<int>(::floor(iz));

        int ix_tne = ix_tnw + 1;
        int iy_tne = iy_tnw;
        int iz_tne = iz_tnw;

        int ix_tsw = ix_tnw;
        int iy_tsw = iy_tnw + 1;
        int iz_tsw = iz_tnw;

        int ix_tse = ix_tnw + 1;
        int iy_tse = iy_tnw + 1;
        int iz_tse = iz_tnw;

        int ix_bnw = ix_tnw;
        int iy_bnw = iy_tnw;
        int iz_bnw = iz_tnw + 1;

        int ix_bne = ix_tnw + 1;
        int iy_bne = iy_tnw;
        int iz_bne = iz_tnw + 1;

        int ix_bsw = ix_tnw;
        int iy_bsw = iy_tnw + 1;
        int iz_bsw = iz_tnw + 1;

        int ix_bse = ix_tnw + 1;
        int iy_bse = iy_tnw + 1;
        int iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;
        // calculate bilinear weighted pixel value and set output pixel
        for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, inp_ptr_NC += inp_sC) {
          scalar_t gOut = *gOut_ptr_NCDHW;

          // calculate and set grad_input
          safe_add_3d(gInp_ptr_NC, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut);
          safe_add_3d(gInp_ptr_NC, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut);

          // calculate grad_grid
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
            gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
            giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
            giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
            gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
            giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
            giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
            gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
            giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
            giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
            gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
            giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
            giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
            gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
            giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
            giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
            gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
            giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
            giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
            gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
            giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
            giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
            gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
            giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
            giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
          }
        }

        // un-normalize grad_grid values back to [-1, 1] constraints
        gix = gix * static_cast<scalar_t>(inp_W)  / 2.;
        giy = giy * static_cast<scalar_t>(inp_H) / 2.;
        giz = giz * static_cast<scalar_t>(inp_D) / 2.;

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        int ix_nearest = static_cast<int>(::round(ix));
        int iy_nearest = static_cast<int>(::round(iy));
        int iz_nearest = static_cast<int>(::round(iz));

        // assign nearest neighor pixel value to output pixel
        scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
          // calculate and set grad_input
          safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
                      gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
      }
    }
  }


  template <typename scalar_t>
  __launch_bounds__(1024)
  __global__ void grid_sampler_3d_backward_backward_kernel(
      const int nthreads,
      TensorInfo<scalar_t, int> grad_output_input, // backward output derivative
      TensorInfo<scalar_t, int> grad_output_grid,
      TensorInfo<scalar_t, int> grad_output,  //forward output derivative
      TensorInfo<scalar_t, int> input,
      TensorInfo<scalar_t, int> grid,
      TensorInfo<scalar_t, int> grad_input,  // initialized to zeros
      TensorInfo<scalar_t, int> grad_grid,   // initialized to empty
      TensorInfo<scalar_t, int> grad_grad_output,   // initialized to zeros
      const GridSamplerInterpolation interpolation_mode,
      const GridSamplerPadding padding_mode) {

    int C = input.sizes[1];
    int inp_D = input.sizes[2];
    int inp_H = input.sizes[3];
    int inp_W = input.sizes[4];
    int out_D = grid.sizes[1];
    int out_H = grid.sizes[2];
    int out_W = grid.sizes[3];
    int inp_sN = input.strides[0];
    int inp_sC = input.strides[1];
    int inp_sD = input.strides[2];
    int inp_sH = input.strides[3];
    int inp_sW = input.strides[4];
    int grid_sN = grid.strides[0];
    int grid_sD = grid.strides[1];
    int grid_sH = grid.strides[2];
    int grid_sW = grid.strides[3];
    int grid_sCoor = grid.strides[4];
    int gOut_sN = grad_output.strides[0];
    int gOut_sC = grad_output.strides[1];
    int gOut_sD = grad_output.strides[2];
    int gOut_sH = grad_output.strides[3];
    int gOut_sW = grad_output.strides[4];
    int gGOut_sN = grad_grad_output.strides[0];
    int gGOut_sC = grad_grad_output.strides[1];
    int gGOut_sD = grad_grad_output.strides[2];
    int gGOut_sH = grad_grad_output.strides[3];
    int gGOut_sW = grad_grad_output.strides[4];
    int gOut_inp_sN = grad_output_input.strides[0];
    int gOut_inp_sC = grad_output_input.strides[1];
    int gOut_inp_sD = grad_output_input.strides[2];
    int gOut_inp_sH = grad_output_input.strides[3];
    int gOut_inp_sW = grad_output_input.strides[4];
    int gOut_grid_sN = grad_output_grid.strides[0];
    int gOut_grid_sC = grad_output_grid.strides[4];
    int gOut_grid_sD = grad_output_grid.strides[1];
    int gOut_grid_sH = grad_output_grid.strides[2];
    int gOut_grid_sW = grad_output_grid.strides[3];
    int gInp_sN = grad_input.strides[0];
    int gInp_sC = grad_input.strides[1];
    int gInp_sD = grad_input.strides[2];
    int gInp_sH = grad_input.strides[3];
    int gInp_sW = grad_input.strides[4];
    int gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP(index, nthreads) {
      const int w = index % out_W;
      const int h = (index / out_W) % out_H;
      const int d = (index / (out_H * out_W)) % out_D;
      const int n = index / (out_D * out_H * out_W);
      const int grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid.data[grid_offset];
      scalar_t iy = grid.data[grid_offset + grid_sCoor];
      scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

      // // normalize ix, iy, iz from [-1, 1] to [0, inp_W-1] & [0, inp_H-1] & [0, inp_D-1]
      // float ixf = ((ix + 1.f) / 2) * (inp_W - 1);
      // float iyf = ((iy + 1.f) / 2) * (inp_H - 1);
      // float izf = ((iz + 1.f) / 2) * (inp_D - 1);

      // align_corners=False
      ix = ((ix + 1.f) *inp_W - 1.)/2.;
      iy = ((iy + 1.f) *inp_H - 1.)/2.;
      iz = ((iz + 1.f) *inp_D - 1.)/2.;      

      // multipliers for gradients on ix, iy, and iz
      // E.g.,  0 for out-of-bound indices when GridSamplerPadding::Border
      scalar_t gix_mult, giy_mult, giz_mult;
      if (padding_mode == GridSamplerPadding::Border) {
        // clip coordinates to image borders
        ix = clip_coordinates_set_grad(ix, inp_W, &gix_mult);
        iy = clip_coordinates_set_grad(iy, inp_H, &giy_mult);
        iz = clip_coordinates_set_grad(iz, inp_D, &giz_mult);
      } else if (padding_mode == GridSamplerPadding::Reflection) {
        // reflect coordinates by image borders
        // to do: double backward relection mode need to correnct
        ix = reflect_coordinates_set_grad(ix, -1, 2*inp_W-1, &gix_mult);
        iy = reflect_coordinates_set_grad(iy, -1, 2*inp_H-1, &giy_mult);
        iz = reflect_coordinates_set_grad(iz, -1, 2*inp_D-1, &giz_mult);
      } else {  // padding_mode == GridSamplerPadding::Zeros
        gix_mult = static_cast<scalar_t>(1);
        giy_mult = static_cast<scalar_t>(1);
        giz_mult = static_cast<scalar_t>(1);
      }
      ix=safe_downgrade_to_int_range(ix);
      iy=safe_downgrade_to_int_range(iy);
      iz=safe_downgrade_to_int_range(iz);
      if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
        // ix = static_cast<scalar_t>(ixf);
        // iy = static_cast<scalar_t>(iyf);
        // iz = static_cast<scalar_t>(izf);

        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        int ix_0 = static_cast<int>(::floor(ix));
        int iy_0 = static_cast<int>(::floor(iy));
        int iz_0 = static_cast<int>(::floor(iz));

        int ix_1 = ix_0 + 1;
        int iy_1 = iy_0;
        int iz_1 = iz_0;

        int ix_2 = ix_0;
        int iy_2 = iy_0 + 1;
        int iz_2 = iz_0;

        int ix_3 = ix_0 + 1;
        int iy_3 = iy_0 + 1;
        int iz_3 = iz_0;

        int ix_4 = ix_0;
        int iy_4 = iy_0;
        int iz_4 = iz_0 + 1;

        int ix_5 = ix_0 + 1;
        int iy_5 = iy_0;
        int iz_5 = iz_0 + 1;

        int ix_6 = ix_0;
        int iy_6 = iy_0 + 1;
        int iz_6 = iz_0 + 1;

        int ix_7 = ix_0 + 1;
        int iy_7 = iy_0 + 1;
        int iz_7 = iz_0 + 1;

        // get surfaces to each neighbor:
        scalar_t w0 = (ix_7 - ix)    * (iy_7 - iy)    * (iz_7 - iz);
        scalar_t w1 = (ix    - ix_6) * (iy_6 - iy)    * (iz_6 - iz);
        scalar_t w2 = (ix_5 - ix)    * (iy    - iy_5) * (iz_5 - iz);
        scalar_t w3 = (ix    - ix_4) * (iy    - iy_4) * (iz_4 - iz);
        scalar_t w4 = (ix_3 - ix)    * (iy_3 - iy)    * (iz - iz_3);
        scalar_t w5 = (ix    - ix_2) * (iy_2 - iy)    * (iz - iz_2);
        scalar_t w6 = (ix_1 - ix)    * (iy    - iy_1) * (iz - iz_1);
        scalar_t w7 = (ix    - ix_0) * (iy    - iy_0) * (iz - iz_0);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        scalar_t *gOut_grid_ptr = grad_output_grid.data+n*gOut_grid_sN+d*gOut_grid_sD+h*gOut_grid_sH+w*gOut_grid_sW;
        scalar_t *gOut_inp_ptr_NC = grad_output_input.data+n*gOut_inp_sN;
        scalar_t *gGout_ptr_NCDHW = grad_grad_output.data+n*gGOut_sN+d*gGOut_sD+h*gGOut_sH+w*gGOut_sW;
        scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        scalar_t *inp_ptr_NC = input.data + n * inp_sN;

        scalar_t gOut_grid_x=*gOut_grid_ptr;
        scalar_t gOut_grid_y=*(gOut_grid_ptr+gOut_grid_sC);
        scalar_t gOut_grid_z=*(gOut_grid_ptr+2*gOut_grid_sC);

        scalar_t scale_x = 0.5*static_cast<scalar_t>(inp_W)*gix_mult;
        scalar_t scale_y = 0.5*static_cast<scalar_t>(inp_H)*giy_mult;
        scalar_t scale_z = 0.5*static_cast<scalar_t>(inp_D)*giz_mult;

        scalar_t scale_xy = scale_x*scale_y;
        scalar_t scale_xz = scale_x*scale_z;
        scalar_t scale_yz = scale_y*scale_z;

        scalar_t tmp0=-gOut_grid_x*scale_x*(iy_7-iy)*(iz_7-iz)-gOut_grid_y*scale_y*(ix_7-ix)*(iz_7-iz)-gOut_grid_z*scale_z*(ix_7-ix)*(iy_7-iy);
        scalar_t tmp1=gOut_grid_x*scale_x*(iy_6-iy)*(iz_6-iz)-gOut_grid_y*scale_y*(ix-ix_6)*(iz_6-iz)-gOut_grid_z*scale_z*(ix-ix_6)*(iy_6-iy);
        scalar_t tmp2=-gOut_grid_x*scale_x*(iy-iy_5)*(iz_5-iz)+gOut_grid_y*scale_y*(ix_5-ix)*(iz_5-iz)-gOut_grid_z*scale_z*(ix_5-ix)*(iy-iy_5);
        scalar_t tmp3=gOut_grid_x*scale_x*(iy-iy_4)*(iz_4-iz)+gOut_grid_y*scale_y*(ix-ix_4)*(iz_4-iz)-gOut_grid_z*scale_z*(ix-ix_4)*(iy-iy_4);
        scalar_t tmp4=-gOut_grid_x*scale_x*(iy_3-iy)*(iz-iz_3)-gOut_grid_y*scale_y*(ix_3-ix)*(iz-iz_3)+gOut_grid_z*scale_z*(ix_3-ix)*(iy_3-iy);
        scalar_t tmp5=gOut_grid_x*scale_x*(iy_2-iy)*(iz-iz_2)-gOut_grid_y*scale_y*(ix-ix_2)*(iz-iz_2)+gOut_grid_z*scale_z*(ix-ix_2)*(iy_2-iy);
        scalar_t tmp6=-gOut_grid_x*scale_x*(iy-iy_1)*(iz-iz_1)+gOut_grid_y*scale_y*(ix_1-ix)*(iz-iz_1)+gOut_grid_z*scale_z*(ix_1-ix)*(iy-iy_1);
        scalar_t tmp7=gOut_grid_x*scale_x*(iy-iy_0)*(iz-iz_0)+gOut_grid_y*scale_y*(ix-ix_0)*(iz-iz_0)+gOut_grid_z*scale_z*(ix-ix_0)*(iy-iy_0);
        // calculate bilinear weighted pixel value and set output pixel
        for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC, gGout_ptr_NCDHW += gGOut_sC, inp_ptr_NC += inp_sC, gOut_inp_ptr_NC += gOut_inp_sC) {
          scalar_t gOut = *gOut_ptr_NCDHW;
          

          // calculate and set grad_input
          safe_add_3d(gInp_ptr_NC, iz_0, iy_0, ix_0, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp0 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_1, iy_1, ix_1, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp1 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_2, iy_2, ix_2, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp2 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_3, iy_3, ix_3, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp3 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_4, iy_4, ix_4, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp4 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_5, iy_5, ix_5, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp5 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_6, iy_6, ix_6, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp6 * gOut);
          safe_add_3d(gInp_ptr_NC, iz_7, iy_7, ix_7, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tmp7 * gOut);      

          // calculate grad_grid
          if (within_bounds_3d(iz_0, iy_0, ix_0, inp_D, inp_H, inp_W)) {
            scalar_t t0_val = gOut_inp_ptr_NC[iz_0*gOut_inp_sD+iy_0*gOut_inp_sH+ix_0*gOut_inp_sW];
            gix -= t0_val * (iy_7 - iy)    * (iz_7 - iz)    * gOut*scale_x;
            giy -= t0_val * (ix_7 - ix)    * (iz_7 - iz)    * gOut*scale_y;
            giz -= t0_val * (ix_7 - ix)    * (iy_7 - iy)    * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t0_val * w0;
            t0_val = inp_ptr_NC[iz_0*inp_sD+iy_0*inp_sH+ix_0*inp_sW];
            gix += t0_val * (gOut_grid_y*(iz_7-iz)*scale_xy+gOut_grid_z*(iy_7-iy)*scale_xz)* gOut;
            giy += t0_val * (gOut_grid_x*(iz_7-iz)*scale_xy+gOut_grid_z*(ix_7-ix)*scale_yz)* gOut;
            giz += t0_val * (gOut_grid_x*(iy_7-iy)*scale_xz+gOut_grid_y*(ix_7-ix)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t0_val * tmp0;
          }
          if (within_bounds_3d(iz_1, iy_1, ix_1, inp_D, inp_H, inp_W)) {
            scalar_t t1_val = gOut_inp_ptr_NC[iz_1*gOut_inp_sD+iy_1*gOut_inp_sH+ix_1*gOut_inp_sW];
            gix += t1_val * (iy_6 - iy)    * (iz_6 - iz)    * gOut*scale_x;
            giy -= t1_val * (ix    - ix_6) * (iz_6 - iz)    * gOut*scale_y;
            giz -= t1_val * (ix    - ix_6) * (iy_6 - iy)    * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t1_val * w1;
            t1_val = inp_ptr_NC[iz_1*inp_sD+iy_1*inp_sH+ix_1*inp_sW];
            gix += t1_val * (gOut_grid_y*(iz-iz_6)*scale_xy+gOut_grid_z*(iy-iy_6)*scale_xz)* gOut;
            giy += t1_val * (gOut_grid_x*(iz-iz_6)*scale_xy+gOut_grid_z*(ix-ix_6)*scale_yz)* gOut;
            giz += t1_val * (gOut_grid_x*(iy-iy_6)*scale_xz+gOut_grid_y*(ix-ix_6)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t1_val * tmp1;
          }
          if (within_bounds_3d(iz_2, iy_2, ix_2, inp_D, inp_H, inp_W)) {
            scalar_t t2_val = gOut_inp_ptr_NC[iz_2*gOut_inp_sD+iy_2*gOut_inp_sH+ix_2*gOut_inp_sW];
            gix -= t2_val * (iy - iy_5)    * (iz_5 - iz)    * gOut*scale_x;
            giy += t2_val * (ix_5 - ix)    * (iz_5 - iz)    * gOut*scale_y;
            giz -= t2_val * (ix_5 - ix)    * (iy    - iy_5) * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t2_val * w2;
            t2_val = inp_ptr_NC[iz_2*inp_sD+iy_2*inp_sH+ix_2*inp_sW];
            gix += t2_val * (gOut_grid_y*(iz-iz_5)*scale_xy+gOut_grid_z*(iy-iy_5)*scale_xz)* gOut;
            giy += t2_val * (gOut_grid_x*(iz-iz_5)*scale_xy+gOut_grid_z*(ix-ix_5)*scale_yz)* gOut;
            giz += t2_val * (gOut_grid_x*(iy-iy_5)*scale_xz+gOut_grid_y*(ix-ix_5)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t2_val * tmp2;
          }
          if (within_bounds_3d(iz_3, iy_3, ix_3, inp_D, inp_H, inp_W)) {
            scalar_t t3_val = gOut_inp_ptr_NC[iz_3*gOut_inp_sD+iy_3*gOut_inp_sH+ix_3*gOut_inp_sW];
            gix += t3_val * (iy - iy_4)    * (iz_4 - iz)    * gOut*scale_x;
            giy += t3_val * (ix - ix_4) * (iz_4 - iz)    * gOut*scale_y;
            giz -= t3_val * (ix - ix_4) * (iy - iy_4) * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t3_val * w3;
            t3_val = inp_ptr_NC[iz_3*inp_sD+iy_3*inp_sH+ix_3*inp_sW];
            gix += t3_val * (gOut_grid_y*(iz_4-iz)*scale_xy+gOut_grid_z*(iy_4-iy)*scale_xz)* gOut;
            giy += t3_val * (gOut_grid_x*(iz_4-iz)*scale_xy+gOut_grid_z*(ix_4-ix)*scale_yz)* gOut;
            giz += t3_val * (gOut_grid_x*(iy_4-iy)*scale_xz+gOut_grid_y*(ix_4-ix)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t3_val * tmp3;
          }
          if (within_bounds_3d(iz_4, iy_4, ix_4, inp_D, inp_H, inp_W)) {
            scalar_t t4_val = gOut_inp_ptr_NC[iz_4*gOut_inp_sD+iy_4*gOut_inp_sH+ix_4*gOut_inp_sW];
            gix -= t4_val * (iy_3 - iy)    * (iz - iz_3)    * gOut*scale_x;
            giy -= t4_val * (ix_3 - ix)    * (iz - iz_3)    * gOut*scale_y;
            giz += t4_val * (ix_3 - ix)    * (iy_3 - iy)    * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t4_val * w4;
            t4_val = inp_ptr_NC[iz_4*inp_sD+iy_4*inp_sH+ix_4*inp_sW];
            gix += t4_val * (gOut_grid_y*(iz-iz_3)*scale_xy+gOut_grid_z*(iy-iy_3)*scale_xz)* gOut;
            giy += t4_val * (gOut_grid_x*(iz-iz_3)*scale_xy+gOut_grid_z*(ix-ix_3)*scale_yz)* gOut;
            giz += t4_val * (gOut_grid_x*(iy-iy_3)*scale_xz+gOut_grid_y*(ix-ix_3)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t4_val * tmp4;
          }
          if (within_bounds_3d(iz_5, iy_5, ix_5, inp_D, inp_H, inp_W)) {
            scalar_t t5_val = gOut_inp_ptr_NC[iz_5*gOut_inp_sD+iy_5*gOut_inp_sH+ix_5*gOut_inp_sW];
            gix += t5_val * (iy_2 - iy)    * (iz - iz_2)    * gOut*scale_x;
            giy -= t5_val * (ix    - ix_2) * (iz - iz_2)    * gOut*scale_y;
            giz += t5_val * (ix    - ix_2) * (iy_2 - iy)    * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t5_val * w5;
            t5_val = inp_ptr_NC[iz_5*inp_sD+iy_5*inp_sH+ix_5*inp_sW];
            gix += t5_val * (gOut_grid_y*(iz_2-iz)*scale_xy+gOut_grid_z*(iy_2-iy)*scale_xz)* gOut;
            giy += t5_val * (gOut_grid_x*(iz_2-iz)*scale_xy+gOut_grid_z*(ix_2-ix)*scale_yz)* gOut;
            giz += t5_val * (gOut_grid_x*(iy_2-iy)*scale_xz+gOut_grid_y*(ix_2-ix)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t5_val * tmp5;
          }
          if (within_bounds_3d(iz_6, iy_6, ix_6, inp_D, inp_H, inp_W)) {
            scalar_t t6_val = gOut_inp_ptr_NC[iz_6*gOut_inp_sD+iy_6*gOut_inp_sH+ix_6*gOut_inp_sW];
            gix -= t6_val * (iy - iy_1)    * (iz - iz_1)    * gOut*scale_x;
            giy += t6_val * (ix_1 - ix)    * (iz - iz_1)    * gOut*scale_y;
            giz += t6_val * (ix_1 - ix)    * (iy    - iy_1) * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t6_val * w6;
            t6_val = inp_ptr_NC[iz_6*inp_sD+iy_6*inp_sH+ix_6*inp_sW];
            gix += t6_val * (gOut_grid_y*(iz_1-iz)*scale_xy+gOut_grid_z*(iy_1-iy)*scale_xz)* gOut;
            giy += t6_val * (gOut_grid_x*(iz_1-iz)*scale_xy+gOut_grid_z*(ix_1-ix)*scale_yz)* gOut;
            giz += t6_val * (gOut_grid_x*(iy_1-iy)*scale_xz+gOut_grid_y*(ix_1-ix)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t6_val * tmp6;
          }
          if (within_bounds_3d(iz_7, iy_7, ix_7, inp_D, inp_H, inp_W)) {
            scalar_t t7_val = gOut_inp_ptr_NC[iz_7*gOut_inp_sD+iy_7*gOut_inp_sH+ix_7*gOut_inp_sW];
            gix += t7_val * (iy - iy_0)    * (iz - iz_0)    * gOut*scale_x;
            giy += t7_val * (ix - ix_0) * (iz - iz_0)    * gOut*scale_y;
            giz += t7_val * (ix - ix_0) * (iy - iy_0) * gOut*scale_z;
            gGout_ptr_NCDHW[0]+=t7_val * w7;
            t7_val = inp_ptr_NC[iz_7*inp_sD+iy_7*inp_sH+ix_7*inp_sW];
            gix += t7_val * (gOut_grid_y*(iz-iz_0)*scale_xy+gOut_grid_z*(iy-iy_0)*scale_xz)* gOut;
            giy += t7_val * (gOut_grid_x*(iz-iz_0)*scale_xy+gOut_grid_z*(ix-ix_0)*scale_yz)* gOut;
            giz += t7_val * (gOut_grid_x*(iy-iy_0)*scale_xz+gOut_grid_y*(ix-ix_0)*scale_yz)* gOut;
            gGout_ptr_NCDHW[0]+=t7_val * tmp7;
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = gix;
        gGrid_ptr_NDHW[1] = giy;
        gGrid_ptr_NDHW[2] = giz;
      } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
        // to do: implement the nearest mode computation, it is similar with bilinear mode, basically one node situation
        // int ix_n = static_cast<int>(::round(ixf));
        // int iy_n = static_cast<int>(::round(iyf));
        // int iz_n = static_cast<int>(::round(izf));

        // scalar_t gOut_grid_x=*gOut_grid_ptr;
        // scalar_t gOut_grid_y=*(gOut_grid_ptr+gOut_grid_sC);
        // scalar_t gOut_grid_z=*(gOut_grid_ptr+2*gOut_grid_sC);

        // scalar_t scale_x = 0.5*(inp_W - 1);
        // scalar_t scale_y = 0.5*(inp_H - 1);
        // scalar_t scale_z = 0.5*(inp_D - 1);

        // scalar_t scale_xy = scale_x*scale_y;
        // scalar_t scale_xz = scale_x*scale_z;
        // scalar_t scale_yz = scale_y*scale_z;

        // scalar_t tmp=-gOut_grid_x*scale_x*(iy_7-iy)*(iz_7-iz)-gOut_grid_y*scale_y*(ix_7-ix)*(iz_7-iz)-gOut_grid_z*scale_z*(ix_7-ix)*(iy_7-iy);

        // // assign nearest neighor pixel value to output pixel
        // scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        // scalar_t *gInp_ptr_NC = grad_input.data + n * gInp_sN;
        // for (int c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, gInp_ptr_NC += gInp_sC) {
        //   // calculate and set grad_input
        //   safe_add_3d(gInp_ptr_NC, iz_nearest, iy_nearest, ix_nearest,
        //               gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW);
        // }

        // // assuming grad_grid is contiguous
        // // thus we can
        // //   1. use index with gGrid_sW to diectly compute gGrid_ptr_NDHW
        // //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        // scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
        // gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
        // gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
        // gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
      }
    }
  }}



// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
Tensor grid_sampler_3d_cuda_mine(const Tensor& input, const Tensor& grid,
                            int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto output = at::empty({N, input.size(1), D, H, W}, input.options());
  int count = static_cast<int>(N * D * H * W);
  // if(at::any(at::isnan(input)).item().to<bool>()) std::cerr<<"cudaf0_0"<<std::endl;
  // if(at::any(at::isnan(grid)).item().to<bool>()) std::cerr<<"cudaf0_1"<<std::endl;
  // CUDA_POST_KERNEL_CHECK;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_3d_cuda", [&] {
      grid_sampler_3d_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(output),
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode));
    });
  }
  // if(at::any(at::isnan(output)).item().to<bool>()) std::cerr<<"cudaf1_0"<<std::endl;
  // CUDA_POST_KERNEL_CHECK;
  return output;
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor>
grid_sampler_3d_backward_cuda_mine(const Tensor& input, const Tensor& grid,const Tensor& grad_output, 
                              int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);
  int count = static_cast<int>(N * D * H * W);
  // if(at::any(at::isnan(input)).item().to<bool>()) std::cerr<<"cudab0_0"<<std::endl;
  // if(at::any(at::isnan(grid)).item().to<bool>()) std::cerr<<"cudab0_1"<<std::endl;
  // if(at::any(at::isnan(grad_output)).item().to<bool>()) std::cerr<<"cudab0_2"<<std::endl;
  // CUDA_POST_KERNEL_CHECK;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_3d_backward_cuda", [&] {
      grid_sampler_3d_backward_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(grad_output),
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(grad_input),
          getTensorInfo<scalar_t, int>(grad_grid),
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode));
    });
  }
  // CUDA_POST_KERNEL_CHECK;
  // if(at::any(at::isnan(grad_input)).item().to<bool>()) std::cerr<<"cudab1_0"<<std::endl;
  // if(at::any(at::isnan(grad_grid)).item().to<bool>()) std::cerr<<"cudab1_1"<<std::endl;
  return std::make_tuple(grad_input, grad_grid);
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
std::tuple<Tensor, Tensor, Tensor>
grid_sampler_3d_backward_backward_cuda_mine(const Tensor& grad_output_input, const Tensor& grad_output_grid, const Tensor& input, const Tensor& grid, const Tensor& grad_output,
                              int64_t interpolation_mode, int64_t padding_mode) {
  auto N = input.size(0);
  auto D = grid.size(1);
  auto H = grid.size(2);
  auto W = grid.size(3);
  auto grad_input = at::zeros_like(input);
  auto grad_grid = at::empty_like(grid);
  auto grad_grad_output = at::zeros_like(grad_output);
  int count = static_cast<int>(N * D * H * W);
  // if(at::any(at::isnan(input)).item().to<bool>()) std::cerr<<"cudad0_0"<<std::endl;
  // if(at::any(at::isnan(grid)).item().to<bool>()) std::cerr<<"cudad0_1"<<std::endl;
  // if(at::any(at::isnan(grad_output)).item().to<bool>()) std::cerr<<"cudad0_2"<<std::endl;
  // if(at::any(at::isnan(grad_output_input)).item().to<bool>()) std::cerr<<"cudad0_3"<<std::endl;
  // if(at::any(at::isnan(grad_output_grid)).item().to<bool>()) std::cerr<<"cudad0_4"<<std::endl;
  // CUDA_POST_KERNEL_CHECK;
  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "grid_sampler_3d_backward_backward_cuda", [&] {
      grid_sampler_3d_backward_backward_kernel<scalar_t>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          count,
          getTensorInfo<scalar_t, int>(grad_output_input),
          getTensorInfo<scalar_t, int>(grad_output_grid),
          getTensorInfo<scalar_t, int>(grad_output),
          getTensorInfo<scalar_t, int>(input),
          getTensorInfo<scalar_t, int>(grid),
          getTensorInfo<scalar_t, int>(grad_input),
          getTensorInfo<scalar_t, int>(grad_grid),
          getTensorInfo<scalar_t, int>(grad_grad_output),
          static_cast<GridSamplerInterpolation>(interpolation_mode),
          static_cast<GridSamplerPadding>(padding_mode));
    });
  }
  // CUDA_POST_KERNEL_CHECK;
  // if(at::any(at::isnan(grad_input)).item().to<bool>()) std::cerr<<"cudad1_0"<<std::endl;
  // if(at::any(at::isnan(grad_grid)).item().to<bool>()) std::cerr<<"cudad1_1"<<std::endl;
  // if(at::any(at::isnan(grad_grad_output)).item().to<bool>()) std::cerr<<"cudad1_2"<<std::endl;
  return std::make_tuple(grad_input, grad_grid,grad_grad_output);
}
}}