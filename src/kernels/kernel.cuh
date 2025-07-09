#ifndef GC_MICROBENCHMARK_SINGLE_GPU_KERNEL_CUH
#define GC_MICROBENCHMARK_SINGLE_GPU_KERNEL_CUH

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
/*
mma.sync.aligned.shape.row.col.f64.f64.f64.f64 d, a, b, c;

.shape   = {.m16n8k16};
*/
namespace ftxj
{
#define MINIBATCH 1
#define UNROLL 8
#ifdef GC22
#define TILE_DIM 64
#elif FLOATMAT
#define TILE_DIM 64
#elif TILE64
#define TILE_DIM 64
#else
#define TILE_DIM 128
#endif
#define TC_SIZE 16 // tensor core size
#define BLOCK_ROWS 8
#define WARP_SIZE 32
#define THREAD_COPY_BYTES 16
#define DMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RA4, RA5, RA6, RA7, RB0, RB1, RB2, RB3, RC0, RC1, RC2, RC3)                                                  \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, {%16, %17, %18, %19};\n" \
               : "=l"(RD0), "=l"(RD1), "=l"(RD2), "=l"(RD3)                                                                                                            \
               : "l"(RA0), "l"(RA1), "l"(RA2), "l"(RA3), "l"(RA4), "l"(RA5), "l"(RA6), "l"(RA7),                                                                       \
                 "l"(RB0), "l"(RB1), "l"(RB2), "l"(RB3),                                                                                                               \
                 "l"(RC0), "l"(RC1), "l"(RC2), "l"(RC3))

#define DMMA1688(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                  \
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
               : "=l"(RD0), "=l"(RD1), "=l"(RD2), "=l"(RD3)                                                                             \
               : "l"(RA0), "l"(RA1), "l"(RA2), "l"(RA3),                                                                                \
                 "l"(RB0), "l"(RB1),                                                                                                    \
                 "l"(RC0), "l"(RC1), "l"(RC2), "l"(RC3))

#define DMMA1684(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RC0, RC1, RC2, RC3)                                                  \
  asm volatile("mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n" \
               : "=l"(RD0), "=l"(RD1), "=l"(RD2), "=l"(RD3)                                                              \
               : "l"(RA0), "l"(RA1),                                                                                     \
                 "l"(RB0),                                                                                               \
                 "l"(RC0), "l"(RC1), "l"(RC2), "l"(RC3))

#define DMMA884(RD0, RD1, RA0, RB0, RC0, RC1)                                                      \
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%4, %5};\n" \
               : "=l"(RD0), "=l"(RD1)                                                              \
               : "l"(RA0),                                                                         \
                 "l"(RB0),                                                                         \
                 "l"(RC0), "l"(RC1))
#define HMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                \
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1,%2,%3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
               : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                                            \
               : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3),                                                                               \
                 "r"(RB0), "r"(RB1),                                                                                                   \
                 "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
#define HMMA1688(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RC0, RC1, RC2, RC3)                                                \
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1,%2,%3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n" \
               : "=f"(RD0), "=f"(RD1), "=f"(RD2), "=f"(RD3)                                                            \
               : "r"(RA0), "r"(RA1),                                                                                   \
                 "r"(RB0),                                                                                             \
                 "f"(RC0), "f"(RC1), "f"(RC2), "f"(RC3))
#define LDMATRIX_X1(R, addr) \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2(R0, R1, addr) \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                           \
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
               : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
               : "r"(addr))

#define LDMATRIX_X1_T(R, addr) \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" : "=r"(R) : "r"(addr))

#define LDMATRIX_X2_T(R0, R1, addr) \
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))

#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                               \
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
               : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                   \
               : "r"(addr))

#define CP_ASYNC_CG(dst, src, Bytes) \
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))
#define CP_ASYNC_CA(dst, src, Bytes)                  \
  asm("cp.async.ca.shared.global [%0], [%1], %2;\n" : \
      : "r"(dst), "l"(src), "n"(Bytes));
#define __THALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))

#define F32_TO_REG64(dst, src) \
  asm("{.reg .f64 temp;\n"     \
      "cvt.f64.f32 temp,%1;\n" \
      "mov.b64 %0,temp;\n}" : "=l"(dst) : "r"(src))
#define FP32_TO_REG64(dst, src) \
  asm("{.reg .f64 temp;\n"      \
      "cvt.f64.f32 temp,%1;\n"  \
      "mov.b64 %0,temp;\n}" : "=l"(dst) : "f"(src))
#define FP16_TO_REG64(dst, src) \
  asm("{.reg .f64 temp;\n"      \
      "cvt.f64.f16 temp,%1;\n"  \
      "mov.b64 %0,temp;\n}" : "=l"(dst) : "h"(__THALF_TO_CUS(src)))

#define DOUBLE_TO_REG64(dst, src) \
  asm("mov.b64 %0,%1;" : "=l"(dst) : "d"(src))

#define REG64_TO_DOUBLE(dst, src) \
  asm("mov.b64 %0,%1;" : "=d"(dst) : "l"(src))

#define FP32_TO_DOUBLE(dst, src) \
  asm("cvt.f64.f32 %0,%1;" : "=d"(dst) : "f"(src))

#define DOUBLE_TO_FP32(dst, src) \
  asm("cvt.f32.f64 %0,%1;" : "=f"(dst) : "d"(src))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

#define FP16_TO_DOUBLE(dst, src) \
  asm("cvt.f64.f16 %0,%1;" : "=d"(dst) : "h"(__THALF_TO_CUS(src)))

  __device__ inline float __ReLU(float x)
  {
    return x < 0.0 ? 0.0 : x > 32.0 ? 32.0
                                    : x;
  };

  __global__ void check_active_h(half *__restrict__ mat, int *__restrict__ active,
                                 int neuron, int batch);

  // row major
  __global__ void check_active_f_row(float *__restrict__ mat, int *__restrict__ active,
                                     int neuron, int batch);

  // col major
  __global__ void check_active_f_col(float *__restrict__ mat, int *__restrict__ active,
                                     int neuron, int batch);

  __global__ void matrix_transpose(float *__restrict__ odata, float *__restrict__ idata,
                                   int neuron, int batch);

  __global__ void matrix_re_transpose_and_delete(float *__restrict__ odata, float *__restrict__ idata,
                                                 int *__restrict__ old_to_new_map, int neuron, int batch);

  __global__ void matrix_reshape(float *__restrict__ odata,
                                 float *__restrict__ idata,
                                 int *__restrict__ category,
                                 int neuron,
                                 int odata_batch,
                                 int idata_batch,
                                 int add_padding_batch);
  __global__ void floatTohalf2(float *__restrict__ fmat,
                               half *__restrict__ hmat,
                               int neuron, int batch);
  // add bias and relu
  __global__ void add_bias_relu_h(half *mat_higher, half *mat_lower, half bias,
                                  int neuron, int batch, int *active);

  // tensor core
  // type covertion
  __global__ void matrix_float2half(float *__restrict__ matrix, half *__restrict__ mat_higher,
                                    half *__restrict__ mat_lower, int mat_size);

  __global__ void matrix_half2float(half *__restrict__ mat_higher, half *__restrict__ mat_lower,
                                    float *__restrict__ matrix, int mat_size);

  void element_float2half(float &element, half &ele_higher, half &ele_lower);

  __global__ void cospdnn_single(half *__restrict__ A,
                                                               half *__restrict__ B,
                                                               float *__restrict__ C,
                                                               int *__restrict__ fuse,
                                                               int *__restrict__ idx_ptr,
                                                               int *__restrict__ index,
                                                               float bias,
                                                               int batch,
                                                               int neuron,
                                                               int blocksize_c,
                                                               int blocksize_r);
  __global__ void cospdnn_double(half *__restrict__ A1,
                                                        half *__restrict__ B1,
                                                        float *__restrict__ C1,
                                                        int *__restrict__ fuse1,
                                                        int *__restrict__ idx_ptr1,
                                                        int *__restrict__ index1,
                                                        int dim1,
                                                        float bias1,
                                                        int batch1,
                                                        int neuron1,
                                                        half *__restrict__ A2,
                                                        half *__restrict__ B2,
                                                        float *__restrict__ C2,
                                                        int *__restrict__ fuse2,
                                                        int *__restrict__ idx_ptr2,
                                                        int *__restrict__ index2,
                                                        int dim2,
                                                        float bias2,
                                                        int batch2,
                                                        int neuron2,
                                                        int blocksize_c,
                                                        int blocksize_r,
                                                        int boundary);
};

#endif