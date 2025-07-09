#include "./kernel.cuh"
#include <cstdio>
#include <mma.h>
#include <stdio.h>
namespace ftxj
{  __host__ __device__ dim3
  get_swizzled_data_block_idx(const int gridDim_x,
                              const int gridDim_y,
                              const int blockIdx_x,
                              const int blockIdx_y,
                              const int TILE_WIDTH)
  {
    const int blocks_per_tile = gridDim_y * TILE_WIDTH;
    const int num_tiles = gridDim_x / TILE_WIDTH;
    const int block_idx_flatterned = blockIdx_y * gridDim_x + blockIdx_x;
    const int tile_id = block_idx_flatterned / blocks_per_tile;
    int block_idx_in_tile = block_idx_flatterned % blocks_per_tile;
    int block_idx_x_in_tile = block_idx_in_tile % TILE_WIDTH;
    int block_idx_y_in_tile = block_idx_in_tile / TILE_WIDTH;
    if (blockIdx_x >= num_tiles * TILE_WIDTH)
    {
      const int last_tile_dim_x = gridDim_x - num_tiles * TILE_WIDTH;
      block_idx_x_in_tile = block_idx_in_tile % last_tile_dim_x;
      block_idx_y_in_tile = block_idx_in_tile / last_tile_dim_x;
    }
    const int swizzled_block_idx_flatterned =
        block_idx_y_in_tile * gridDim_x + block_idx_x_in_tile + tile_id * TILE_WIDTH;
    const int swizzled_block_idx_x = swizzled_block_idx_flatterned % gridDim_x;
    const int swizzled_block_idx_y = swizzled_block_idx_flatterned / gridDim_x;

    return dim3(swizzled_block_idx_x, swizzled_block_idx_y, 1);
  }
  template <uint32_t S, uint32_t B, uint32_t M>
  __device__ __forceinline__ uint32_t swizzle(uint32_t addr)
  {
    constexpr auto Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
  }
  __device__ inline void loadSmem4(void *smem, void *gmem)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    float4 *gmem_addr = (float4 *)gmem;
    CP_ASYNC_CA(smem_addr, gmem_addr, 16);
  }
  __device__ inline void loadSmem2(void *smem, void *gmem)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    float2 *gmem_addr = (float2 *)gmem;
    CP_ASYNC_CA(smem_addr, gmem_addr, 8);
  }
  __device__ inline void loadSmem1(void *smem, void *gmem)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    float *gmem_addr = (float *)gmem;
    CP_ASYNC_CA(smem_addr, gmem_addr, 4);
  }
  __device__ inline void loadSmem(void *smem, void *gmem)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    float4 *gmem_addr = (float4 *)gmem;
    CP_ASYNC_CG(smem_addr, gmem_addr, THREAD_COPY_BYTES);
  }

  __device__ inline void loadSmem(void *smem, void *gmem, bool flag)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    float4 *gmem_addr = (float4 *)gmem;
    if (flag)
      CP_ASYNC_CG(smem_addr, gmem_addr, THREAD_COPY_BYTES);
  }

  __device__ inline void loadFragX1(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X1(*Reg, smem_addr);
  }
  __device__ inline void loadFragX1_T(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X1_T(*Reg, smem_addr);
  }
  __device__ inline void loadFragX2(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X2(Reg[0], Reg[1], smem_addr);
  }
  __device__ inline void loadFragX2_T(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X2_T(Reg[0], Reg[1], smem_addr);
  }
  __device__ inline void loadFragX4(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X4(Reg[0], Reg[1], Reg[2], Reg[3], smem_addr);
  }

  __device__ inline void loadFragX4_T(void *smem, uint32_t *Reg)
  {
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    LDMATRIX_X4_T(Reg[0], Reg[1], Reg[2], Reg[3], smem_addr);
  }
  // general
  __global__ void check_active_h(half *__restrict__ mat, int *__restrict__ active,
                                 int neuron, int batch)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < neuron * batch && __hne(mat[idx], 0))
    {
      active[idx / neuron] = 1;
    }

    return;
  }

  // row_major
  __global__ void check_active_f_row(float *__restrict__ mat, int *__restrict__ active,
                                     int neuron, int batch)
  {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (uint64_t)neuron * batch && mat[idx] != 0)
    {
      active[idx / neuron] += 1;
    }
    return;
  }

  // col_major
  __global__ void check_active_f_col(float *__restrict__ mat, int *__restrict__ active,
                                     int neuron, int batch)
  {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (uint64_t)neuron * batch && mat[idx] - 0 > 1e-5)
    {
      active[idx % batch] += 1;
    }
    __syncthreads();
    return;
  }
  __global__ void floatTohalf2(float *__restrict__ fmat,
                               half *__restrict__ hmat,
                               int neuron, int batch)
  {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (uint64_t)neuron * batch)
    {
      hmat[idx] = __float2half(fmat[idx]);
      hmat[idx + (uint64_t)neuron * batch] = __float2half(fmat[idx] - __half2float(hmat[idx]));
    }
    __syncthreads();
    return;
  }
  // pay attention: the mat is transposed
  // odata: the old data
  // idata: the new data
  __global__ void matrix_reshape(float *__restrict__ odata,
                                 float *__restrict__ idata,
                                 int *__restrict__ category,
                                 int neuron,
                                 int odata_batch,
                                 int idata_batch,
                                 int add_padding_batch)
  {
    uint64_t idx = (uint64_t)blockDim.x * blockIdx.x + threadIdx.x;
    int col_id = idx % idata_batch;
    int row_id = idx / idata_batch;
    if (idx < (uint64_t)neuron * idata_batch)
    {
      idata[(uint64_t)row_id * add_padding_batch + col_id] = odata[(uint64_t)row_id * odata_batch + category[col_id]];
    }
    return;
  }

  // add bias and relu
  __global__ void add_bias_relu_h(half *mat_higher, half *mat_lower, half bias,
                                  int neuron, int batch, int *active)
  {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < neuron * batch && __hne(mat_higher[idx], 0))
    {
      float res = __ReLU(__half2float(mat_higher[idx]));
      if (res != 0)
      {
        active[idx / neuron] = 1;
        mat_higher[idx] = __float2half(res);
      }
      else
      {
        mat_higher = 0;
        mat_lower = 0;
      }
    }
    return;
  }

  // tensor core

  // type covertion
  __global__ void matrix_float2half(float *__restrict__ matrix, half *mat_higher,
                                    half *mat_lower, int mat_size)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < mat_size)
    {
      mat_higher[id] = __float2half(matrix[id]);
      float residual = matrix[id] - __half2float(mat_higher[id]);
      mat_lower[id] = __float2half(mat_lower[id]);
    }
    return;
  }

  __global__ void matrix_half2float(half *__restrict__ mat_higher, half *__restrict__ mat_lower,
                                    float *__restrict__ matrix, int mat_size)
  {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < mat_size)
    {
      matrix[id] = __half2float(mat_higher[id]) + __half2float(mat_lower[id]);
    }
    return;
  }

  void element_float2half(float &element, half &ele_higher, half &ele_lower)
  {
    ele_higher = __float2half(element);
    float residual = element - __half2float(ele_higher);
    ele_lower = __float2half(residual);
    return;
  }

  // Only tensor core
// One x warp processes one sparse column, one y thread handles minibatch * 16 rows
// Includes pipeline
// Includes swizzle
// TILE_M=128, TILE_N=16
// Performance similar to previous version, but warp_stall_long_scoreboard decreased
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
                                                           int blocksize_r)
{
  int lane_id = threadIdx.x % WARP_SIZE; // Thread ID within warp
  int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
  // Start row calculation
  int tidy = blockIdx.y;             // Tile block ID for batch dimension
  int bid = blockIdx.x;              // Block ID for neuron dimension
  int block_pos = idx_ptr[bid];      // Starting row index in neuron dimension (X matrix (weight) stored in transposed BCSR format)
  int A_start_row = tidy * TILE_DIM; // Starting row in batch dimension based on tile ID
  int A_fused_col_idx;               // Fused column index
  /*
  A_Smem[PIPELINE_STAGE][HALF2 * BLOCK_ROW_NUMS * COLUMNS * ROWS]
  PIPELINE_STAGE: Pipeline stages
  HALF2: Original float matrix A is split into two half groups (HALF2=2)
  First 1024 elements store high bits, last 1024 store low bits
  BLOCK_ROW_NUMS: Number of warps processing row dimension within block
  */
  __shared__ half A_Smem[2][2 * 8 * 16 * 8];
  /*
   B_Smem[PIPELINE_STAGE][WARP_COLUMN_NUMS * COLUMNS * ROWS]
   BLOCK_ROW_NUMS: Warp column blocks
  */
  __shared__ half B_Smem[2][2 * 8 * 8]; 

  float RC[2][4];  // RC[WARP_COLUMN_BLOCKS][MMA1688_C]
  uint32_t RAH[2]; // RAH[MMA1688_A]
  uint32_t RAL[2]; // RAL[MMA1688_A]
  uint32_t RB[2];

#pragma unroll
  for (int i =  0; i < 2; i++)
  {
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      RC[i][j] = 0.0f; // Initialize result matrix to zero
    }
  }
  /*
      stAddr: Swizzled SMEM store address
      swizzle<3, 3, 3> configuration:
          First 3: 2^3=8 rows with 2-way bank conflict
          Second 3: 2^3=8 blocks per row
          Third 3: 2^3=8 elements per block
  */
  uint32_t stAddr = swizzle<3, 3, 3>(((lane_id) / 16) * 1024 + (lane_id % 8) * 8 + (warp_id) * 64 + ((lane_id % 16) / 8) * 512);
  /*
  rdAddr: Swizzled SMEM read address (unswizzling)
  */
  uint32_t rdAddr = swizzle<3, 3, 3>((warp_id / 4) * 512 + (warp_id & 3) * 16 + (lane_id % 8) * 64 + 8 * ((lane_id % 16) / 8));

  /*
    stAddr: Used for loading matrix A (input)
    stAddrB: Used for loading matrix B (weight)
    rdAddr and rdAddrB follow same pattern
  */
  uint32_t stAddrB = swizzle<3, 2, 3>((threadIdx.x) * 2);
  uint32_t rdAddrB = swizzle<3, 2, 3>((lane_id % 8) * 16 + (lane_id / 8) * 8);
  // Stage 0
  int k = 0;
  A_fused_col_idx = fuse[index[block_pos + k / 2] + (k % 2) * 8 + warp_id];
  loadSmem(&A_Smem[0][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
  if (threadIdx.x < 64)
    loadSmem1(&B_Smem[0][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + k * 128 + threadIdx.x * 2]);
  CP_ASYNC_COMMIT_GROUP(); // Commit async group (loadSmem + loadSmem1)
  // Stage 1
  for (k = 0; k < 4; k += 2)
  {
    // Get reordered column ID
    A_fused_col_idx = fuse[index[block_pos + (k + 1) / 2] + ((k + 1) % 2) * 8 + warp_id];
    loadSmem(&A_Smem[1][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
    if (threadIdx.x < 64)
      loadSmem1(&B_Smem[1][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + (k + 1) * 128 + threadIdx.x * 2]);
    CP_ASYNC_COMMIT_GROUP(); // Commit next async group
    CP_ASYNC_WAIT_GROUP(1);  // Sync async groups (1 pending group allowed)

    __syncthreads(); // Block synchronization

    // Load matrix A high/low bits to RAH/RAL, matrix B to RB
    loadFragX2_T(&A_Smem[0][rdAddr], RAH);
    loadFragX2_T(&A_Smem[0][rdAddr + 1024], RAL);
    loadFragX2_T(&B_Smem[0][rdAddrB], RB);
#pragma unroll
    for (int ci = 0; ci < 2; ci++)
    {
      // Perform HALF MMA computation
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAH[0], RAH[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAL[0], RAL[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
    }

    __syncthreads(); // Block synchronization

    // Second half of pipeline
    // Handles pipeline boundary without branching
    if (k + 2 < 4)
    {
      A_fused_col_idx = fuse[index[block_pos + (k + 2) / 2] + ((k + 2) % 2) * 8 + warp_id];
      loadSmem(&A_Smem[0][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
      if (threadIdx.x < 64)
        loadSmem1(&B_Smem[0][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + (k + 2) * 128 + threadIdx.x * 2]);
      CP_ASYNC_COMMIT_GROUP();
    }
    //
    CP_ASYNC_WAIT_GROUP(1);
    __syncthreads();
    loadFragX2_T(&A_Smem[1][rdAddr], RAH);
    loadFragX2_T(&A_Smem[1][rdAddr + 1024], RAL);
    loadFragX2_T(&B_Smem[1][rdAddrB], RB);
#pragma unroll
    for (int ci = 0; ci < 2; ci++)
    {
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAH[0], RAH[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAL[0], RAL[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
    }
    __syncthreads();
  }

  // Write results from registers to global memory
#pragma unroll
  for (int ci = 0; ci < 2; ci++)
  {
    uint64_t pos = (bid * blocksize_c + ci * 8 + (lane_id % 4) * 2) * (uint64_t)batch + A_start_row + warp_id * 16 + lane_id / 4;
    C[pos] = __ReLU(RC[ci][0] + bias);
    C[pos + batch] = __ReLU(RC[ci][1] + bias);
    C[pos + 8] = __ReLU(RC[ci][2] + bias);
    C[pos + batch + 8] = __ReLU(RC[ci][3] + bias);
  }

  return;
}
// final multi-network concurrent version
// Includes pipeline
// Includes swizzle
// TILE_M=128, TILE_N=16
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
                                                    int boundary)
{
  half *__restrict__ A;
  half *__restrict__ B;
  float *__restrict__ C;
  int *__restrict__ fuse;
  int *__restrict__ idx_ptr;
  int *__restrict__ index;
  float bias;
  int batch;
  int neuron;
  int tidy;
  int bid;
  if (blockIdx.x < boundary) // Distinguish SpDNN1/SpDNN2 (boundary = SpDNN1 block count)
  {
    A = A1;
    B = B1;
    C = C1;
    fuse = fuse1;
    idx_ptr = idx_ptr1;
    index = index1;
    bias = bias1;
    batch = batch1;
    neuron = neuron1;
    tidy = blockIdx.x / dim1;
    bid = blockIdx.x % dim1;
  }
  else
  {
    A = A2;
    B = B2;
    C = C2;
    fuse = fuse2;
    idx_ptr = idx_ptr2;
    index = index2;
    bias = bias2;
    batch = batch2;
    neuron = neuron2;
    tidy = (blockIdx.x - boundary) / dim2;
    bid = (blockIdx.x - boundary) % dim2;
  }
  int lane_id = threadIdx.x % WARP_SIZE; // Thread ID within warp
  int warp_id = threadIdx.x / WARP_SIZE; // Warp ID within block
  // Start row calculation
  int block_pos = idx_ptr[bid];      // Starting row index in neuron dimension 
  int A_start_row = tidy * TILE_DIM; // Starting row in batch dimension
  int A_fused_col_idx;               // Fused column index
  __shared__ half A_Smem[2][2 * 8 * 16 * 8];

  __shared__ half B_Smem[2][2 * 8 * 8]; 

  float RC[2][4];  // RC[WARP_COLUMN_BLOCKS][MMA1688_C]
  uint32_t RAH[2]; // RAH[MMA1688_A]
  uint32_t RAL[2]; // RAL[MMA1688_A]
  uint32_t RB[2];

#pragma unroll
  for (int i = 0; i < 2; i++)
  {
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
      RC[i][j] = 0.0f; // Initialize result matrix to zero
    }
  }

  uint32_t stAddr = swizzle<3, 3, 3>(((lane_id) / 16) * 1024 + (lane_id % 8) * 8 + (warp_id) * 64 + ((lane_id % 16) / 8) * 512);
  uint32_t rdAddr = swizzle<3, 3, 3>((warp_id / 4) * 512 + (warp_id & 3) * 16 + (lane_id % 8) * 64 + 8 * ((lane_id % 16) / 8));
  uint32_t stAddrB = swizzle<3, 2, 3>((threadIdx.x) * 2);
  uint32_t rdAddrB = swizzle<3, 2, 3>((lane_id % 8) * 16 + (lane_id / 8) * 8);
  // Stage 0
  int k = 0;
  A_fused_col_idx = fuse[index[block_pos + k / 2] + (k % 2) * 8 + warp_id];
  loadSmem(&A_Smem[0][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
  if (threadIdx.x < 64)
    loadSmem1(&B_Smem[0][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + k * 128 + threadIdx.x * 2]);
  CP_ASYNC_COMMIT_GROUP();
  // Stage 1
  for (k = 0; k < 4; k += 2)
  {
    A_fused_col_idx = fuse[index[block_pos + (k + 1) / 2] + ((k + 1) % 2) * 8 + warp_id];
    loadSmem(&A_Smem[1][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
    if (threadIdx.x < 64)
      loadSmem1(&B_Smem[1][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + (k + 1) * 128 + threadIdx.x * 2]);
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(1);
    __syncthreads();
    loadFragX2_T(&A_Smem[0][rdAddr], RAH);
    loadFragX2_T(&A_Smem[0][rdAddr + 1024], RAL);
    loadFragX2_T(&B_Smem[0][rdAddrB], RB);
#pragma unroll
    for (int ci = 0; ci < 2; ci++)
    {
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAH[0], RAH[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAL[0], RAL[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
    }
    __syncthreads();
    if (k + 2 < 4)
    {
      A_fused_col_idx = fuse[index[block_pos + (k + 2) / 2] + ((k + 2) % 2) * 8 + warp_id];
      loadSmem(&A_Smem[0][stAddr], &A[(uint64_t)(A_fused_col_idx + (((lane_id) / 16) * neuron)) * batch + A_start_row + (lane_id % 16) * 8]);
      if (threadIdx.x < 64)
        loadSmem1(&B_Smem[0][stAddrB], &B[(uint64_t)(block_pos)*blocksize_c * blocksize_r + (k + 2) * 128 + threadIdx.x * 2]);
      CP_ASYNC_COMMIT_GROUP();
    }
    //
    CP_ASYNC_WAIT_GROUP(1);
    __syncthreads();
    loadFragX2_T(&A_Smem[1][rdAddr], RAH);
    loadFragX2_T(&A_Smem[1][rdAddr + 1024], RAL);
    loadFragX2_T(&B_Smem[1][rdAddrB], RB);
#pragma unroll
    for (int ci = 0; ci < 2; ci++)
    {
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAH[0], RAH[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
      HMMA1688(RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3], RAL[0], RAL[1], RB[ci], RC[ci][0], RC[ci][1], RC[ci][2], RC[ci][3]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int ci = 0; ci < 2; ci++)
  {
    uint64_t pos = (bid * blocksize_c + ci * 8 + (lane_id % 4) * 2) * (uint64_t)batch + A_start_row + warp_id * 16 + lane_id / 4;
    C[pos] = __ReLU(RC[ci][0] + bias);
    C[pos + batch] = __ReLU(RC[ci][1] + bias);
    C[pos + 8] = __ReLU(RC[ci][2] + bias);
    C[pos + batch + 8] = __ReLU(RC[ci][3] + bias);
  }

  return;
}

};