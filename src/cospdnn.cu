#include "./cospdnn.cuh"
#include <algorithm>
void test_graph_challenge_tc(std::vector<std::vector<float>> &input,
                             std::vector<std::vector<float>> &weight,
                             std::vector<std::vector<int>> &row_idx,
                             std::vector<std::vector<int>> &ptr,
                             std::vector<std::vector<int>> &fuse_list,
                             int batch, int neuron, float bias,
                             int blocksize_c, int blocksize_r,
                             int gpu_id, int gpu_num)
{
  float *A, *A_d;
  float *C, *C_d;
  half **B, *B_d, *B_d_1, *B_d_2, *AH_d;
  int **index, *index_d, *index_d_1, *index_d_2;
  int **col_ptr, *col_ptr_d, *col_ptr_d_1, *col_ptr_d_2;
  int **fuse, *fuse_d, *fuse_d_1, *fuse_d_2;

  int *category, *category_d;
  int *active, *active_d;
  int *old_to_new_map, *old_to_new_map_d;

  batch = (batch + gpu_num - 1) / gpu_num;
  int this_round_batch = (batch + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
  int last_feature = (batch + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
  int layer = weight.size();
  std::cout << "initial padding batch " << this_round_batch << std::endl;

  // initialize and generate input
  A = (float *)malloc(sizeof(float) * (uint64_t)neuron * this_round_batch);
  C = (float *)malloc(sizeof(float) * (uint64_t)neuron * this_round_batch);
  memset(A, 0, sizeof(float) * (uint64_t)neuron * this_round_batch);
  memset(C, 0, sizeof(float) * (uint64_t)neuron * this_round_batch);

  // transpose
  int data_start = gpu_id * batch;
  int data_end = std::min((gpu_id + 1) * batch, (int)input.size());
  for (int l = data_start; l < data_end; ++l)
  {
    for (int i = 0; i < input[l].size(); ++i)
    {
      A[i * (uint64_t)this_round_batch + l - data_start] = input[l][i];
    }
  }

  size_t max_B_size = 0;
  B = (half **)malloc(sizeof(half *) * weight.size());
  for (int l = 0; l < weight.size(); ++l)
  {
    B[l] = (half *)malloc(sizeof(half *) * weight[l].size());
    max_B_size = std::max(max_B_size, weight[l].size());
    for (int i = 0; i < weight[l].size(); ++i)
    {
      B[l][i] = __float2half(weight[l][i]);
    }
  }

  size_t max_idx_size = 0;
  index = (int **)malloc(sizeof(int *) * row_idx.size());
  for (int l = 0; l < row_idx.size(); ++l)
  {
    index[l] = (int *)malloc(sizeof(int *) * row_idx[l].size());
    max_idx_size = std::max(max_idx_size, row_idx[l].size());
    for (int i = 0; i < row_idx[l].size(); ++i)
    {
      index[l][i] = row_idx[l][i];
    }
  }

  size_t max_ptr_size = 0;
  col_ptr = (int **)malloc(sizeof(int *) * ptr.size());
  for (int l = 0; l < ptr.size(); ++l)
  {
    col_ptr[l] = (int *)malloc(sizeof(int *) * ptr[l].size());
    max_ptr_size = std::max(max_ptr_size, ptr[l].size());
    for (int i = 0; i < ptr[l].size(); ++i)
    {
      col_ptr[l][i] = ptr[l][i];
    }
  }

  size_t max_fuse_size = 0;
  fuse = (int **)malloc(sizeof(int *) * fuse_list.size());
  for (int l = 0; l < fuse_list.size(); ++l)
  {
    fuse[l] = (int *)malloc(sizeof(int *) * fuse_list[l].size());
    max_fuse_size = std::max(max_fuse_size, fuse_list[l].size());
    for (int i = 0; i < fuse_list[l].size(); ++i)
    {
      fuse[l][i] = fuse_list[l][i];
    }
  }

  category = (int *)malloc(sizeof(int *) * this_round_batch);
  for (int i = 0; i < this_round_batch; ++i)
  {
    category[i] = i;
  }

  old_to_new_map = (int *)malloc(sizeof(int *) * this_round_batch);
  for (int i = 0; i < this_round_batch; ++i)
  {
    old_to_new_map[i] = i;
  }

  active = (int *)malloc(sizeof(int *) * this_round_batch);
  for (int i = 0; i < this_round_batch; ++i)
  {
    active[i] = 0;
  }
  std::cout << "CPU data alloc done!" << std::endl;

  Safe_Call(cudaMalloc((void **)&A_d, sizeof(float) * (uint64_t)neuron * this_round_batch));
  Safe_Call(cudaMemset(A_d, 0, sizeof(float) * (uint64_t)neuron * this_round_batch));
  Safe_Call(cudaMemcpy(A_d, A, sizeof(float) * (uint64_t)neuron * this_round_batch, cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&AH_d, sizeof(half) * (uint64_t)neuron * this_round_batch * 2));
  Safe_Call(cudaMemset(AH_d, 0, sizeof(half) * (uint64_t)neuron * this_round_batch * 2));


  Safe_Call(cudaMalloc((void **)&C_d, sizeof(float) * (uint64_t)neuron * this_round_batch));
  Safe_Call(cudaMemset(C_d, 0, sizeof(float) * (uint64_t)neuron * this_round_batch));

  Safe_Call(cudaMalloc((void **)&active_d, sizeof(int) * this_round_batch));
  Safe_Call(cudaMemset(active_d, 0, sizeof(int) * this_round_batch));
  Safe_Call(cudaMalloc((void **)&category_d, sizeof(int) * this_round_batch));
  Safe_Call(cudaMemcpy(category_d, category, sizeof(int) * this_round_batch, cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&old_to_new_map_d, sizeof(int) * this_round_batch));
  Safe_Call(cudaMemset(old_to_new_map_d, 0, sizeof(int) * this_round_batch));

  std::cout << "GPU Residency data done!" << std::endl;

  Safe_Call(cudaMalloc((void **)&(B_d_1), sizeof(half) * max_B_size));
  Safe_Call(cudaMemcpy(B_d_1, B[0], sizeof(half) * weight[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(B_d_2), sizeof(half) * max_B_size));
  Safe_Call(cudaMemcpy(B_d_2, B[1], sizeof(half) * weight[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(index_d_1), sizeof(int) * max_idx_size));
  Safe_Call(cudaMemcpy(index_d_1, index[0], sizeof(int) * row_idx[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(index_d_2), sizeof(int) * max_idx_size));
  Safe_Call(cudaMemcpy(index_d_2, index[1], sizeof(int) * row_idx[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(col_ptr_d_1), sizeof(int) * max_ptr_size));
  Safe_Call(cudaMemcpy(col_ptr_d_1, col_ptr[0], sizeof(int) * ptr[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(col_ptr_d_2), sizeof(int) * max_ptr_size));
  Safe_Call(cudaMemcpy(col_ptr_d_2, col_ptr[1], sizeof(int) * ptr[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(fuse_d_1), sizeof(int) * max_fuse_size));
  Safe_Call(cudaMemcpy(fuse_d_1, fuse[0], sizeof(int) * fuse_list[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(fuse_d_2), sizeof(int) * max_fuse_size));
  Safe_Call(cudaMemcpy(fuse_d_2, fuse[1], sizeof(int) * fuse_list[1].size(), cudaMemcpyHostToDevice));

  std::cout << "GPU Weight data done!" << std::endl;

  float mili_all_time = 0;
  float mili_iter_time = 0;
  cudaEvent_t start, stop;
  Safe_Call(cudaEventCreate(&start));
  Safe_Call(cudaEventCreate(&stop));
  cudaStream_t kernel_stream, memory_stream;
  Safe_Call(cudaStreamCreate(&kernel_stream));
  Safe_Call(cudaStreamCreate(&memory_stream));

  // gpu kernel configuration
  int blocksize_x, blocksize_y;
  int gridsize_x, gridsize_y;

// start iteration
#ifdef GPU_TEST
  for (int l = 0; l < 1; l++)
#else
  for (int l = 0; l < layer; ++l)
#endif
  {
    Safe_Call(cudaStreamSynchronize(memory_stream));

    if (l + 1 < layer)
    {
      if (l % 2 == 1)
      {
        index_d = index_d_2;
        B_d = B_d_2;
        fuse_d = fuse_d_2;
        col_ptr_d = col_ptr_d_2;
        Safe_Call(cudaMemcpyAsync(B_d_1, B[l + 1],
                                  sizeof(half) * weight[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(index_d_1, index[l + 1],
                                  sizeof(int) * row_idx[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(col_ptr_d_1, col_ptr[l + 1],
                                  sizeof(int) * ptr[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(fuse_d_1, fuse[l + 1],
                                  sizeof(int) * fuse_list[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
      }
      else
      {
        index_d = index_d_1;
        B_d = B_d_1;
        fuse_d = fuse_d_1;
        col_ptr_d = col_ptr_d_1;
        Safe_Call(cudaMemcpyAsync(B_d_2, B[l + 1],
                                  sizeof(half) * weight[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(index_d_2, index[l + 1],
                                  sizeof(int) * row_idx[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(col_ptr_d_2, col_ptr[l + 1],
                                  sizeof(int) * ptr[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
        Safe_Call(cudaMemcpyAsync(fuse_d_2, fuse[l + 1],
                                  sizeof(int) * fuse_list[l + 1].size(), cudaMemcpyHostToDevice, memory_stream));
      }
    }

    cudaStream_t stream = kernel_stream;
    Safe_Call(cudaMemsetAsync(active_d, 0, sizeof(int) * this_round_batch, stream));

    blocksize_x = 4 * WARP_SIZE;
    gridsize_x = (neuron * (uint64_t)this_round_batch + blocksize_x - 1) / blocksize_x;
    floatTohalf2<<<gridsize_x, blocksize_x, 0, stream>>>(A_d, AH_d, neuron, this_round_batch);
    Safe_Call(cudaDeviceSynchronize());

    blocksize_x = 4 * WARP_SIZE * 2;
    gridsize_x = (ptr[l].size() - 1);
    blocksize_y = 1;
    gridsize_y = (this_round_batch + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM);
    dim3 gridDim_f(gridsize_x, gridsize_y);
    dim3 blockDim_f(blocksize_x, blocksize_y);

    // trans float to 2half
    Safe_Call(cudaEventRecord(start, kernel_stream));

    cospdnn_single<<<gridDim_f, blockDim_f, 0, stream>>>(AH_d, B_d, C_d, fuse_d, col_ptr_d, index_d, bias, this_round_batch, neuron, blocksize_c, blocksize_r);

    Safe_Call(cudaDeviceSynchronize());
    Safe_Call(cudaEventRecord(stop, stream));
    Safe_Call(cudaEventSynchronize(stop));
    Safe_Call(cudaEventElapsedTime(&mili_iter_time, start, stop));
    // postprocess
    gridsize_x = (neuron * (uint64_t)this_round_batch + blocksize_x - 1) / blocksize_x;
    check_active_f_col<<<gridsize_x, blocksize_x, 0, stream>>>(C_d, active_d, neuron, this_round_batch);
    Safe_Call(cudaDeviceSynchronize());

    if (l <= 21)
    {
      Safe_Call(cudaMemcpy(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost));
    }
    else
    {
      Safe_Call(cudaMemcpyAsync(active, active_d, sizeof(int) * this_round_batch, cudaMemcpyDeviceToHost, stream));
    }

    int feature = 0;
    int padding = 0;
    if (l <= 21)
    {
      for (int k = 0; k < this_round_batch; k++)
      {
        if (active[k] > 0)
        {
          category[feature] = k;
          old_to_new_map[feature] = old_to_new_map[k];
          feature++;
        }
      }
#ifdef PRINT_TEST
      std::cout << "layer " << l << ", feature = " << feature << std::endl;
#endif
      Safe_Call(cudaMemcpy(category_d, category, sizeof(int) * feature, cudaMemcpyHostToDevice));
      // padding
      padding = (feature + TILE_DIM - 1) / TILE_DIM * TILE_DIM - feature;
      last_feature = this_round_batch;
      this_round_batch = (feature + padding);
      // if(l==21) this_round_batch = (feature + padding)*2;
      gridsize_x = (feature * (uint64_t)neuron + blocksize_x - 1) / blocksize_x;
      Safe_Call(cudaMemset(A_d, 0, sizeof(float) * (uint64_t)neuron * this_round_batch));

      matrix_reshape<<<gridsize_x, blocksize_x, 0, stream>>>(C_d, A_d, category_d, neuron, last_feature, feature, this_round_batch);
      Safe_Call(cudaDeviceSynchronize());
    }
    else
    {
      float *tmp = C_d;
      C_d = A_d;
      A_d = tmp;
      if (l == 22)
      {
        Safe_Call(cudaFree(C_d));
        Safe_Call(cudaFree(AH_d));
        Safe_Call(cudaMalloc((void **)&C_d, sizeof(float) * (uint64_t)neuron * this_round_batch));
        Safe_Call(cudaMemset(C_d, 0, sizeof(float) * (uint64_t)neuron * this_round_batch));
        Safe_Call(cudaMalloc((void **)&AH_d, sizeof(half) * (uint64_t)neuron * this_round_batch * 2));


        float *AT_d;
        Safe_Call(cudaMalloc((void **)&AT_d, sizeof(float) * (uint64_t)neuron * this_round_batch));
        Safe_Call(cudaMemcpy(AT_d, A_d, sizeof(float) * (uint64_t)neuron * this_round_batch, cudaMemcpyDeviceToDevice));
        Safe_Call(cudaFree(A_d));
        A_d = AT_d;
      }
    }
    for (int i = 0; i < this_round_batch; ++i)
    {
      active[i] = 0;
    }
#ifdef PRINT_TEST
    std::cout << "layer " << l << ", batch = " << this_round_batch << std::endl;
    std::cout << "Layer " << l << " exec Time = " << mili_iter_time << std::endl;
#endif
    mili_all_time += mili_iter_time;
  }

  Safe_Call(cudaMemcpy(C, C_d, sizeof(float) * (uint64_t)neuron * this_round_batch, cudaMemcpyDeviceToHost));
  std::cout << "Kernel Exec Time = " << mili_all_time << "ms" << std::endl;
#ifdef PRINT_RESULT
  for (int i = 0; i < neuron; i++)
  {
    for (int j = 0; j < this_round_batch; j++)
    {
      std::cout << C[i * this_round_batch + j] << " ";
    }
    std::cout << "\n";
  }
#endif
  // free the variables
  // free the host pointers
  free(A);
  free(C);
  for (int i = 0; i < weight.size(); i++)
  {
    free(B[i]);
  }
  free(B);
  for (int i = 0; i < row_idx.size(); i++)
  {
    free(index[i]);
  }
  free(index);
  for (int i = 0; i < ptr.size(); i++)
  {
    free(col_ptr[i]);
  }
  free(col_ptr);
  for (int i = 0; i < fuse_list.size(); i++)
  {
    free(fuse[i]);
  }
  free(fuse);
  free(category);
  free(active);
  free(old_to_new_map);

  // free the device pointers
  Safe_Call(cudaFree(A_d));
  Safe_Call(cudaFree(AH_d));
  Safe_Call(cudaFree(active_d));
  Safe_Call(cudaFree(category_d));
  Safe_Call(cudaFree(old_to_new_map_d));
  Safe_Call(cudaFree(B_d_1));
  Safe_Call(cudaFree(B_d_2));
  Safe_Call(cudaFree(index_d_1));
  Safe_Call(cudaFree(index_d_2));
  Safe_Call(cudaFree(col_ptr_d_1));
  Safe_Call(cudaFree(col_ptr_d_2));
  Safe_Call(cudaFree(fuse_d_1));
  Safe_Call(cudaFree(fuse_d_2));
  Safe_Call(cudaEventDestroy(start));
  Safe_Call(cudaEventDestroy(stop));
  Safe_Call(cudaStreamDestroy(kernel_stream));
  Safe_Call(cudaStreamDestroy(memory_stream));

  return;
}
