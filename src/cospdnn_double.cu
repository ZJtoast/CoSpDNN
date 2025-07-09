#include "./cospdnn.cuh"
#include <algorithm>
bool evaluateConfig(int layer, const cudaDeviceProp &prop, dim3 blockDims, dim3 gridDims)
{
  if (layer < 5)
    return false;
  int warpsPerBlock = (blockDims.x * blockDims.y * blockDims.z + prop.warpSize - 1) / prop.warpSize;
  int totalWarps = warpsPerBlock * gridDims.x * gridDims.y * gridDims.z;
  int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;

  int activeBlocksPerSM = 0;

  int regsPerThread = 32; // by NsightCompute
  int regsPerBlock = regsPerThread * blockDims.x * blockDims.y * blockDims.z;
  if (regsPerBlock > 0)
  {
    activeBlocksPerSM = prop.regsPerMultiprocessor / regsPerBlock;
  }
  activeBlocksPerSM = std::min(activeBlocksPerSM, prop.maxBlocksPerMultiProcessor);
  if (warpsPerBlock * activeBlocksPerSM < maxWarpsPerSM || totalWarps < 65536) 
  {
    std::cout << "Parallel Compute" << std::endl;
    return true;
  }
  else
  {
    std::cout << "Serial Compute" << std::endl;
    return false;
  }
}
void performPre(
    int l,                                    
    cudaStream_t kernel_stream,               
    cudaStream_t memory_stream,               
    int &this_round_batch,                    
    int &last_feature,                        
    int neuron,                              
    float bias,                              
    float *&A_d,                              
    float *&C_d,                              
    half *&AH_d,                              
    half **B,                                 
    int **index,                            
    int **col_ptr,                         
    int **fuse,                           
    std::vector<std::vector<float>> &weight, 
    std::vector<std::vector<int>> &row_idx,  
    std::vector<std::vector<int>> &ptr,      
    std::vector<std::vector<int>> &fuse_list, 
    int *active,                             
    int *active_d,                           
    int *category,                         
    int *old_to_new_map,                   
    int *category_d,                        
    half *&B_d_1, half *&B_d_2,             
    int *&index_d_1, int *&index_d_2,      
    int *&col_ptr_d_1, int *&col_ptr_d_2,   
    int *&fuse_d_1, int *&fuse_d_2,         
    int blocksize_c,                         
    int blocksize_r,                     
    float &mili_iter_time,                  
    float &mili_all_time,                   
    int layer,                               
    cudaEvent_t start,                        
    cudaEvent_t stop                        
)
{
  half *B_d;    
  int *index_d;  
  int *col_ptr_d; 
  int *fuse_d;    
  int blocksize_x, blocksize_y;
  int gridsize_x, gridsize_y;
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
  // gridsize_x = (ptr.size() - 1 + blocksize_x / WARP_SIZE - 1) / (blocksize_x / WARP_SIZE);
  blocksize_y = 1;
  gridsize_y = (this_round_batch + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM);
  // gridsize_y = (batch + blocksize_y * MINIBATCH * 16 - 1)/ (blocksize_y * MINIBATCH * 16);

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
void test_graph_challenge_tc_co(std::vector<std::vector<float>> &input1,
                                std::vector<std::vector<float>> &weight1,
                                std::vector<std::vector<int>> &row_idx1,
                                std::vector<std::vector<int>> &ptr1,
                                std::vector<std::vector<int>> &fuse_list1,
                                int batch1, int neuron1, float bias1,
                                std::vector<std::vector<float>> &input2,
                                std::vector<std::vector<float>> &weight2,
                                std::vector<std::vector<int>> &row_idx2,
                                std::vector<std::vector<int>> &ptr2,
                                std::vector<std::vector<int>> &fuse_list2,
                                int batch2, int neuron2, float bias2,
                                int blocksize_c, int blocksize_r,
                                int gpu_id, int gpu_num)
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  float *A1, *A1_d, *A2, *A2_d;
  float *C1, *C1_d, *C2, *C2_d;
  half **B1, **B2, *B1_d_1, *B1_d_2, *B2_d_1, *B2_d_2, *AH1_d, *AH2_d;
  int **index1, **index2, *index1_d_1, *index1_d_2, *index2_d_1, *index2_d_2;
  int **col_ptr1, **col_ptr2, *col_ptr1_d_1, *col_ptr1_d_2, *col_ptr2_d_1, *col_ptr2_d_2;
  int **fuse1, **fuse2, *fuse1_d_1, *fuse1_d_2, *fuse2_d_1, *fuse2_d_2;

  int *category1, *category1_d, *category2, *category2_d;
  int *active1, *active1_d, *active2, *active2_d;
  int *old_to_new_map1, *old_to_new_map1_d, *old_to_new_map2, *old_to_new_map2_d;

  batch1 = (batch1 + gpu_num - 1) / gpu_num;
  batch2 = (batch2 + gpu_num - 1) / gpu_num;
  int this_round_batch1 = (batch1 + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
  int this_round_batch2 = (batch2 + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
  int last_feature1 = (batch1 + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
  int last_feature2 = (batch2 + TILE_DIM - 1) / TILE_DIM * TILE_DIM;

  int layer = weight1.size();
  if (layer != weight2.size())
  {
    std::cerr << "Error: Two networks must have the same number of layers!" << std::endl;
    return;
  }
  std::cout << "initial padding batch " << this_round_batch1 << " and " << this_round_batch2 << std::endl;
  std::cout << "TILE_DIM " << TILE_DIM << std::endl;

  A1 = (float *)malloc(sizeof(float) * (uint64_t)neuron1 * this_round_batch1);
  C1 = (float *)malloc(sizeof(float) * (uint64_t)neuron1 * this_round_batch1);
  memset(A1, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1);
  memset(C1, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1);


  int data_start1 = gpu_id * batch1;
  int data_end1 = std::min((gpu_id + 1) * batch1, (int)input1.size());
  for (int l = data_start1; l < data_end1; ++l)
  {
    for (int i = 0; i < input1[l].size(); ++i)
    {
      A1[i * (uint64_t)this_round_batch1 + l - data_start1] = input1[l][i];
    }
  }

  A2 = (float *)malloc(sizeof(float) * (uint64_t)neuron2 * this_round_batch2);
  C2 = (float *)malloc(sizeof(float) * (uint64_t)neuron2 * this_round_batch2);
  memset(A2, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2);
  memset(C2, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2);

  int data_start2 = gpu_id * batch2;
  int data_end2 = std::min((gpu_id + 1) * batch2, (int)input2.size());
  for (int l = data_start2; l < data_end2; ++l)
  {
    for (int i = 0; i < input2[l].size(); ++i)
    {
      A2[i * (uint64_t)this_round_batch2 + l - data_start2] = input2[l][i];
    }
  }

  size_t max_B_size1 = 0;
  B1 = (half **)malloc(sizeof(half *) * weight1.size());
  for (int l = 0; l < weight1.size(); ++l)
  {
    B1[l] = (half *)malloc(sizeof(half) * weight1[l].size());
    max_B_size1 = std::max(max_B_size1, weight1[l].size());
    for (int i = 0; i < weight1[l].size(); ++i)
    {
      B1[l][i] = __float2half(weight1[l][i]);
    }
  }

  size_t max_idx_size1 = 0;
  index1 = (int **)malloc(sizeof(int *) * row_idx1.size());
  for (int l = 0; l < row_idx1.size(); ++l)
  {
    index1[l] = (int *)malloc(sizeof(int) * row_idx1[l].size());
    max_idx_size1 = std::max(max_idx_size1, row_idx1[l].size());
    for (int i = 0; i < row_idx1[l].size(); ++i)
    {
      index1[l][i] = row_idx1[l][i];
    }
  }

  size_t max_ptr_size1 = 0;
  col_ptr1 = (int **)malloc(sizeof(int *) * ptr1.size());
  for (int l = 0; l < ptr1.size(); ++l)
  {
    col_ptr1[l] = (int *)malloc(sizeof(int) * ptr1[l].size());
    max_ptr_size1 = std::max(max_ptr_size1, ptr1[l].size());
    for (int i = 0; i < ptr1[l].size(); ++i)
    {
      col_ptr1[l][i] = ptr1[l][i];
    }
  }

  size_t max_fuse_size1 = 0;
  fuse1 = (int **)malloc(sizeof(int *) * fuse_list1.size());
  for (int l = 0; l < fuse_list1.size(); ++l)
  {
    fuse1[l] = (int *)malloc(sizeof(int) * fuse_list1[l].size());
    max_fuse_size1 = std::max(max_fuse_size1, fuse_list1[l].size());
    for (int i = 0; i < fuse_list1[l].size(); ++i)
    {
      fuse1[l][i] = fuse_list1[l][i];
    }
  }


  size_t max_B_size2 = 0;
  B2 = (half **)malloc(sizeof(half *) * weight2.size());
  for (int l = 0; l < weight2.size(); ++l)
  {
    B2[l] = (half *)malloc(sizeof(half) * weight2[l].size());
    max_B_size2 = std::max(max_B_size2, weight2[l].size());
    for (int i = 0; i < weight2[l].size(); ++i)
    {
      B2[l][i] = __float2half(weight2[l][i]);
    }
  }

  size_t max_idx_size2 = 0;
  index2 = (int **)malloc(sizeof(int *) * row_idx2.size());
  for (int l = 0; l < row_idx2.size(); ++l)
  {
    index2[l] = (int *)malloc(sizeof(int) * row_idx2[l].size());
    max_idx_size2 = std::max(max_idx_size2, row_idx2[l].size());
    for (int i = 0; i < row_idx2[l].size(); ++i)
    {
      index2[l][i] = row_idx2[l][i];
    }
  }

  size_t max_ptr_size2 = 0;
  col_ptr2 = (int **)malloc(sizeof(int *) * ptr2.size());
  for (int l = 0; l < ptr2.size(); ++l)
  {
    col_ptr2[l] = (int *)malloc(sizeof(int) * ptr2[l].size());
    max_ptr_size2 = std::max(max_ptr_size2, ptr2[l].size());
    for (int i = 0; i < ptr2[l].size(); ++i)
    {
      col_ptr2[l][i] = ptr2[l][i];
    }
  }

  size_t max_fuse_size2 = 0;
  fuse2 = (int **)malloc(sizeof(int *) * fuse_list2.size());
  for (int l = 0; l < fuse_list2.size(); ++l)
  {
    fuse2[l] = (int *)malloc(sizeof(int) * fuse_list2[l].size());
    max_fuse_size2 = std::max(max_fuse_size2, fuse_list2[l].size());
    for (int i = 0; i < fuse_list2[l].size(); ++i)
    {
      fuse2[l][i] = fuse_list2[l][i];
    }
  }


  category1 = (int *)malloc(sizeof(int) * this_round_batch1);
  for (int i = 0; i < this_round_batch1; ++i)
    category1[i] = i;

  old_to_new_map1 = (int *)malloc(sizeof(int) * this_round_batch1);
  for (int i = 0; i < this_round_batch1; ++i)
    old_to_new_map1[i] = i;

  active1 = (int *)malloc(sizeof(int) * this_round_batch1);
  memset(active1, 0, sizeof(int) * this_round_batch1);


  category2 = (int *)malloc(sizeof(int) * this_round_batch2);
  for (int i = 0; i < this_round_batch2; ++i)
    category2[i] = i;

  old_to_new_map2 = (int *)malloc(sizeof(int) * this_round_batch2);
  for (int i = 0; i < this_round_batch2; ++i)
    old_to_new_map2[i] = i;

  active2 = (int *)malloc(sizeof(int) * this_round_batch2);
  memset(active2, 0, sizeof(int) * this_round_batch2);

  std::cout << "CPU data alloc done!" << std::endl;


  Safe_Call(cudaMalloc((void **)&A1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
  Safe_Call(cudaMemcpy(A1_d, A1, sizeof(float) * (uint64_t)neuron1 * this_round_batch1, cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&C1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
  Safe_Call(cudaMemset(C1_d, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));

  Safe_Call(cudaMalloc((void **)&active1_d, sizeof(int) * this_round_batch1));
  Safe_Call(cudaMemset(active1_d, 0, sizeof(int) * this_round_batch1));
  Safe_Call(cudaMalloc((void **)&category1_d, sizeof(int) * this_round_batch1));
  Safe_Call(cudaMemcpy(category1_d, category1, sizeof(int) * this_round_batch1, cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&old_to_new_map1_d, sizeof(int) * this_round_batch1));
  Safe_Call(cudaMemset(old_to_new_map1_d, 0, sizeof(int) * this_round_batch1));

  Safe_Call(cudaMalloc((void **)&AH1_d, sizeof(half) * (uint64_t)neuron1 * this_round_batch1 * 2));
  Safe_Call(cudaMemset(AH1_d, 0, sizeof(half) * (uint64_t)neuron1 * this_round_batch1 * 2));


  Safe_Call(cudaMalloc((void **)&(B1_d_1), sizeof(half) * max_B_size1));
  Safe_Call(cudaMemcpy(B1_d_1, B1[0], sizeof(half) * weight1[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(B1_d_2), sizeof(half) * max_B_size1));
  Safe_Call(cudaMemcpy(B1_d_2, B1[1], sizeof(half) * weight1[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(index1_d_1), sizeof(int) * max_idx_size1));
  Safe_Call(cudaMemcpy(index1_d_1, index1[0], sizeof(int) * row_idx1[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(index1_d_2), sizeof(int) * max_idx_size1));
  Safe_Call(cudaMemcpy(index1_d_2, index1[1], sizeof(int) * row_idx1[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(col_ptr1_d_1), sizeof(int) * max_ptr_size1));
  Safe_Call(cudaMemcpy(col_ptr1_d_1, col_ptr1[0], sizeof(int) * ptr1[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(col_ptr1_d_2), sizeof(int) * max_ptr_size1));
  Safe_Call(cudaMemcpy(col_ptr1_d_2, col_ptr1[1], sizeof(int) * ptr1[1].size(), cudaMemcpyHostToDevice));

  Safe_Call(cudaMalloc((void **)&(fuse1_d_1), sizeof(int) * max_fuse_size1));
  Safe_Call(cudaMemcpy(fuse1_d_1, fuse1[0], sizeof(int) * fuse_list1[0].size(), cudaMemcpyHostToDevice));
  Safe_Call(cudaMalloc((void **)&(fuse1_d_2), sizeof(int) * max_fuse_size1));
  Safe_Call(cudaMemcpy(fuse1_d_2, fuse1[1], sizeof(int) * fuse_list1[1].size(), cudaMemcpyHostToDevice));

  std::cout << "GPU Residency data done for network 1!" << std::endl;

  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  double used_mem = total_mem - free_mem;
  double ratio = used_mem / total_mem;
  bool delayed_allocation = (ratio > 0.45); 
  std::cout << "GPU Memory Usage: " << ratio * 100 << "% - Delayed Allocation: "
            << (delayed_allocation ? "YES" : "NO") << std::endl;

  if (!delayed_allocation)
  {

    Safe_Call(cudaMalloc((void **)&A2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
    Safe_Call(cudaMemcpy(A2_d, A2, sizeof(float) * (uint64_t)neuron2 * this_round_batch2, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&C2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
    Safe_Call(cudaMemset(C2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));

    Safe_Call(cudaMalloc((void **)&active2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemset(active2_d, 0, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMalloc((void **)&category2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemcpy(category2_d, category2, sizeof(int) * this_round_batch2, cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&old_to_new_map2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemset(old_to_new_map2_d, 0, sizeof(int) * this_round_batch2));

    Safe_Call(cudaMalloc((void **)&AH2_d, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));
    Safe_Call(cudaMemset(AH2_d, 0, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));

    Safe_Call(cudaMalloc((void **)&(B2_d_1), sizeof(half) * max_B_size2));
    Safe_Call(cudaMemcpy(B2_d_1, B2[0], sizeof(half) * weight2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(B2_d_2), sizeof(half) * max_B_size2));
    Safe_Call(cudaMemcpy(B2_d_2, B2[1], sizeof(half) * weight2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(index2_d_1), sizeof(int) * max_idx_size2));
    Safe_Call(cudaMemcpy(index2_d_1, index2[0], sizeof(int) * row_idx2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(index2_d_2), sizeof(int) * max_idx_size2));
    Safe_Call(cudaMemcpy(index2_d_2, index2[1], sizeof(int) * row_idx2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(col_ptr2_d_1), sizeof(int) * max_ptr_size2));
    Safe_Call(cudaMemcpy(col_ptr2_d_1, col_ptr2[0], sizeof(int) * ptr2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(col_ptr2_d_2), sizeof(int) * max_ptr_size2));
    Safe_Call(cudaMemcpy(col_ptr2_d_2, col_ptr2[1], sizeof(int) * ptr2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(fuse2_d_1), sizeof(int) * max_fuse_size2));
    Safe_Call(cudaMemcpy(fuse2_d_1, fuse2[0], sizeof(int) * fuse_list2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(fuse2_d_2), sizeof(int) * max_fuse_size2));
    Safe_Call(cudaMemcpy(fuse2_d_2, fuse2[1], sizeof(int) * fuse_list2[1].size(), cudaMemcpyHostToDevice));

    std::cout << "GPU Weight data done for network 2!" << std::endl;
  }

  float mili_all_time = 0;
  float mili_iter_time = 0;
  cudaEvent_t start, stop;
  Safe_Call(cudaEventCreate(&start));
  Safe_Call(cudaEventCreate(&stop));
  cudaStream_t kernel_stream, memory_stream;
  Safe_Call(cudaStreamCreate(&kernel_stream));
  Safe_Call(cudaStreamCreate(&memory_stream));


  int blocksize_x, blocksize_y;
  int gridsize_x, gridsize_y = 1;


  if (delayed_allocation)
  {
    std::cout << "Executing delayed allocation strategy..." << std::endl;

    for (int l = 0; l < 23; ++l)
    {
      performPre(
          l,                        
          kernel_stream,            
          memory_stream,            
          this_round_batch1,     
          last_feature1,           
          neuron1,                  
          bias1,                  
          A1_d,                   
          C1_d,                  
          AH1_d,                   
          B1,                    
          index1,               
          col_ptr1,                  
          fuse1,                     
          weight1,           
          row_idx1,            
          ptr1,                  
          fuse_list1,              
          active1,                    
          active1_d,               
          category1,                  
          old_to_new_map1,        
          category1_d,                
          B1_d_1, B1_d_2,           
          index1_d_1, index1_d_2,    
          col_ptr1_d_1, col_ptr1_d_2, 
          fuse1_d_1, fuse1_d_2,       
          blocksize_c,               
          blocksize_r,                
          mili_iter_time,             
          mili_all_time,              
          layer,                      
          start,                     
          stop                        
      );
    }


    std::cout << "Allocating GPU memory for network 2..." << std::endl;
    Safe_Call(cudaMalloc((void **)&A2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
    Safe_Call(cudaMemcpy(A2_d, A2, sizeof(float) * (uint64_t)neuron2 * this_round_batch2, cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&C2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
    Safe_Call(cudaMemset(C2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));

    Safe_Call(cudaMalloc((void **)&active2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemset(active2_d, 0, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMalloc((void **)&category2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemcpy(category2_d, category2, sizeof(int) * this_round_batch2, cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&old_to_new_map2_d, sizeof(int) * this_round_batch2));
    Safe_Call(cudaMemset(old_to_new_map2_d, 0, sizeof(int) * this_round_batch2));

    Safe_Call(cudaMalloc((void **)&AH2_d, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));
    Safe_Call(cudaMemset(AH2_d, 0, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));

    Safe_Call(cudaMalloc((void **)&(B2_d_1), sizeof(half) * max_B_size2));
    Safe_Call(cudaMemcpy(B2_d_1, B2[0], sizeof(half) * weight2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(B2_d_2), sizeof(half) * max_B_size2));
    Safe_Call(cudaMemcpy(B2_d_2, B2[1], sizeof(half) * weight2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(index2_d_1), sizeof(int) * max_idx_size2));
    Safe_Call(cudaMemcpy(index2_d_1, index2[0], sizeof(int) * row_idx2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(index2_d_2), sizeof(int) * max_idx_size2));
    Safe_Call(cudaMemcpy(index2_d_2, index2[1], sizeof(int) * row_idx2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(col_ptr2_d_1), sizeof(int) * max_ptr_size2));
    Safe_Call(cudaMemcpy(col_ptr2_d_1, col_ptr2[0], sizeof(int) * ptr2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(col_ptr2_d_2), sizeof(int) * max_ptr_size2));
    Safe_Call(cudaMemcpy(col_ptr2_d_2, col_ptr2[1], sizeof(int) * ptr2[1].size(), cudaMemcpyHostToDevice));

    Safe_Call(cudaMalloc((void **)&(fuse2_d_1), sizeof(int) * max_fuse_size2));
    Safe_Call(cudaMemcpy(fuse2_d_1, fuse2[0], sizeof(int) * fuse_list2[0].size(), cudaMemcpyHostToDevice));
    Safe_Call(cudaMalloc((void **)&(fuse2_d_2), sizeof(int) * max_fuse_size2));
    Safe_Call(cudaMemcpy(fuse2_d_2, fuse2[1], sizeof(int) * fuse_list2[1].size(), cudaMemcpyHostToDevice));

    std::cout << "GPU memory allocated for network 2, executing first 21 layers..." << std::endl;

    for (int l = 0; l < 23; ++l)
    {
      performPre(
          l,                          
          kernel_stream,              
          memory_stream,              
          this_round_batch2,        
          last_feature2,             
          neuron2,                   
          bias2,                    
          A2_d,                       
          C2_d,                     
          AH2_d,                    
          B2,                        
          index2,                   
          col_ptr2,                 
          fuse2,                     
          weight2,                   
          row_idx2,                   
          ptr2,                      
          fuse_list2,                
          active2,                  
          active2_d,                 
          category2,                  
          old_to_new_map2,            
          category2_d,                
          B2_d_1, B2_d_2,             
          index2_d_1, index2_d_2,     
          col_ptr2_d_1, col_ptr2_d_2,
          fuse2_d_1, fuse2_d_2,       
          blocksize_c,                
          blocksize_r,                
          mili_iter_time,             
          mili_all_time,              
          layer,                      
          start,                      
          stop                       
      );
    }
    for (int l = 23; l < layer; ++l)
    {

      Safe_Call(cudaStreamSynchronize(memory_stream));

      half *B1_d = (l % 2 == 0) ? B1_d_1 : B1_d_2;
      int *index1_d = (l % 2 == 0) ? index1_d_1 : index1_d_2;
      int *col_ptr1_d = (l % 2 == 0) ? col_ptr1_d_1 : col_ptr1_d_2;
      int *fuse1_d = (l % 2 == 0) ? fuse1_d_1 : fuse1_d_2;

      half *B2_d = (l % 2 == 0) ? B2_d_1 : B2_d_2;
      int *index2_d = (l % 2 == 0) ? index2_d_1 : index2_d_2;
      int *col_ptr2_d = (l % 2 == 0) ? col_ptr2_d_1 : col_ptr2_d_2;
      int *fuse2_d = (l % 2 == 0) ? fuse2_d_1 : fuse2_d_2;

      if (l + 1 < layer)
      {
        if (l % 2 == 1)
        {
          // prefetch spdnn1
          Safe_Call(cudaMemcpyAsync(B1_d_1, B1[l + 1],
                                    sizeof(half) * weight1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index1_d_1, index1[l + 1],
                                    sizeof(int) * row_idx1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr1_d_1, col_ptr1[l + 1],
                                    sizeof(int) * ptr1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse1_d_1, fuse1[l + 1],
                                    sizeof(int) * fuse_list1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));

          // prefetch spdnn2
          Safe_Call(cudaMemcpyAsync(B2_d_1, B2[l + 1],
                                    sizeof(half) * weight2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index2_d_1, index2[l + 1],
                                    sizeof(int) * row_idx2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr2_d_1, col_ptr2[l + 1],
                                    sizeof(int) * ptr2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse2_d_1, fuse2[l + 1],
                                    sizeof(int) * fuse_list2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
        }
        else
        {

          Safe_Call(cudaMemcpyAsync(B1_d_2, B1[l + 1],
                                    sizeof(half) * weight1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index1_d_2, index1[l + 1],
                                    sizeof(int) * row_idx1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr1_d_2, col_ptr1[l + 1],
                                    sizeof(int) * ptr1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse1_d_2, fuse1[l + 1],
                                    sizeof(int) * fuse_list1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));

          Safe_Call(cudaMemcpyAsync(B2_d_2, B2[l + 1],
                                    sizeof(half) * weight2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index2_d_2, index2[l + 1],
                                    sizeof(int) * row_idx2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr2_d_2, col_ptr2[l + 1],
                                    sizeof(int) * ptr2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse2_d_2, fuse2[l + 1],
                                    sizeof(int) * fuse_list2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
        }
      }

      cudaStream_t stream = kernel_stream;
      Safe_Call(cudaMemsetAsync(active1_d, 0, sizeof(int) * this_round_batch1, stream));
      Safe_Call(cudaMemsetAsync(active2_d, 0, sizeof(int) * this_round_batch2, stream));


      blocksize_x = 4 * WARP_SIZE;
      gridsize_x = (neuron1 * (uint64_t)this_round_batch1 + blocksize_x - 1) / blocksize_x;
      floatTohalf2<<<gridsize_x, blocksize_x, 0, stream>>>(A1_d, AH1_d, neuron1, this_round_batch1);


      gridsize_x = (neuron2 * (uint64_t)this_round_batch2 + blocksize_x - 1) / blocksize_x;
      floatTohalf2<<<gridsize_x, blocksize_x, 0, stream>>>(A2_d, AH2_d, neuron2, this_round_batch2);
      Safe_Call(cudaDeviceSynchronize());

      // set grid & block
      blocksize_x = 4 * WARP_SIZE * 2;
      blocksize_y = 1;

      gridsize_x = ((ptr1[l].size() - 1) * ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))) + ((ptr2[l].size() - 1) * ((this_round_batch2 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
      dim3 gridDim_f(gridsize_x, gridsize_y);
      dim3 blockDim_f(blocksize_x, blocksize_y);
      bool isCo = evaluateConfig(l, prop, blockDim_f, dim3((ptr1[l].size() - 1), ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))));
      if (isCo)
      {
        Safe_Call(cudaEventRecord(start, kernel_stream));
        cospdnn_double<<<gridDim_f, blockDim_f, 0, stream>>>(

            AH1_d, B1_d, C1_d, fuse1_d, col_ptr1_d, index1_d, (ptr1[l].size() - 1), bias1,
            this_round_batch1, neuron1,

            AH2_d, B2_d, C2_d, fuse2_d, col_ptr2_d, index2_d, (ptr2[l].size() - 1), bias2,
            this_round_batch2, neuron2,
            blocksize_c, blocksize_r, ((ptr1[l].size() - 1) * ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))));
        Safe_Call(cudaDeviceSynchronize());
        Safe_Call(cudaEventRecord(stop, stream));
        Safe_Call(cudaEventSynchronize(stop));
        Safe_Call(cudaEventElapsedTime(&mili_iter_time, start, stop));
      }
      else
      {
        dim3 gridDim_f1((ptr1[l].size() - 1), ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
        dim3 gridDim_f2((ptr2[l].size() - 1), ((this_round_batch2 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
        Safe_Call(cudaEventRecord(start, kernel_stream));
        cospdnn_single<<<gridDim_f1, blockDim_f, 0, stream>>>(AH1_d, B1_d, C1_d, fuse1_d, col_ptr1_d, index1_d, bias1, this_round_batch1, neuron1, blocksize_c, blocksize_r);
        cospdnn_single<<<gridDim_f2, blockDim_f, 0, stream>>>(AH2_d, B2_d, C2_d, fuse2_d, col_ptr2_d, index2_d, bias2, this_round_batch2, neuron2, blocksize_c, blocksize_r);

        Safe_Call(cudaDeviceSynchronize());
        Safe_Call(cudaEventRecord(stop, stream));
        Safe_Call(cudaEventSynchronize(stop));
        Safe_Call(cudaEventElapsedTime(&mili_iter_time, start, stop));
      }
      mili_all_time += mili_iter_time;

      gridsize_x = (neuron1 * (uint64_t)this_round_batch1 + blocksize_x - 1) / blocksize_x;
      check_active_f_col<<<gridsize_x, blocksize_x, 0, stream>>>(C1_d, active1_d, neuron1, this_round_batch1);

      gridsize_x = (neuron2 * (uint64_t)this_round_batch2 + blocksize_x - 1) / blocksize_x;
      check_active_f_col<<<gridsize_x, blocksize_x, 0, stream>>>(C2_d, active2_d, neuron2, this_round_batch2);
      Safe_Call(cudaDeviceSynchronize());
      if (l <= 21)
      {
        Safe_Call(cudaMemcpy(active1, active1_d, sizeof(int) * this_round_batch1, cudaMemcpyDeviceToHost));
        Safe_Call(cudaMemcpy(active2, active2_d, sizeof(int) * this_round_batch2, cudaMemcpyDeviceToHost));
      }
      else
      {
        Safe_Call(cudaMemcpyAsync(active1, active1_d, sizeof(int) * this_round_batch1, cudaMemcpyDeviceToHost, stream));
        Safe_Call(cudaMemcpyAsync(active2, active2_d, sizeof(int) * this_round_batch2, cudaMemcpyDeviceToHost, stream));
      }

      if (l <= 21)
      {
        int feature1 = 0;
        for (int k = 0; k < this_round_batch1; k++)
        {
          if (active1[k] > 0)
          {
            category1[feature1] = k;
            old_to_new_map1[feature1] = old_to_new_map1[k];
            feature1++;
          }
        }
#ifdef PRINT_TEST
        std::cout << "SpDNN 1 layer " << l << ", feature = " << feature1 << std::endl;
#endif
        Safe_Call(cudaMemcpy(category1_d, category1, sizeof(int) * feature1, cudaMemcpyHostToDevice));
        int padding1 = (feature1 + TILE_DIM - 1) / TILE_DIM * TILE_DIM - feature1;
        last_feature1 = this_round_batch1;
        this_round_batch1 = feature1 + padding1;
        gridsize_x = (feature1 * (uint64_t)neuron1 + blocksize_x - 1) / blocksize_x;
        Safe_Call(cudaMemset(A1_d, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
        matrix_reshape<<<gridsize_x, blocksize_x, 0, stream>>>(C1_d, A1_d, category1_d, neuron1,
                                                               last_feature1, feature1, this_round_batch1);
        Safe_Call(cudaDeviceSynchronize());
      }
      else
      {
        float *tmp = C1_d;
        C1_d = A1_d;
        A1_d = tmp;
        if (l == 22)
        {
          Safe_Call(cudaFree(C1_d));
          Safe_Call(cudaFree(AH1_d));
          Safe_Call(cudaMalloc((void **)&C1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMemset(C1_d, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMalloc((void **)&AH1_d, sizeof(half) * (uint64_t)neuron1 * this_round_batch1 * 2));

          float *AT_d;
          Safe_Call(cudaMalloc((void **)&AT_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMemcpy(AT_d, A1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1, cudaMemcpyDeviceToDevice));
          Safe_Call(cudaFree(A1_d));
          A1_d = AT_d;
        }
      }


      if (l <= 21)
      {
        int feature2 = 0;
        for (int k = 0; k < this_round_batch2; k++)
        {
          if (active2[k] > 0)
          {
            category2[feature2] = k;
            old_to_new_map2[feature2] = old_to_new_map2[k];
            feature2++;
          }
        }
#ifdef PRINT_TEST
        std::cout << "SpDNN 2 layer " << l << ", feature = " << feature2 << std::endl;
#endif
        Safe_Call(cudaMemcpy(category2_d, category2, sizeof(int) * feature2, cudaMemcpyHostToDevice));
        int padding2 = (feature2 + TILE_DIM - 1) / TILE_DIM * TILE_DIM - feature2;
        last_feature2 = this_round_batch2;
        this_round_batch2 = feature2 + padding2;
        gridsize_x = (feature2 * (uint64_t)neuron2 + blocksize_x - 1) / blocksize_x;
        Safe_Call(cudaMemset(A2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
        matrix_reshape<<<gridsize_x, blocksize_x, 0, stream>>>(C2_d, A2_d, category2_d, neuron2,
                                                               last_feature2, feature2, this_round_batch2);
        Safe_Call(cudaDeviceSynchronize());
      }
      else
      {
        float *tmp = C2_d;
        C2_d = A2_d;
        A2_d = tmp;
        if (l == 22)
        {
          Safe_Call(cudaFree(C2_d));
          Safe_Call(cudaFree(AH2_d));
          Safe_Call(cudaMalloc((void **)&C2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
          Safe_Call(cudaMemset(C2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));

          Safe_Call(cudaMalloc((void **)&AH2_d, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));

          float *AT_d;
          Safe_Call(cudaMalloc((void **)&AT_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
          Safe_Call(cudaMemcpy(AT_d, A2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2, cudaMemcpyDeviceToDevice));
          Safe_Call(cudaFree(A2_d));
          A2_d = AT_d;
        }
      }


      memset(active1, 0, sizeof(int) * this_round_batch1);
      memset(active2, 0, sizeof(int) * this_round_batch2);

#ifdef PRINT_TEST
      std::cout << "layer " << l << ", batch1 = " << this_round_batch1
                << ", batch2 = " << this_round_batch2 << std::endl;
      std::cout << "Layer " << l << " exec Time = " << mili_iter_time << "ms" << std::endl;
#endif
    }
  }
  else
  {
    for (int l = 0; l < layer; ++l)
    {

      Safe_Call(cudaStreamSynchronize(memory_stream));

      half *B1_d = (l % 2 == 0) ? B1_d_1 : B1_d_2;
      int *index1_d = (l % 2 == 0) ? index1_d_1 : index1_d_2;
      int *col_ptr1_d = (l % 2 == 0) ? col_ptr1_d_1 : col_ptr1_d_2;
      int *fuse1_d = (l % 2 == 0) ? fuse1_d_1 : fuse1_d_2;

      half *B2_d = (l % 2 == 0) ? B2_d_1 : B2_d_2;
      int *index2_d = (l % 2 == 0) ? index2_d_1 : index2_d_2;
      int *col_ptr2_d = (l % 2 == 0) ? col_ptr2_d_1 : col_ptr2_d_2;
      int *fuse2_d = (l % 2 == 0) ? fuse2_d_1 : fuse2_d_2;


      if (l + 1 < layer)
      {
        if (l % 2 == 1)
        {

          Safe_Call(cudaMemcpyAsync(B1_d_1, B1[l + 1],
                                    sizeof(half) * weight1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index1_d_1, index1[l + 1],
                                    sizeof(int) * row_idx1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr1_d_1, col_ptr1[l + 1],
                                    sizeof(int) * ptr1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse1_d_1, fuse1[l + 1],
                                    sizeof(int) * fuse_list1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));

          Safe_Call(cudaMemcpyAsync(B2_d_1, B2[l + 1],
                                    sizeof(half) * weight2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index2_d_1, index2[l + 1],
                                    sizeof(int) * row_idx2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr2_d_1, col_ptr2[l + 1],
                                    sizeof(int) * ptr2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse2_d_1, fuse2[l + 1],
                                    sizeof(int) * fuse_list2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
        }
        else
        {

          Safe_Call(cudaMemcpyAsync(B1_d_2, B1[l + 1],
                                    sizeof(half) * weight1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index1_d_2, index1[l + 1],
                                    sizeof(int) * row_idx1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr1_d_2, col_ptr1[l + 1],
                                    sizeof(int) * ptr1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse1_d_2, fuse1[l + 1],
                                    sizeof(int) * fuse_list1[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));


          Safe_Call(cudaMemcpyAsync(B2_d_2, B2[l + 1],
                                    sizeof(half) * weight2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(index2_d_2, index2[l + 1],
                                    sizeof(int) * row_idx2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(col_ptr2_d_2, col_ptr2[l + 1],
                                    sizeof(int) * ptr2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
          Safe_Call(cudaMemcpyAsync(fuse2_d_2, fuse2[l + 1],
                                    sizeof(int) * fuse_list2[l + 1].size(),
                                    cudaMemcpyHostToDevice, memory_stream));
        }
      }

      cudaStream_t stream = kernel_stream;
      Safe_Call(cudaMemsetAsync(active1_d, 0, sizeof(int) * this_round_batch1, stream));
      Safe_Call(cudaMemsetAsync(active2_d, 0, sizeof(int) * this_round_batch2, stream));

      blocksize_x = 4 * WARP_SIZE;
      gridsize_x = (neuron1 * (uint64_t)this_round_batch1 + blocksize_x - 1) / blocksize_x;
      floatTohalf2<<<gridsize_x, blocksize_x, 0, stream>>>(A1_d, AH1_d, neuron1, this_round_batch1);


      gridsize_x = (neuron2 * (uint64_t)this_round_batch2 + blocksize_x - 1) / blocksize_x;
      floatTohalf2<<<gridsize_x, blocksize_x, 0, stream>>>(A2_d, AH2_d, neuron2, this_round_batch2);
      Safe_Call(cudaDeviceSynchronize());

      blocksize_x = 4 * WARP_SIZE * 2;
      blocksize_y = 1;

      gridsize_x = ((ptr1[l].size() - 1) * ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))) + ((ptr2[l].size() - 1) * ((this_round_batch2 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
      dim3 gridDim_f(gridsize_x, gridsize_y);
      dim3 blockDim_f(blocksize_x, blocksize_y);
      bool isCo = evaluateConfig(l, prop, blockDim_f, dim3((ptr1[l].size() - 1), ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))));
      if (isCo)
      {
        Safe_Call(cudaEventRecord(start, kernel_stream));
        cospdnn_double<<<gridDim_f, blockDim_f, 0, stream>>>(

            AH1_d, B1_d, C1_d, fuse1_d, col_ptr1_d, index1_d, (ptr1[l].size() - 1), bias1,
            this_round_batch1, neuron1,

            AH2_d, B2_d, C2_d, fuse2_d, col_ptr2_d, index2_d, (ptr2[l].size() - 1), bias2,
            this_round_batch2, neuron2,
            blocksize_c, blocksize_r, ((ptr1[l].size() - 1) * ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM))));
        Safe_Call(cudaDeviceSynchronize());
        Safe_Call(cudaEventRecord(stop, stream));
        Safe_Call(cudaEventSynchronize(stop));
        Safe_Call(cudaEventElapsedTime(&mili_iter_time, start, stop));
      }
      else
      {
        dim3 gridDim_f1((ptr1[l].size() - 1), ((this_round_batch1 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
        dim3 gridDim_f2((ptr2[l].size() - 1), ((this_round_batch2 + blocksize_y * TILE_DIM - 1) / (blocksize_y * TILE_DIM)));
        Safe_Call(cudaEventRecord(start, kernel_stream));
        cospdnn_single<<<gridDim_f1, blockDim_f, 0, stream>>>(AH1_d, B1_d, C1_d, fuse1_d, col_ptr1_d, index1_d, bias1, this_round_batch1, neuron1, blocksize_c, blocksize_r);
        cospdnn_single<<<gridDim_f2, blockDim_f, 0, stream>>>(AH2_d, B2_d, C2_d, fuse2_d, col_ptr2_d, index2_d, bias2, this_round_batch2, neuron2, blocksize_c, blocksize_r);

        Safe_Call(cudaDeviceSynchronize());
        Safe_Call(cudaEventRecord(stop, stream));
        Safe_Call(cudaEventSynchronize(stop));
        Safe_Call(cudaEventElapsedTime(&mili_iter_time, start, stop));
      }
      mili_all_time += mili_iter_time;

      gridsize_x = (neuron1 * (uint64_t)this_round_batch1 + blocksize_x - 1) / blocksize_x;
      check_active_f_col<<<gridsize_x, blocksize_x, 0, stream>>>(C1_d, active1_d, neuron1, this_round_batch1);

      gridsize_x = (neuron2 * (uint64_t)this_round_batch2 + blocksize_x - 1) / blocksize_x;
      check_active_f_col<<<gridsize_x, blocksize_x, 0, stream>>>(C2_d, active2_d, neuron2, this_round_batch2);
      Safe_Call(cudaDeviceSynchronize());
      if (l <= 21)
      {
        Safe_Call(cudaMemcpy(active1, active1_d, sizeof(int) * this_round_batch1, cudaMemcpyDeviceToHost));
        Safe_Call(cudaMemcpy(active2, active2_d, sizeof(int) * this_round_batch2, cudaMemcpyDeviceToHost));
      }
      else
      {
        Safe_Call(cudaMemcpyAsync(active1, active1_d, sizeof(int) * this_round_batch1, cudaMemcpyDeviceToHost, stream));
        Safe_Call(cudaMemcpyAsync(active2, active2_d, sizeof(int) * this_round_batch2, cudaMemcpyDeviceToHost, stream));
      }

      if (l <= 21)
      {
        int feature1 = 0;
        for (int k = 0; k < this_round_batch1; k++)
        {
          if (active1[k] > 0)
          {
            category1[feature1] = k;
            old_to_new_map1[feature1] = old_to_new_map1[k];
            feature1++;
          }
        }
#ifdef PRINT_TEST
        std::cout << "SpDNN 1 layer " << l << ", feature = " << feature1 << std::endl;
#endif
        Safe_Call(cudaMemcpy(category1_d, category1, sizeof(int) * feature1, cudaMemcpyHostToDevice));
        int padding1 = (feature1 + TILE_DIM - 1) / TILE_DIM * TILE_DIM - feature1;
        last_feature1 = this_round_batch1;
        this_round_batch1 = feature1 + padding1;
        gridsize_x = (feature1 * (uint64_t)neuron1 + blocksize_x - 1) / blocksize_x;
        Safe_Call(cudaMemset(A1_d, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
        matrix_reshape<<<gridsize_x, blocksize_x, 0, stream>>>(C1_d, A1_d, category1_d, neuron1,
                                                               last_feature1, feature1, this_round_batch1);
        Safe_Call(cudaDeviceSynchronize());
      }
      else
      {
        float *tmp = C1_d;
        C1_d = A1_d;
        A1_d = tmp;
        if (l == 22)
        {
          Safe_Call(cudaFree(C1_d));
          Safe_Call(cudaFree(AH1_d));
          Safe_Call(cudaMalloc((void **)&C1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMemset(C1_d, 0, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMalloc((void **)&AH1_d, sizeof(half) * (uint64_t)neuron1 * this_round_batch1 * 2));

          float *AT_d;
          Safe_Call(cudaMalloc((void **)&AT_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1));
          Safe_Call(cudaMemcpy(AT_d, A1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1, cudaMemcpyDeviceToDevice));
          Safe_Call(cudaFree(A1_d));
          A1_d = AT_d;
        }
      }

      if (l <= 21)
      {
        int feature2 = 0;
        for (int k = 0; k < this_round_batch2; k++)
        {
          if (active2[k] > 0)
          {
            category2[feature2] = k;
            old_to_new_map2[feature2] = old_to_new_map2[k];
            feature2++;
          }
        }
#ifdef PRINT_TEST
        std::cout << "SpDNN 2 layer " << l << ", feature = " << feature2 << std::endl;
#endif
        Safe_Call(cudaMemcpy(category2_d, category2, sizeof(int) * feature2, cudaMemcpyHostToDevice));
        int padding2 = (feature2 + TILE_DIM - 1) / TILE_DIM * TILE_DIM - feature2;
        last_feature2 = this_round_batch2;
        this_round_batch2 = feature2 + padding2;
        gridsize_x = (feature2 * (uint64_t)neuron2 + blocksize_x - 1) / blocksize_x;
        Safe_Call(cudaMemset(A2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
        matrix_reshape<<<gridsize_x, blocksize_x, 0, stream>>>(C2_d, A2_d, category2_d, neuron2,
                                                               last_feature2, feature2, this_round_batch2);
        Safe_Call(cudaDeviceSynchronize());
      }
      else
      {
        float *tmp = C2_d;
        C2_d = A2_d;
        A2_d = tmp;
        if (l == 22)
        {
          Safe_Call(cudaFree(C2_d));
          Safe_Call(cudaFree(AH2_d));
          Safe_Call(cudaMalloc((void **)&C2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
          Safe_Call(cudaMemset(C2_d, 0, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));

          Safe_Call(cudaMalloc((void **)&AH2_d, sizeof(half) * (uint64_t)neuron2 * this_round_batch2 * 2));

          float *AT_d;
          Safe_Call(cudaMalloc((void **)&AT_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2));
          Safe_Call(cudaMemcpy(AT_d, A2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2, cudaMemcpyDeviceToDevice));
          Safe_Call(cudaFree(A2_d));
          A2_d = AT_d;
        }
      }


      memset(active1, 0, sizeof(int) * this_round_batch1);
      memset(active2, 0, sizeof(int) * this_round_batch2);

#ifdef PRINT_TEST
      std::cout << "layer " << l << ", batch1 = " << this_round_batch1
                << ", batch2 = " << this_round_batch2 << std::endl;
      std::cout << "Layer " << l << " exec Time = " << mili_iter_time << "ms" << std::endl;
#endif
    }
  }

  Safe_Call(cudaMemcpy(C1, C1_d, sizeof(float) * (uint64_t)neuron1 * this_round_batch1, cudaMemcpyDeviceToHost));
  Safe_Call(cudaMemcpy(C2, C2_d, sizeof(float) * (uint64_t)neuron2 * this_round_batch2, cudaMemcpyDeviceToHost));
  std::cout << "Total Kernel Exec Time = " << mili_all_time << "ms" << std::endl;

  free(A1);
  free(A2);
  free(C1);
  free(C2);

  for (int i = 0; i < layer; i++)
  {
    free(B1[i]);
    free(B2[i]);
    free(index1[i]);
    free(index2[i]);
    free(col_ptr1[i]);
    free(col_ptr2[i]);
    free(fuse1[i]);
    free(fuse2[i]);
  }
  free(B1);
  free(B2);
  free(index1);
  free(index2);
  free(col_ptr1);
  free(col_ptr2);
  free(fuse1);
  free(fuse2);

  free(category1);
  free(category2);
  free(active1);
  free(active2);
  free(old_to_new_map1);
  free(old_to_new_map2);

  Safe_Call(cudaFree(A1_d));
  Safe_Call(cudaFree(A2_d));

  Safe_Call(cudaFree(AH1_d));
  Safe_Call(cudaFree(AH2_d));

  Safe_Call(cudaFree(C1_d));
  Safe_Call(cudaFree(C2_d));
  Safe_Call(cudaFree(active1_d));
  Safe_Call(cudaFree(active2_d));
  Safe_Call(cudaFree(category1_d));
  Safe_Call(cudaFree(category2_d));
  Safe_Call(cudaFree(old_to_new_map1_d));
  Safe_Call(cudaFree(old_to_new_map2_d));

  Safe_Call(cudaFree(B1_d_1));
  Safe_Call(cudaFree(B1_d_2));
  Safe_Call(cudaFree(B2_d_1));
  Safe_Call(cudaFree(B2_d_2));
  Safe_Call(cudaFree(index1_d_1));
  Safe_Call(cudaFree(index1_d_2));
  Safe_Call(cudaFree(index2_d_1));
  Safe_Call(cudaFree(index2_d_2));
  Safe_Call(cudaFree(col_ptr1_d_1));
  Safe_Call(cudaFree(col_ptr1_d_2));
  Safe_Call(cudaFree(col_ptr2_d_1));
  Safe_Call(cudaFree(col_ptr2_d_2));
  Safe_Call(cudaFree(fuse1_d_1));
  Safe_Call(cudaFree(fuse1_d_2));
  Safe_Call(cudaFree(fuse2_d_1));
  Safe_Call(cudaFree(fuse2_d_2));

  Safe_Call(cudaEventDestroy(start));
  Safe_Call(cudaEventDestroy(stop));
  Safe_Call(cudaStreamDestroy(kernel_stream));
  Safe_Call(cudaStreamDestroy(memory_stream));

  return;
}