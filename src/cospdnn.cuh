#ifndef GC_TEST_GC_GPU_CUH_
#define GC_TEST_GC_GPU_CUH_

#pragma once
#include "./kernels/kernel.cuh"
#include "./gpu_lib/gpu_env.h"
#include "./gpu_lib/gpu_runtime.h"
#include <vector>
using namespace ftxj;

/**
 * @parameter: batch is the total batch
 **/
void test_graph_challenge_tc(std::vector<std::vector<float>> &input,
                             std::vector<std::vector<float>> &weight,
                             std::vector<std::vector<int>> &row_idx,
                             std::vector<std::vector<int>> &ptr,
                             std::vector<std::vector<int>> &fuse_list,
                             int batch, int neuron, float bias,
                             int blocksize_c, int blocksize_r,
                             int gpu_id, int gpu_num);
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
                                int gpu_id, int gpu_num);
#endif