
#include "./utils/matrix.h"
#include "./cospdnn.cuh"
#include "./gpu_lib/gpu_env.h"
#include <vector>
#include <map>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <omp.h>
using namespace ftxj;
void read_fuse(std::vector<std::vector<int>> &input, int neuron, int layer,std::string dir)
{
    std::string input_file_name = dir+"/fuse_matrix/neuron";
    input_file_name += std::to_string(neuron) + "/" + std::to_string(neuron) + "_" + std::to_string(layer) + "_fuse" + ".tsv";
    std::ifstream input_file(input_file_name);
    if (!input_file)
    {
        std::cout << "FILE:" << input_file_name << " does not exists.\n";
        exit(-1);
    }
    int b, n;
    int val;
    long read_num = 0;
    for (int i = 0; i < layer; i++)
    {
        for (int j = 0; j < neuron; j++)
        {
            float temp;
            if (input_file >> temp)
            {
                input[i][j] = temp;
            }
            else
            {
                std::cout << " read error\n";
                exit(0);
            }
        }
    }
    std::cout << "Read fuse success! read_numeber = " << read_num << std::endl;
}
void read_input(std::vector<std::vector<float>> &input, int neuron, int batch,std::string dir)
{
    std::string input_file_name =dir+ "/sparse-images-";
    input_file_name += std::to_string(neuron) + ".tsv";
    std::ifstream input_file(input_file_name);
    if (!input_file)
    {
        std::cout << "FILE:" << input_file_name << " does not exists.\n";
        exit(-1);
    }
    int b, n;
    float val;
    long read_num = 0;
    while (input_file >> b >> n >> val)
    {
        if (b <= batch)
        {
            read_num++;
            input[b - 1][n - 1] = val;
            if (val != 1.00)
            {
                printf("read input %d, %f\n", b, val);
            }
        }
    }
    std::cout << "Read Input success! read_numeber = " << read_num << std::endl;
}
std::string get_preprocessed_weight_file_name(int neuron, int layer,std::string dir)
{
    ///home/data/sparsemat/graphchallenge/22data
    std::string weight_file_dir = dir+"/neuron";
    std::string neuron_str = std::to_string(neuron);
    weight_file_dir += neuron_str + "/" + neuron_str + "-l" + std::to_string(layer + 1) + ".tsv";
    return weight_file_dir;
}
int main(int argc, char *argv[])
{
    if (argc != 7)
    {
        std::cout << "Usage: exe neuron batch layer gpu_num gpu_id dataset_dir" << std::endl;
        return 0;
    }

    int neuron = atoi(argv[1]);
    int batch = atoi(argv[2]);
    int layer = atoi(argv[3]);
    int gpu_num = atoi(argv[4]);
    int dev= atoi(argv[5]);
    std::string dir(argv[6]);
    int blocksize_c = 16;
    int blocksize_r = 16;

    std::map<int, float> bias_map =
        {
            {65536, -0.45},
            {16384, -0.4},
            {4096, -0.35},
            {1024, -0.3}};

    std::map<int, float> type_1 =
        {
            {65536, 12},
            {16384, 10},
            {4096, 8},
            {1024, 6}};

    std::vector<std::vector<float>> input(batch, std::vector<float>(neuron));
    std::vector<std::vector<float>> weight;
    std::vector<std::vector<int>> fuse_list(layer + 1, std::vector<int>(neuron));
    std::vector<std::vector<int>> row_idx; // row_idx in bcsc
    std::vector<std::vector<int>> ptr;     // col_ptr in bcsc

    std::cout << "[BEGIN]..." << std::endl;
    std::cout << "[Batch] :" <<batch<< std::endl;
    std::cout << "[Neuron] :" <<neuron<< std::endl;
    std::cout << "[Layer] :" <<layer<< std::endl;
    read_input(input, neuron, batch,dir);
    std::cout << "Read Input success!" << std::endl;

    read_fuse(fuse_list, neuron, layer,dir);
    std::cout << "Read Fuse success!" << std::endl;

    // read the weight
    for (int l = 0; l < layer; ++l)
    {
        std::string weight_file = get_preprocessed_weight_file_name(neuron, l,dir);
        COOMatrix coo(weight_file, 1, true);
#ifdef PRINT_TEST
        std::cout << "[" << weight_file << "] to COO success!" << std::endl;
#endif
        // generate the bcsrcsc format
        BCSRCSCMatrix weight_bcsr_csc(coo, blocksize_r, blocksize_c);
#ifdef PRINT_TEST
        std::cout << "[" << weight_file << "] BCSR_CSC Generation Success! " << std::endl;
#endif
        // generate the weight and row/col idx
        weight.push_back(weight_bcsr_csc.bcsc_values_);
        row_idx.push_back(weight_bcsr_csc.bcsc_row_index_);
        ptr.push_back(weight_bcsr_csc.bcsc_len_);
    }
    std::cout << "weight len: " << weight.size() << std::endl;
    std::cout << "row_idx len: " << row_idx.size() << std::endl;
    std::cout << "ptr len: " << ptr.size() << std::endl;
    if (gpu_num == 1) // single gpu
    {
        Safe_Call(cudaSetDevice(dev));
        test_graph_challenge_tc(input, weight, row_idx, ptr,
                                fuse_list, batch, neuron, bias_map[neuron],
                                blocksize_c, blocksize_r, 0, 1);
    }
    else // utilizing OpenMP on multiple GPU
    {
        int device_counter;
        // get max gpu num
        Safe_Call(cudaGetDeviceCount(&device_counter));

        if (gpu_num > device_counter)
        {
            std::cout << "Exceeding Maximum GPU Number! Automatically Change to Max GPU Number " << device_counter << std::endl;
            gpu_num = device_counter;
        }

        omp_set_num_threads(gpu_num);

#pragma omp parallel
        {
            // prepare the threads
            unsigned int cpu_thread_id = omp_get_thread_num();
            unsigned int num_cpu_threads = omp_get_num_threads();

            int gpu_id = cpu_thread_id % gpu_num;
            Safe_Call(cudaSetDevice(gpu_id));
            std::cout << "GPU[" << gpu_id << "] " << "[BEGIN]..." << std::endl;

            test_graph_challenge_tc(input, weight, row_idx, ptr,
                                    fuse_list, batch, neuron, bias_map[neuron],
                                    blocksize_c, blocksize_r, gpu_id, gpu_num);

            std::cout << "GPU[" << gpu_id << "] " << "[END]..." << std::endl;
#pragma omp barrier
        }
    }
    std::cout << "[END]..." << std::endl;
    return 0;
}