
#include "./utils/matrix.h"
// #include "../utils/io.h"
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
    if (argc != 9)
    {
        std::cout << "Usage: exe neuron1 batch1 layer1 neuron2 batch2 layer2 gpu_id dataset_dir" << std::endl;
        return 0;
    }
    //./gc25_co.out 1024 60000 120 1024 60000 120 1
    int neuron1 = atoi(argv[1]);
    int batch1 = atoi(argv[2]);
    int layer1 = atoi(argv[3]);
    int neuron2 = atoi(argv[4]);
    int batch2 = atoi(argv[5]);
    int layer2 = atoi(argv[6]);
    int dev= atoi(argv[7]);
    std::string dir(argv[8]);
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
    //1
    std::vector<std::vector<float>> input1(batch1, std::vector<float>(neuron1));
    std::vector<std::vector<float>> weight1;
    std::vector<std::vector<int>> fuse_list1(layer1 + 1, std::vector<int>(neuron1));
    std::vector<std::vector<int>> row_idx1; // row_idx in bcsc
    std::vector<std::vector<int>> ptr1;     // col_ptr in bcsc
    //2
    std::vector<std::vector<float>> input2(batch2, std::vector<float>(neuron2));
    std::vector<std::vector<float>> weight2;
    std::vector<std::vector<int>> fuse_list2(layer2 + 1, std::vector<int>(neuron2));
    std::vector<std::vector<int>> row_idx2; // row_idx in bcsc
    std::vector<std::vector<int>> ptr2;     // col_ptr in bcsc

    std::cout << "-------------------------------------" << std::endl;
    std::cout << "[BEGIN]..." << std::endl;
    std::cout << "[Batch1] :" <<batch1<< std::endl;
    std::cout << "[Neuron1] :" <<neuron1<< std::endl;
    std::cout << "[Layer1] :" <<layer1<< std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "[Batch2] :" <<batch2<< std::endl;
    std::cout << "[Neuron2] :" <<neuron2<< std::endl;
    std::cout << "[Layer2] :" <<layer2<< std::endl;
    std::cout << "-------------------------------------" << std::endl;
    read_input(input1, neuron1, batch1,dir);
    read_input(input2, neuron2, batch2,dir);
    std::cout << "Read Input success!" << std::endl;

    read_fuse(fuse_list1, neuron1, layer1,dir);
    read_fuse(fuse_list2, neuron2, layer2,dir);
    std::cout << "Read Fuse success!" << std::endl;

    // read the weight
    for (int l = 0; l < layer1; ++l)
    {
        std::string weight_file = get_preprocessed_weight_file_name(neuron1, l,dir);
        COOMatrix coo(weight_file, 1, true);
#ifdef PRINT_TEST
       // std::cout << "[" << weight_file << "] to COO success!" << std::endl;
#endif
        // generate the bcsrcsc format
        BCSRCSCMatrix weight_bcsr_csc(coo, blocksize_r, blocksize_c);
#ifdef PRINT_TEST
       // std::cout << "[" << weight_file << "] BCSR_CSC Generation Success! " << std::endl;
#endif
        // generate the weight and row/col idx
        weight1.push_back(weight_bcsr_csc.bcsc_values_);
        row_idx1.push_back(weight_bcsr_csc.bcsc_row_index_);
        ptr1.push_back(weight_bcsr_csc.bcsc_len_);
    }
    for (int l = 0; l < layer2; ++l)
    {
        std::string weight_file = get_preprocessed_weight_file_name(neuron2, l,dir);
        COOMatrix coo(weight_file, 1, true);
#ifdef PRINT_TEST
        //std::cout << "[" << weight_file << "] to COO success!" << std::endl;
#endif
        // generate the bcsrcsc format
        BCSRCSCMatrix weight_bcsr_csc(coo, blocksize_r, blocksize_c);
#ifdef PRINT_TEST
       // std::cout << "[" << weight_file << "] BCSR_CSC Generation Success! " << std::endl;
#endif
        // generate the weight and row/col idx
        weight2.push_back(weight_bcsr_csc.bcsc_values_);
        row_idx2.push_back(weight_bcsr_csc.bcsc_row_index_);
        ptr2.push_back(weight_bcsr_csc.bcsc_len_);
    }
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "weight1 len: " << weight1.size() << std::endl;
    std::cout << "row_idx1 len: " << row_idx1.size() << std::endl;
    std::cout << "ptr1 len: " << ptr1.size() << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "weight2 len: " << weight2.size() << std::endl;
    std::cout << "row_idx2 len: " << row_idx2.size() << std::endl;
    std::cout << "ptr2 len: " << ptr2.size() << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    Safe_Call(cudaSetDevice(dev));
    
    test_graph_challenge_tc_co(input1, weight1, row_idx1, ptr1,
                                fuse_list1, batch1, neuron1, bias_map[neuron1],
                                input2, weight2, row_idx2, ptr2,
                                fuse_list2, batch2, neuron2, bias_map[neuron2],
                                blocksize_c, blocksize_r, 0, 1);
    std::cout << "[END]..." << std::endl;
    return 0;
}