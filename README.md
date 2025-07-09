# SpDNN Graph_challenge - CoSpDNN
source code for Sparse Deep Neural Network Graph Challenge (more detail:http://graphchallenge.mit.edu/challenges).


## Preliminaries

Clone the project and download the dataset(https://graphchallenge.s3.amazonaws.com/synthetic/sparsechallenge_2019).

You also need to implement the Similarity-based Matrix Transformation Scheme of Graph-challenge22[1] and put the implemented column index into /dataset/fuse_matrix.

## Get Start (single task)
Compile and run on a single task.
```
cd src/
make cospdnn 
./cospdnn.out 1024 60000 120 1 0 dataset_dir

```
## Get Start (concurrent task)
Compile and run on concurrenct cospdnn version.
```
cd src/
make cospdnn_double 
./cospdnn_double.out 1024 60000 120 1024 60000 120 0 dataset_dir

```
## Reproducing Paper Results
To reproduce the experimental results presented in our paper, run the following script:

For Table I
```bash
cd scripts/
bash run_single.sh
```

For Table II & III
```bash
cd scripts/
bash run_concurrency.sh
```
## References
[1] Y. Sun et al., “Accelerating Sparse Deep Neural Network Inference Using GPU Tensor Cores,” in 2022 IEEE High Performance Extreme Computing Conference (HPEC), Waltham, MA, USA: IEEE, Sep. 2022, pp. 1–7. doi: 10.1109/HPEC55821.2022.9926300.