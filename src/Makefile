FLAG  :=  -std=c++11 -Xcompiler -fopenmp -gencode arch=compute_80,code=sm_80 -DPRINT_TEST -lineinfo

TARGET1=cospdnn
$(TARGET1):
	nvcc ${FLAG} -o $(TARGET1).out $(TARGET1).cpp ./$(TARGET1).cu ./kernels/kernel.cu
TARGET2=cospdnn_double
$(TARGET2):
	nvcc ${FLAG} -o $(TARGET2).out $(TARGET2).cpp ./$(TARGET2).cu ./kernels/kernel.cu
clean:
	rm cospdnn*.out