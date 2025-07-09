#!/bin/bash
mkdir -p ./log
mkdir -p ../dataset
export dev=0
export dir=../dataset
for neuron1 in 1024 4096 16384 65536
do  
  for layer in 120 480 1920
    do
        echo "---------------------------------------------------------------------------------------------------------------" >> ./log/single.log
        ../src/cospdnn.out $neuron1 60000 $layer 1 $dev $dir >> ./log/single.log
    done 
done
