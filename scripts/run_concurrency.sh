#!/bin/bash
export dev=0
export dir=./dataset
for neuron1 in 1024 4096 16384 65536
do 
  for neuron2 in 1024 4096 16384 65536
  do 
    if (( neuron1 <= neuron2 )); then
      for layer in 120 480 1920 
      do
        echo "---------------------------------------------------------------------------------------------------------------" >> ./log/concurrency.log
        ../src/cospdnn_double.out $neuron1 60000 $layer $neuron2 60000 $layer $dev $dir >> ./log/concurrency.log
      done 
    fi
  done
done
