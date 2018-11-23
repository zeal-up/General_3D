#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd ./fps/src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o farthest_point_sample.cu.o fps_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py

cd ../
cd ./query_ball_point/src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o query_ball_point.cu.o query_ball_point_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py

