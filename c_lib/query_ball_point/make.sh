#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o query_ball_point.cu.o query_ball_point_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
