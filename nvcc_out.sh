#!/bin/bash
cp linear.cpp linear.cu
/usr/local/cuda/bin/nvcc linear.cu -o linear
