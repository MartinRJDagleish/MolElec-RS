#!/bin/bash

# 1. Install prerequisite software
sudo apt install build-essential g++ gfortran git -y

# 2. Create installation directory
OPENBLAS_DIR=/opt/openblas
sudo mkdir $OPENBLAS_DIR

# 3. Build and install openblas library from source
cd $HOME
git clone https://github.com/xianyi/OpenBLAS

cd $HOME/OpenBLAS
export USE_THREAD=1
export NUM_THREADS=64
export DYNAMIC_ARCH=0
export NO_WARMUP=1
export BUILD_RELAPACK=0
export COMMON_OPT="-O2 -march=native"
export CFLAGS="-O2 -march=native"
export FCOMMON_OPT="-O2 -march=native"
export FCFLAGS="-O2 -march=native"
make -j DYNAMIC_ARCH=0 CC=gcc FC=gfortran HOSTCC=gcc BINARY=64 INTERFACE=64 \
  USE_OPENMP=1 LIBNAMESUFFIX=openmp
sudo make PREFIX=$OPENBLAS_DIR LIBNAMESUFFIX=openmp install

# 4. Install
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/OpenBLAS/include
export CPATH=$CPATH:/opt/OpenBLAS/include
export LIBRARY_PATH=$LIBRARY_PATH:/opt/OpenBLAS/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib


# ulimit -s unlimited
# export MAX_THREADS=4
# gfortran -I/opt/openblas/include -pthread -fopenmp -O3 -funroll-all-loops -fexpensive-optimizations -ftree-vectorize -fprefetch-loop-arrays -floop-parallelize-all \
# -ftree-parallelize-loops=$MAX_THREADS -m64 -Wall example.f90 -o example -L/opt/openblas/lib -lm -lpthread -lgfortran -lopenblas_openmp