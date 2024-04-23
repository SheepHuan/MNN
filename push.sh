#!/bin/bash

export DEVICE="172.16.101.223:5555"
export MNNV2Basic_PATH="/root/code/MNN/android_build/MNNV2Basic.out"
export Benchmark_PATH="/root/code/MNN/android_build/benchmark.out"
export MODEL_PATH="/root/code/DeployNN/tmp/fft.mnn"
export MODEL_PATH2="/root/code/DeployNN/tmp/mnn"
export workspace="/data/local/tmp/mnn_240423"
export input="/root/code/MNN/build/input_0.txt"

adb -s $DEVICE shell "mkdir -p $workspace"
adb -s $DEVICE push $MNNV2Basic_PATH $workspace
adb -s $DEVICE push $MODEL_PATH $workspace
adb -s $DEVICE push $MODEL_PATH2 $workspace
adb -s $DEVICE push $Benchmark_PATH $workspace
# adb -s $DEVICE push $MNNV2Basic_PATH $workspace
adb -s $DEVICE push $input $workspace

adb -s $DEVICE push "/root/code/fftw/android/lib/libfftw3f.so" $workspace
adb -s $DEVICE push "/root/code/MNN/android_build/install/lib/libMNN.so" $workspace
adb -s $DEVICE push "/root/code/MNN/android_build/install/lib/libMNN_Express.so" $workspace


# adb -s $DEVICE shell cd $workspace && export LD_LIBRARY_PATH=/data/local/tmp/mnn_240423 && chmod +x MNNV2Basic.out && ./MNNV2Basic.out fft.mnn 1 2 0 1 1 "1x3x512"

# export LD_LIBRARY_PATH=/data/local/tmp/mnn_240423 && chmod +x MNNV2Basic.out && ./MNNV2Basic.out fft.mnn 1 2 0 1 1 "1x3x512"

# export LD_LIBRARY_PATH=/data/local/tmp/mnn_240423 && chmod +x MNNV2Basic.out && ./MNNV2Basic.out mnn/custom_melspectrogram.mnn 100 0 0 1 1 "1920"

# export LD_LIBRARY_PATH=/data/local/tmp/mnn_240423 && chmod +x benchmark.out && ./benchmark.out mnn/ 10 100 0 1 0 

