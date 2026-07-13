#!/bin/sh
clang++ -mcmodel=large -Wall -O3 -lzstd -v -mllvm -inline-threshold=10000 ./main_mod_.cpp -o main