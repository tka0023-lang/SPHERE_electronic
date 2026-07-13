#!/bin/sh
clang++ -mcmodel=large -Wall -O3 -v -mllvm -inline-threshold=10000 ./main_mod.cpp -o main