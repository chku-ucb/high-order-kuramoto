#!/bin/bash
#SBATCH --job-name=km_N
#SBATCH --output=/Users/chku8451/high-order-kuramoto/logs/Large_N_%j.logs
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=72:00:00

export JULIA_NUM_THREADS=20
julia /Users/chku8451/high-order-kuramoto/largeN.jl

