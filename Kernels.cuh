#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Similarity
void launchKernel_updateSimilarity(int blockCount, int threadCount, float* points, float* similarity, int pointCount, int pointDimension);
__global__ void Kernel_updateSimilarity(float* points, float* similarity, int pointCount, int pointDimension);

// Responsibility
void launchKernel_updateResponsibility(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor);
__global__ void Kernel_updateResponsibility(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor);

// Availability
void launchKernel_updateAvailability(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor);
__global__ void Kernel_updateAvailability(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor);

// Point labeling
void launchKernel_extractExemplars(int blockCount, int threadCount, float* responsibility, float* availability, int pointCount);
__global__ void Kernel_extractExemplars(float* responsibility, float* availability, int pointCount);