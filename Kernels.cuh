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

// Exemplars
void launchKernel_extractExemplars(int blockCount, int threadCount, float* responsibility, float* availability, int pointCount, char* exemplars);
__global__ void Kernel_extractExemplars(float* responsibility, float* availability, int pointCount, char* exemplars);

// Point labeling
void launchKernel_labelPoints(int blockCount, int threadCount, float* similarity, char* exemplars, int* pointLabels, int pointCount);
__global__ void Kernel_labelPoints(float* similarity, char* exemplars, int* pointLabels, int pointCount);

// Sums of responsibility
void launchKernel_sumOfResponsibility(int blockCount, int threadCount, float* responsibility, int pointCount, float* sumsOfResponsibility);
__global__ void Kernel_sumOfResponsibility(float* responsibility, int pointCount, float* sumsOfResponsibility);

// Availability with sum data precomputed
void launchKernel_updateAvailabilityWithSum(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* sumsOfResponsibility);
__global__ void Kernel_updateAvailabilityWithSum(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* sumsOfResponsibility);
