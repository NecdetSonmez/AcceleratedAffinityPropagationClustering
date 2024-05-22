#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Similarity
void launchKernel_updateSimilarity(int blockCount, int threadCount, float* points, float* similarity, int pointCount, int pointDimension);
__global__ void Kernel_updateSimilarity(float* points, float* similarity, int pointCount, int pointDimension);

// Slow Responsibility for V1
void launchKernel_updateResponsibility(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount);
__global__ void Kernel_updateResponsibility(float* similarity, float* responsibility, float* availability, int pointCount);

// Availability
void launchKernel_updateAvailability(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount);
__global__ void Kernel_updateAvailability(float* similarity, float* responsibility, float* availability, int pointCount);

// Exemplars
void launchKernel_extractExemplars(int blockCount, int threadCount, float* responsibility, float* availability, int pointCount, char* exemplars);
__global__ void Kernel_extractExemplars(float* responsibility, float* availability, int pointCount, char* exemplars);

// Point labeling
void launchKernel_labelPoints(int blockCount, int threadCount, float* similarity, char* exemplars, int* pointLabels, int pointCount);
__global__ void Kernel_labelPoints(float* similarity, char* exemplars, int* pointLabels, int pointCount);

// Max Finder for V2
void launchKernel_findMaxForResponsibility(int blockCount, int threadCount, float* similarity, float* availability, float* maxValues, int pointCount);
__global__ void Kernel_findMaxForResponsibility(float* similarity, float* availability, float* maxValues, int pointCount);

// Responsibility with max for V2
void launchKernel_updateResponsibilityWithMax(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float* maxValues);
__global__ void Kernel_updateResponsibilityWithMax(float* similarity, float* responsibility, float* availability, int pointCount, float* maxValues);
