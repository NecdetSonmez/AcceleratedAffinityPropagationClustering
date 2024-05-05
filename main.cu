#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <chrono>

#include "ApcCpu.hpp"
#include "ApcGpu.hpp"
#include "ApcGpuV2.hpp"
#include "Points.hpp"

#define POINT_COUNT 100
#define POINT_DIM 2
#define POINT_VARIATION 1.0

#define DAMPING_FACTOR 0.5
#define ITERATION_COUNT 100

#define USE_GPU true
#define USE_GPU_V2 true
#define USE_CPU true // NOTE: Cpu execution takes a VERY long time.

int main()
{
    // Generate points
    Points pointsObject;
    //std::vector<std::vector<float>> centers = {{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};
    std::vector<std::vector<float>> centers = {{0.0f, 0.0f}, {5.0f, 5.0f}};
    pointsObject.generatePoints(centers, POINT_COUNT, POINT_VARIATION, POINT_DIM);
    float* points = pointsObject.getPoints();

#if USE_GPU
    ApcGpu gpuClustererV1(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    gpuClustererV1.cluster(ITERATION_COUNT);
#endif

#if USE_GPU_V2
    ApcGpuV2 gpuClustererV2(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    gpuClustererV2.cluster(ITERATION_COUNT);
#endif

#if USE_CPU
    ApcCpu cpuClusterer(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    cpuClusterer.cluster(ITERATION_COUNT);
#endif

    return 0;
}