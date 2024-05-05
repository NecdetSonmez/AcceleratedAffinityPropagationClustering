#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <chrono>

#include "ApcCpu.hpp"
#include "ApcGpu.hpp"
#include "ApcGpuV2.hpp"
#include "Points.hpp"

#define POINT_COUNT 1000
#define POINT_DIM 2
#define POINT_VARIATION 1.0

#define DAMPING_FACTOR 0.5
#define ITERATION_COUNT 100

#define USE_CPU false
#define USE_GPU true
#define USE_GPU_V2 true

int main()
{
    // Generate points
    Points pointsObject;
    std::vector<std::tuple<float, float>> centers = {{0.0, 0.0}, {5.0, 5.0}};
    pointsObject.generatePoints(centers, POINT_COUNT, POINT_VARIATION);
    float* points = pointsObject.getPoints();

#if USE_CPU
    ApcCpu cpuClusterer(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    cpuClusterer.cluster(ITERATION_COUNT);
#endif

#if USE_GPU
    ApcGpu gpuClustererV1(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    gpuClustererV1.cluster(ITERATION_COUNT);
#endif

#if USE_GPU_V2
    ApcGpuV2 gpuClustererV2(points, POINT_COUNT, POINT_DIM, DAMPING_FACTOR);
    gpuClustererV2.cluster(ITERATION_COUNT);
#endif

    return 0;
}