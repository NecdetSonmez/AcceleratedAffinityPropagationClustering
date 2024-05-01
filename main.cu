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

#define USE_CPU false
#define USE_GPU true
#define USE_GPU_V2 true

int main()
{
    // Generate points
    Points pointsObject;
    std::vector<std::tuple<float, float>> centers = {{0.0, 0.0}, {5.0, 5.0}};
    pointsObject.generatePoints(centers, POINT_COUNT, 0.25);
    float* points = pointsObject.getPoints();

#if USE_CPU
    // Cluster with CPU and measure time
    ApcCpu cpuClusterer(points, POINT_COUNT, POINT_DIM, 0.5);

    auto startTimeCpu = std::chrono::high_resolution_clock::now();
    cpuClusterer.cluster();
    auto endTimeCpu = std::chrono::high_resolution_clock::now();
    auto durationCpu = std::chrono::duration_cast<std::chrono::microseconds>(endTimeCpu - startTimeCpu).count();
    std::cout << "CPU Time taken: " << durationCpu << " microseconds." << std::endl;
#endif

#if USE_GPU
    // Cluster with GPU and measure time
    ApcGpu gpuClusterer(points, POINT_COUNT, POINT_DIM, 0.5);

    auto startTimeGpu = std::chrono::high_resolution_clock::now();
    gpuClusterer.cluster(25);
    auto endTimeGpu = std::chrono::high_resolution_clock::now();
    auto durationGpu = std::chrono::duration_cast<std::chrono::microseconds>(endTimeGpu - startTimeGpu).count();
    std::cout << "GPU Time taken: " << durationGpu << " microseconds." << std::endl;
#endif

#if USE_GPU_V2
    // Cluster with GPU and measure time
    ApcGpuV2 gpuClustererV2(points, POINT_COUNT, POINT_DIM, 0.5);

    auto startTimeGpuV2 = std::chrono::high_resolution_clock::now();
    gpuClustererV2.cluster(25);
    auto endTimeGpuV2 = std::chrono::high_resolution_clock::now();
    auto durationGpuV2 = std::chrono::duration_cast<std::chrono::microseconds>(endTimeGpuV2 - startTimeGpuV2).count();
    std::cout << "GPU Time taken: " << durationGpuV2 << " microseconds." << std::endl;
#endif

    return 0;
}