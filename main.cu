#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <chrono>

#include "ApcCpu.hpp"
#include "ApcGpu.hpp"
#include "Points.hpp"

#define POINT_COUNT 100
#define POINT_DIM 2

int main()
{
    // Generate points
    Points pointsObject;
    std::vector<std::tuple<float, float>> centers = {{0.0, 0.0}, {5.0, 5.0}};
    pointsObject.generatePoints(centers, POINT_COUNT, 0.25);
    float* points = pointsObject.getPoints();

    // Cluster with CPU and measure time
    ApcCpu cpuClusterer(points, POINT_COUNT, POINT_DIM, 0.5);

    auto startTime = std::chrono::high_resolution_clock::now();
    cpuClusterer.cluster();
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "CPU Time taken: " << duration << " microseconds." << std::endl;

    // Cluster with GPU and measure time
    ApcGpu gpuClusterer(points, POINT_COUNT, POINT_DIM, 0.5);

    startTime = std::chrono::high_resolution_clock::now();
    gpuClusterer.cluster();
    endTime = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "GPU Time taken: " << duration << " microseconds." << std::endl;

    return 0;
}