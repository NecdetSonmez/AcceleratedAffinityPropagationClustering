#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <chrono>

#include "ApcCpu.hpp"
#include "ApcGpu.hpp"
#include "ApcGpuV2.hpp"
#include "Points.hpp"

// Parameters
#define GENERATE_POINTS true // NOTE: all of the parameters below still need to be set irregardless of the value of this variable.
#define POINT_COUNT 250 // ALWAYS set this to the correct point count.
#define POINT_DIM 2 // ALWAYS set this to the correct point dimension.
#define FILENAME "input.txt" // Only used if GENERATE_POINTS is false. Has 100 2D points, given as an example. 

#define ITERATION_COUNT 100 // Not recommended to go over 100, time taken increases to impractical levels.

#define USE_GPU_V2 true // Chosen implementation for the paper, the most efficient.
#define USE_CPU true // NOTE: Cpu execution takes a VERY long time.
#define USE_GPU true // Naive implementation for comparison.

int main()
{
    Points pointsObject;
#if GENERATE_POINTS
    // Generate points
    //std::vector<std::vector<float>> centers = {{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}}; // For 3D, set POINT_DIM to 3 before testing.
    std::vector<std::vector<float>> centers = {{0.0f, 0.0f}, {5.0f, 5.0f}}; // For 2D, set POINT_DIM to 2 before testing.
    pointsObject.generatePoints(centers, POINT_COUNT, POINT_DIM);
#else
    // Load points from file
    pointsObject.loadFromFile(FILENAME);
#endif
    float* points = pointsObject.getPoints();

#if USE_GPU
    ApcGpu gpuClustererV1(points, POINT_COUNT, POINT_DIM);
    gpuClustererV1.cluster(ITERATION_COUNT);
#endif

#if USE_GPU_V2
    ApcGpuV2 gpuClustererV2(points, POINT_COUNT, POINT_DIM);
    gpuClustererV2.cluster(ITERATION_COUNT);
#endif

#if USE_CPU
    ApcCpu cpuClusterer(points, POINT_COUNT, POINT_DIM);
    cpuClusterer.cluster(ITERATION_COUNT);
#endif

    return 0;
}