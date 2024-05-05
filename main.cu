#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <chrono>

#include "ApcCpu.hpp"
#include "ApcGpu.hpp"
#include "ApcGpuV2.hpp"
#include "Points.hpp"

#define GENERATE_POINTS true // NOTE: all of the parameters below still need to be set.
#define POINT_COUNT 100 // ALWAYS set this to the correct point count.
#define POINT_DIM 2 // ALWAYS set this to the correct point dimension.
#define POINT_VARIATION 1.0 // Used if GENERATE_POINTS is true as standard deviation.
#define FILENAME "input.txt" // Has 100 2D points, given as an example. Only used if GENERATE_POINTS is false.

#define DAMPING_FACTOR 0.5
#define ITERATION_COUNT 100

#define USE_GPU true
#define USE_GPU_V2 true
#define USE_CPU true // NOTE: Cpu execution takes a VERY long time.

int main()
{
    Points pointsObject;
#if GENERATE_POINTS
    // Generate points
    //std::vector<std::vector<float>> centers = {{0.0f, 0.0f, 0.0f}, {5.0f, 5.0f, 5.0f}};
    std::vector<std::vector<float>> centers = {{0.0f, 0.0f}, {5.0f, 5.0f}};
    pointsObject.generatePoints(centers, POINT_COUNT, POINT_VARIATION, POINT_DIM);
#else
    // Load points from file
    pointsObject.loadFromFile(FILENAME);
#endif
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