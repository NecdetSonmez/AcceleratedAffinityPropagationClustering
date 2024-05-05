#include "ApcGpuV2.hpp"
#include "Timer.hpp"
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>
#include <fstream>

#include "Kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ApcGpuV2::ApcGpuV2(float* points, int pointCount, int pointDimension, float dampingFactor)
{
	if (pointCount <= 0)
	{
		std::cout << "Point count is not positive!\n";
		exit(-1);
	}
	m_pointCount = pointCount;

	if (pointDimension <= 0)
	{
		std::cout << "Point dimension is not positive!\n";
		exit(-1);
	}
	m_pointDimension = pointDimension;

	if (dampingFactor < 0 || dampingFactor > 1)
	{
		std::cout << "Damping factor outside the range!\n";
		exit(-1);	
	}
	m_dampingFactor = dampingFactor;

    // CPU allocations and copy operations
    m_points = new float[m_pointCount * m_pointDimension]();
    m_similarity = new float[m_pointCount * m_pointCount]();
    m_responsibility = new float[m_pointCount * m_pointCount]();
    m_availability = new float[m_pointCount * m_pointCount]();
	memcpy(m_points, points, 4 * m_pointCount * m_pointDimension);

    // GPU allocations andcopy operations
    cudaMalloc((void**)&m_devicePoints, 4 * m_pointCount * m_pointDimension);
    cudaMalloc((void**)&m_deviceSimilarity, 4 * m_pointCount * m_pointCount);
    cudaMalloc((void**)&m_deviceResponsibility, 4 * m_pointCount * m_pointCount);
    cudaMalloc((void**)&m_deviceAvailability, 4 * m_pointCount * m_pointCount);

    cudaMalloc((void**)&m_deviceExemplars, m_pointCount);
    cudaMalloc((void**)&m_devicePointLabel, 4 * m_pointCount);
    cudaMalloc((void**)&m_deviceMaxForResponsibility, 2 * m_pointCount);

    cudaMemset(m_deviceSimilarity, 0, 4 * m_pointCount * m_pointCount);
    cudaMemset(m_deviceResponsibility, 0, 4 * m_pointCount * m_pointCount);
    cudaMemset(m_deviceAvailability, 0, 4 * m_pointCount * m_pointCount);

    cudaMemcpy(m_devicePoints, m_points, 4 * m_pointCount * m_pointDimension, cudaMemcpyHostToDevice);
}

ApcGpuV2::~ApcGpuV2()
{
    delete m_points;
    delete m_similarity;
    delete m_responsibility;
    delete m_availability;
    cudaFree(m_devicePoints);
    cudaFree(m_deviceSimilarity);
    cudaFree(m_deviceResponsibility);
    cudaFree(m_deviceAvailability);
    cudaFree(m_deviceExemplars);
    cudaFree(m_devicePointLabel);
    cudaFree(m_deviceMaxForResponsibility);
}

void ApcGpuV2::cluster(int iterations)
{
    Timer timer("GpuV2");
	timer.start();
	updateSimilarity();
	for (int iter = 0; iter < iterations; iter++)
	{
		updateResponsibility();
		updateAvailability();
	}
    cudaDeviceSynchronize();
	timer.endAndPrint();
    labelPoints();
}

void ApcGpuV2::updateSimilarity()
{
    // Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
    // Call 2D similarity kernel
	launchKernel_updateSimilarity(blockCount, threadCount, m_devicePoints, m_deviceSimilarity, m_pointCount, m_pointDimension);
}

void ApcGpuV2::updateResponsibility()
{
    // Find max and runnerup values
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
    int blockCount1d = ((m_pointCount - 1)/ 1024) + 1;
    launchKernel_findMaxForResponsibility(blockCount1d, 1024, m_deviceSimilarity, m_deviceAvailability, m_deviceMaxForResponsibility, m_pointCount);
    launchKernel_updateResponsibilityWithMax(blockCount, 32, m_deviceSimilarity, m_deviceResponsibility, m_deviceAvailability, m_pointCount, m_dampingFactor, m_deviceMaxForResponsibility);
}

void ApcGpuV2::updateAvailability()
{
	// Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateAvailability(blockCount, threadCount, m_deviceSimilarity, m_deviceResponsibility, m_deviceAvailability, m_pointCount, m_dampingFactor);
}

void ApcGpuV2::labelPoints()
{
    int blockCount = ((m_pointCount - 1) / 1024) + 1;
    launchKernel_extractExemplars(blockCount, 1024, m_deviceResponsibility, m_deviceAvailability, m_pointCount, m_deviceExemplars);
    launchKernel_labelPoints(blockCount, 1024, m_deviceSimilarity, m_deviceExemplars, m_devicePointLabel, m_pointCount);
    
    int* labels = new int[m_pointCount];
    cudaMemcpy(labels, m_devicePointLabel, 4 * m_pointCount, cudaMemcpyDeviceToHost);
    std::ofstream clusterFile("GpuV2Clusters.txt");
    for (int i = 0; i < m_pointCount; i++)
    {
		for (int d = 0; d < m_pointDimension; d++)
            clusterFile << m_points[m_pointDimension * i + d] << " ";
        clusterFile << labels[i] + 1 << "\n";
    }
    clusterFile.close();
    std::cout << "Labels written to GpuV2Clusters.txt\n\n";
    delete labels;
}
