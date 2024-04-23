#include "ApcGpu.hpp"
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>

#include "Kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ApcGpu::ApcGpu(float* points, int pointCount, int pointDimension, float dampingFactor)
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

	cudaMallocManaged((void**)&m_points, 4 * m_pointCount * m_pointDimension);
    cudaMallocManaged((void**)&m_similarity, 4 * m_pointCount * m_pointCount);
    cudaMallocManaged((void**)&m_responsibility, 4 * m_pointCount * m_pointCount);
    cudaMallocManaged((void**)&m_availability, 4 * m_pointCount * m_pointCount);

	memcpy(m_points, points, 4 * m_pointCount * m_pointDimension);
}

ApcGpu::~ApcGpu()
{
    cudaFree(m_points);
    cudaFree(m_similarity);
    cudaFree(m_responsibility);
    cudaFree(m_availability);
}

void ApcGpu::cluster(int iterations)
{
	updateSimilarity();
	for (int iter = 0; iter < iterations; iter++)
	{
		updateResponsibility();
		updateAvailability();
	}
	cudaDeviceSynchronize();
	labelPoints();
}

void ApcGpu::updateSimilarity()
{
    // Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;

    // Call 2D similarity kernel
	launchKernel_updateSimilarity(blockCount, threadCount, m_points, m_similarity, m_pointCount, m_pointDimension);
}

void ApcGpu::updateResponsibility()
{
	// Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateResponsibility(blockCount, threadCount, m_similarity, m_responsibility, m_availability, m_pointCount, m_dampingFactor);
}

void ApcGpu::updateAvailability()
{
	// Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateAvailability(blockCount, threadCount, m_similarity, m_responsibility, m_availability, m_pointCount, m_dampingFactor);
}

void ApcGpu::labelPoints()
{
	// TODO: PRIORITY Can switch this to GPU as well. Currently in CPU.

	// Find all exemplar points by checking the criteria
	std::vector<int> exemplars;
	for (int i = 0; i < m_pointCount; i++)
	{
		float criteria = m_availability[m_pointCount * i + i] + m_responsibility[m_pointCount * i + i];
		std::cout << "A + R for " << i << ": " << criteria << "\n";
		if (criteria > 0)
			exemplars.push_back(i);
	}

	// Label all points and print
	for (int i = 0; i < m_pointCount; i++)
	{
		// Find max similarity to an exemplar per point
		float max = -std::numeric_limits<float>::max();
		int selectedExemplar = -1;
		for (int e = 0; e < exemplars.size(); e++)
		{
			if (m_similarity[m_pointCount * i + exemplars[e]] > max)
			{
				max = m_similarity[m_pointCount * i + exemplars[e]];
				selectedExemplar = e;
			}
		}

		if (selectedExemplar == -1)
			std::cout << "No exemplar selected for" << i << "!";
		else
			std::cout << "Point " << i << ": Cluster " << selectedExemplar << " around point " << exemplars[selectedExemplar] <<"\n";
	}
}
