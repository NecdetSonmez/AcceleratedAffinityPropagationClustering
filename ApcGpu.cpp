#include "ApcGpu.hpp"
#include "Timer.hpp"
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>
#include <fstream>

#include "Kernels.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

ApcGpu::ApcGpu(float* points, int pointCount, int pointDimension)
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
	Timer timer("GpuV1");
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

void ApcGpu::updateSimilarity()
{
    // Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateSimilarity(blockCount, threadCount, m_points, m_similarity, m_pointCount, m_pointDimension);
}

void ApcGpu::updateResponsibility()
{
	// Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateResponsibility(blockCount, threadCount, m_similarity, m_responsibility, m_availability, m_pointCount);
}

void ApcGpu::updateAvailability()
{
	// Run 1024 = 32*32 threads per block
    int threadCount = 32;
    // Calculate block count
    int blockCount = ((m_pointCount - 1)/ 32) + 1;
	launchKernel_updateAvailability(blockCount, threadCount, m_similarity, m_responsibility, m_availability, m_pointCount);
}

void ApcGpu::labelPoints()
{
	// Find all exemplar points by checking the criteria
	std::ofstream clusterFile("GpuV1Clusters.txt");
	std::vector<int> exemplars;
	for (int i = 0; i < m_pointCount; i++)
	{
		float criteria = m_availability[m_pointCount * i + i] + m_responsibility[m_pointCount * i + i];
		if (criteria > 0)
		{
			exemplars.push_back(i);
		}
	}

	// Label all points and print
	for (int i = 0; i < m_pointCount; i++)
	{
		// Find max similarity to an exemplar per point
		float max = -std::numeric_limits<float>::max();
		int selectedExemplar = -1;
		for (int e = 0; e < exemplars.size(); e++)
		{
			if (exemplars[e] == i)
			{
				selectedExemplar = e;
				break;
			}

			if (m_similarity[m_pointCount * i + exemplars[e]] > max)
			{
				max = m_similarity[m_pointCount * i + exemplars[e]];
				selectedExemplar = e;
			}
		}
		for (int d = 0; d < m_pointDimension; d++)
			clusterFile << m_points[m_pointDimension * i + d] << " ";
		clusterFile << exemplars[selectedExemplar] + 1 << "\n";
	}
	clusterFile.close();
	std::cout << "Labels written to GpuV1Clusters.txt\n\n";
}
