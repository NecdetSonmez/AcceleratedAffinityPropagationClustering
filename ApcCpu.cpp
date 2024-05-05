#include "ApcCpu.hpp"
#include "Timer.hpp"
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <vector>
#include <fstream>

ApcCpu::ApcCpu(float* points, int pointCount, int pointDimension, float dampingFactor)
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

	m_points = new float[m_pointCount * m_pointDimension]();
	m_similarity = new float[m_pointCount * m_pointCount]();
	m_responsibility = new float[m_pointCount * m_pointCount]();
	m_availability = new float[m_pointCount * m_pointCount]();

	memcpy(m_points, points, 4 * m_pointCount * m_pointDimension);
}

ApcCpu::~ApcCpu()
{
	delete m_points;
	delete m_similarity;
	delete m_responsibility;
	delete m_availability;
}

void ApcCpu::cluster(int iterations)
{
	Timer timer("Cpu");
	timer.start();
	updateSimilarity();
	for (int iter = 0; iter < iterations; iter++)
	{
		updateResponsibility();
		updateAvailability();
	}
	timer.endAndPrint();
	labelPoints();
}

void ApcCpu::updateSimilarity()
{
	for (int i = 0; i < m_pointCount; i++)
	{
		for (int j = 0; j < m_pointCount; j++)
		{
			if (i == j)
			{
				// TODO: Change diagonal values to a parameter
				m_similarity[m_pointCount * i + j] = -1.0f;
				continue;
			}

			// Sweep through each similarity case by dimension
			float sim = 0;
			for (int d = 0; d < m_pointDimension; d++)
			{
				float difference = (m_points[m_pointDimension * i + d] - m_points[m_pointDimension * j + d]);
				sim += difference * difference;
			}
			m_similarity[m_pointCount * i + j] = -sim;
		}
	}
}

void ApcCpu::updateResponsibility()
{
	for (int i = 0; i < m_pointCount; i++)
	{
		for (int j = 0; j < m_pointCount; j++)
		{
			// Find max of A(i,k) + S(i,k), sweep k.
			float max = -std::numeric_limits<float>::max();
			for (int k = 0; k < m_pointCount; k++)
			{
				if (k == j)
					continue;

				float temp = m_availability[m_pointCount * i + k] + m_similarity[m_pointCount * i + k];
				if (temp > max)
					max = temp;
			}

			if (max == -std::numeric_limits<float>::max())
				std::cout << "updateResponsibilityCpu: No max found for " << i << "," << j << "\n";
				
			// Max found, calculate responsibility.
			float newResponsibility = m_similarity[m_pointCount * i + j] - max;
			m_responsibility[m_pointCount * i + j] = m_dampingFactor * m_responsibility[m_pointCount * i + j] + (1 - m_dampingFactor) * newResponsibility;
		}
	}
}

void ApcCpu::updateAvailability()
{
	for (int i = 0; i < m_pointCount; i++)
	{
		for (int j = 0; j < m_pointCount; j++)
		{
			if (i == j)
			{
				float newAvailability = 0;
				// Sum positive R(k,j) values, sweep k.
				for (int k = 0; k < m_pointCount; k++)
				{
					if (k == i)
						continue;
					newAvailability += (m_responsibility[m_pointCount * k + j] > 0 ? m_responsibility[m_pointCount * k + j] : 0);
				}
				m_availability[m_pointCount * i + j] = m_dampingFactor * m_availability[m_pointCount * i + j] + (1 - m_dampingFactor) * newAvailability;
			}
			else
			{
				float newAvailability = 0;
				// Sum positive R(k,j) values, sweep k.
				for (int k = 0; k < m_pointCount; k++)
				{
					if (k == i || k==j)
						continue;
					newAvailability += (m_responsibility[m_pointCount * k + j] > 0 ? m_responsibility[m_pointCount * k + j] : 0);
				}
				newAvailability += m_responsibility[m_pointCount * j + j];
				
				// min(0, newAvailability)
				if (newAvailability > 0)
					newAvailability = 0;
				m_availability[m_pointCount * i + j] = m_dampingFactor * m_availability[m_pointCount * i + j] + (1 - m_dampingFactor) * newAvailability;
			}
		}
	}
}

void ApcCpu::labelPoints()
{
	// Find all exemplar points by checking the criteria
	std::ofstream clusterFile("CpuClusters.txt");
	std::vector<int> exemplars;
	for (int i = 0; i < m_pointCount; i++)
	{
		float criteria = m_availability[m_pointCount * i + i] + m_responsibility[m_pointCount * i + i];
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
	std::cout << "Labels written to CpuClusters.txt\n\n";
}