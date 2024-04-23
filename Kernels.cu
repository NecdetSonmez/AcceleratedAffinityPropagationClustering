#include "Kernels.cuh"
#include <cfloat>

void launchKernel_updateSimilarity(int blockCount, int threadCount, float* points, float* similarity, int pointCount, int pointDimension)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateSimilarity<<<blockCount2d, threadCount2d>>>(points, similarity, pointCount, pointDimension);
    cudaDeviceSynchronize();
}

__global__ void Kernel_updateSimilarity(float* points, float* similarity, int pointCount, int pointDimension)
{
    // Open n^2 threads for this kernel. Each thread processes one (i,j) pair. Does not sweep.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pointCount || j >= pointCount)
        return;
    if (i == j)
    {
        similarity[pointCount * i + j] = -1;
    }
    else
    {
        float sim = 0;
        for (int d = 0; d < pointDimension; d++)
		{
			float difference = (points[pointDimension * i + d] - points[pointDimension * j + d]);
			sim += difference * difference;
		}
		similarity[pointCount * i + j] = -sim;
    }
}

void launchKernel_updateResponsibility(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateResponsibility<<<blockCount2d, threadCount2d>>>(similarity, responsibility, availability, pointCount, dampingFactor);
}

__global__ void Kernel_updateResponsibility(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor)
{
    // Open n^2 threads for this kernel. Each thread finds max of A(i,k) + S(i,k). For each point (i,j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pointCount || j >= pointCount)
        return;

    float max = -FLT_MAX;
    for (int k = 0; k < pointCount; k++)
	{
		if (k == j)
			continue;
		float temp = availability[pointCount * i + k] + similarity[pointCount * i + k];
		if (temp > max)
			max = temp;
	}
		
	// Max found, calculate responsibility.
	float newResponsibility = similarity[pointCount * i + j] - max;
	responsibility[pointCount * i + j] = dampingFactor * responsibility[pointCount * i + j] + (1 - dampingFactor) * newResponsibility;
}

void launchKernel_updateAvailability(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateAvailability<<<blockCount2d, threadCount2d>>>(similarity, responsibility, availability, pointCount, dampingFactor);
}

__global__ void Kernel_updateAvailability(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor)
{
    // Open n^2 threads for this kernel. For each point (i,j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pointCount || j >= pointCount)
        return;

	if (i == j)
	{
		float newAvailability = 0;
		// Sum positive R(k,j) values, sweep k.
		for (int k = 0; k < pointCount; k++)
		{
			if (k == i)
				continue;
			newAvailability += (responsibility[pointCount * k + j] > 0 ? responsibility[pointCount * k + j] : 0);
		}
		availability[pointCount * i + j] = dampingFactor * availability[pointCount * i + j] + (1 - dampingFactor) * newAvailability;
	}
	else
	{
		float newAvailability = 0;
		// Sum positive R(k,j) values, sweep k.
		for (int k = 0; k < pointCount; k++)
		{
			if (k == i || k==j)
				continue;
			newAvailability += (responsibility[pointCount * k + j] > 0 ? responsibility[pointCount * k + j] : 0);
		}
		newAvailability += responsibility[pointCount * j + j];
		
		// min(0, newAvailability)
		if (newAvailability > 0)
			newAvailability = 0;
		availability[pointCount * i + j] = dampingFactor * availability[pointCount * i + j] + (1 - dampingFactor) * newAvailability;
	}
}