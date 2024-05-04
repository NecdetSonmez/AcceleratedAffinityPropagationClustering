#include "Kernels.cuh"
#include <cfloat>

void launchKernel_updateSimilarity(int blockCount, int threadCount, float* points, float* similarity, int pointCount, int pointDimension)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateSimilarity<<<blockCount2d, threadCount2d>>>(points, similarity, pointCount, pointDimension);
    //cudaDeviceSynchronize();
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
        similarity[pointCount * i + j] = -1.0f;
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

void launchKernel_extractExemplars(int blockCount, int threadCount, float* responsibility, float* availability, int pointCount, char* exemplars)
{
    Kernel_extractExemplars<<<blockCount, threadCount>>>(responsibility, availability, pointCount, exemplars);
}

__global__ void Kernel_extractExemplars(float* responsibility, float* availability, int pointCount, char* exemplars)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pointCount)
        return;

	float criteria = availability[pointCount * i + i] + responsibility[pointCount * i + i];
	if (criteria > 0)
		exemplars[i] = 1;
	else
		exemplars[i] = 0;
}

void launchKernel_labelPoints(int blockCount, int threadCount, float* similarity, char* exemplars, int* pointLabels, int pointCount)
{
	Kernel_labelPoints<<<blockCount, threadCount>>>(similarity, exemplars, pointLabels, pointCount);
}

__global__ void Kernel_labelPoints(float* similarity, char* exemplars, int* pointLabels, int pointCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pointCount)
        return;

	float max = -FLT_MAX;
	int selectedExemplar = -1;
	for (int e = 0; e < pointCount; e++)
	{
		if (exemplars[e] == 0)
			continue;

		// If current point is an exemplar, label it as itself.
		if (e == i)
		{
			selectedExemplar = e;
			break;
		}

		float temp = similarity[pointCount * i + e];
		if (temp > max)
		{
			max = temp;
			selectedExemplar = e;
		}
	}
	pointLabels[i] = selectedExemplar;
}

void launchKernel_sumOfResponsibility(int blockCount, int threadCount, float* responsibility, int pointCount, float* sumsOfResponsibility)
{
	//cudaMemset(sumsOfResponsibility, 0, 4 * pointCount);
	Kernel_sumOfResponsibility<<<blockCount, threadCount>>>(responsibility, pointCount, sumsOfResponsibility);
}

__global__ void Kernel_sumOfResponsibility(float* responsibility, int pointCount, float* sumsOfResponsibility)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= pointCount)
        return;

	float sum = 0;
	for (int k = 0; k < pointCount; k++)
    	sum += (responsibility[pointCount * k + j] > 0 ? responsibility[pointCount * k + j] : 0);
	sumsOfResponsibility[j] = sum;
}

void launchKernel_updateAvailabilityWithSum(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* sumsOfResponsibility)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateAvailabilityWithSum<<<blockCount2d, threadCount2d>>>(similarity, responsibility, availability, pointCount, dampingFactor, sumsOfResponsibility);
}

__global__ void Kernel_updateAvailabilityWithSum(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* sumsOfResponsibility)
{
    // Open n^2 threads for this kernel. For each point (i,j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pointCount || j >= pointCount)
        return;

	if (i == j)
	{
		float newAvailability = sumsOfResponsibility[j];

		newAvailability -= (responsibility[pointCount * i + j] > 0 ? responsibility[pointCount * i + j] : 0);

		availability[pointCount * i + j] = dampingFactor * availability[pointCount * i + j] + (1 - dampingFactor) * newAvailability;
	}
	else
	{
		float newAvailability = sumsOfResponsibility[j];

		newAvailability -= (responsibility[pointCount * i + j] > 0 ? responsibility[pointCount * i + j] : 0);
		newAvailability -= (responsibility[pointCount * j + j] > 0 ? responsibility[pointCount * j + j] : 0);

		newAvailability += responsibility[pointCount * j + j];
		
		// min(0, newAvailability)
		if (newAvailability > 0)
			newAvailability = 0;
		availability[pointCount * i + j] = dampingFactor * availability[pointCount * i + j] + (1 - dampingFactor) * newAvailability;
	}
}

void launchKernel_findMaxForResponsibility(int blockCount, int threadCount, float* similarity, float* availability, float* maxValues, int pointCount)
{
	Kernel_findMaxForResponsibility<<<blockCount, threadCount>>>(similarity, availability, maxValues, pointCount);
}

__global__ void Kernel_findMaxForResponsibility(float* similarity, float* availability, float* maxValues, int pointCount)
{
    // Open n threads for this kernel. Each thread finds max 2 values of A(i,k) + S(i,k).
	// This will then be used for updating the responsibility.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pointCount)
        return;

    float max = -FLT_MAX;
	float runnerup = -FLT_MAX;

    for (int k = 0; k < pointCount; k++)
	{
		float temp = availability[pointCount * i + k] + similarity[pointCount * i + k];
		if (temp > max)
		{
			runnerup = max;
			max = temp;
		}
		else if (temp > runnerup)
			runnerup = temp;
	}
		
	// Max values found, save to memory.
	maxValues[2 * i] = max;
	maxValues[2 * i + 1] = runnerup;
}

void launchKernel_updateResponsibilityWithMax(int blockCount, int threadCount, float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* maxValues)
{
    dim3 blockCount2d(blockCount, blockCount);
    dim3 threadCount2d(threadCount, threadCount);
    Kernel_updateResponsibilityWithMax<<<blockCount2d, threadCount2d>>>(similarity, responsibility, availability, pointCount, dampingFactor, maxValues);
}

__global__ void Kernel_updateResponsibilityWithMax(float* similarity, float* responsibility, float* availability, int pointCount, float dampingFactor, float* maxValues)
{
    // Open n^2 threads for this kernel. Each thread reads max of A(i,k) + S(i,k). For each point (i,j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pointCount || j >= pointCount)
        return;

	// Check which to use from max and runnerup
	float temp = availability[pointCount * i + j] + similarity[pointCount * i + j];
	float max = maxValues[2 * i];
	float runnerup = maxValues[2 * i + 1];
	if (temp == max)
		max = runnerup;
		
	// Max found, calculate responsibility.
	float newResponsibility = similarity[pointCount * i + j] - max;
	responsibility[pointCount * i + j] = dampingFactor * responsibility[pointCount * i + j] + (1 - dampingFactor) * newResponsibility;
}
