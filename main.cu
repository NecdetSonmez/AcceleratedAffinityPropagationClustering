
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "AffinityPropagationClustering.hpp"

int main()
{
    // Test points for the algorithm
    float points[] = {1.1, 2.0, 1.0, 2.1, 0.9, 2.0, 1.2, 2.1, 3.0, 4.5, 3.1, 4.3, 3.2, 4.3, 3.4, 4.4};
    AffinityPropagationClustering a(points, 8, 2, 0.5);
    a.clusterCpu();
    return 0;
}