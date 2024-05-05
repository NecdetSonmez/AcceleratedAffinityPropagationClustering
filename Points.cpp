#include "Points.hpp"

#include <iostream>

void Points::generatePoints(std::vector<std::vector<float>> centers, int count, float standardDeviation, int dimension)
{
    // Allocate for m_points based on the provided dimension
    m_count = count;
    m_points = new float[dimension * m_count];

    // Generator for points
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());

    // Generator for selecting a center point at random
    std::random_device randomDeviceCenter;
    std::mt19937 generatorCenter(randomDeviceCenter());
    std::uniform_int_distribution<> distributionCenter(0, centers.size() - 1);

    int pointsIndex = 0;
    for (int i = 0; i < m_count; i++)
    {
        // Select a random center point
        int randomCenterIndex = distributionCenter(generatorCenter);
        
        for (int dim = 0; dim < dimension; ++dim) {
            std::normal_distribution<float> distribution(centers[randomCenterIndex][dim], standardDeviation);
            m_points[pointsIndex + dim] = distribution(generator);
        }
        
        pointsIndex += dimension;
    }
}

Points::~Points()
{
    delete m_points;
}

float* Points::getPoints()
{
    return m_points;
}

int Points::getCount()
{
    return m_count;
}
