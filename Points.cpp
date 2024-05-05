#include "Points.hpp"

#include <iostream>

void Points::generatePoints(std::vector<std::tuple<float, float>> centers, int count, float standardDeviation)
{
    // Allocate for m_points (point dimension assumed 2D)
    m_count = count;
    m_points = new float[2 * m_count];

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
        
        std::normal_distribution<float> distributionX(std::get<0>(centers[randomCenterIndex]), standardDeviation);
        std::normal_distribution<float> distributionY(std::get<1>(centers[randomCenterIndex]), standardDeviation);

        m_points[pointsIndex] = distributionX(generator);
        m_points[pointsIndex + 1] = distributionX(generator);
        pointsIndex += 2;
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
