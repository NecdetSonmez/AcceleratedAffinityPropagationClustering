#pragma once

#include <random>
#include <tuple>
#include <vector>

// Generates 2D points only
class Points
{
public:
    Points() {};
    ~Points();
    void generatePoints(std::vector<std::vector<float>> centers, int count, float standardDeviation, int dimension);
    float* getPoints();
    int getCount();
private:
    int m_count = 0;
    float* m_points = nullptr;
};