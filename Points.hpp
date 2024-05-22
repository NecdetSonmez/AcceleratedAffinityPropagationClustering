#pragma once

#include <random>
#include <tuple>
#include <vector>

class Points
{
public:
    Points() {};
    ~Points();
    void generatePoints(std::vector<std::vector<float>> centers, int count, int dimension);
    void loadFromFile(std::string path);
    float* getPoints();
    int getCount();
private:
    int m_count = 0;
    float* m_points = nullptr;
};