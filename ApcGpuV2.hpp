#pragma once

class ApcGpuV2
{
public:
	ApcGpuV2(float* points, int pointCount, int pointDimension, float dampingFactor);
	~ApcGpuV2();
	void cluster(int iterations = 100);

private:
	void updateSimilarity();
	void updateResponsibility();
	void updateAvailability();
	void labelPoints();

	float m_dampingFactor;
	int m_pointCount;
	int m_pointDimension;

	float* m_points;
	float* m_similarity;
	float* m_responsibility;
	float* m_availability;
    
    float* m_devicePoints;
	float* m_deviceSimilarity;
	float* m_deviceResponsibility;
	float* m_deviceAvailability;

    char* m_deviceExemplars; // 1 or 0 indicating if the point is an exemplar 
    int* m_devicePointLabel; // Integer for point cluster exemplar index
    float* m_deviceSumsOfResponsibility;
};


