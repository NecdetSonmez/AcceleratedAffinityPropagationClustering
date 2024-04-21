#pragma once

class ApcGpu
{
public:
	ApcGpu(float* points, int pointCount, int pointDimension, float dampingFactor);
	~ApcGpu();
	void cluster();

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
};
