#pragma once

class ApcCpu
{
public:
	ApcCpu(float* points, int pointCount, int pointDimension);
	~ApcCpu();
	void cluster(int iterations = 100);

private:
	void updateSimilarity();
	void updateResponsibility();
	void updateAvailability();
	void labelPoints();

	int m_pointCount;
	int m_pointDimension;

	float* m_points;
	float* m_similarity;
	float* m_responsibility;
	float* m_availability;
};

