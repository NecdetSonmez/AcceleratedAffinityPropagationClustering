#pragma once

class AffinityPropagationClustering
{
public:
	AffinityPropagationClustering(float* points, int pointCount, int pointDimension, float dampingFactor);
	~AffinityPropagationClustering();
	void clusterCpu();

private:
	void updateSimilarityCpu();
	void updateResponsibilityCpu();
	void updateAvailabilityCpu();
	void labelPoints();

	float m_dampingFactor;
	int m_pointCount;
	int m_pointDimension;

	float* m_points;
	float* m_similarity;
	float* m_responsibility;
	float* m_availability;
};

