/*
 * Clustering.cpp
 *
 *  Created on: Dec 1, 2013
 *      Author: michael
 */

#include <util/interface/Clustering.hpp>
#include <util/interface/Util.hpp>

#include <algorithm>
#include <cstdlib>

#ifdef REPORT_LEVEL
#undef REPORT_LEVEL
#endif

#define REPORT_LEVEL 1
#include <util/interface/Debug.hpp>

static float _distance(float val, float cluster) {
	return (cluster - val) * (cluster - val);
}

void util::Clustering::Cluster::updatePosition() {
	float sum = 0.0;
	for (auto it = data.begin(); it != data.end(); ++it) {
		sum += *it;
	}
	if (data.size()) {
		position = sum / data.size();
	} else {
		position = 0.0;
	}
}

void util::Clustering::Cluster::updateScore() {
	score = 0.0;
	for (auto it = data.begin(); it != data.end(); ++it) {
		score += _distance(*it, position);
	}
// 	if (data.size()) {
// 		score /= data.size();
// 	} else {
// 		score = 1e20;
// 	}
}

util::Clustering::Clustering(const DataVector& _data, unsigned _max) : m_maxClusters(_max) {
	m_data.insert(m_data.begin(), _data.begin(), _data.end());
	
	ClusterMap clusterQualityScores;
	
	for (unsigned clusterCount = 1; clusterCount <= m_maxClusters; clusterCount++) {
		clusterQualityScores[clusterCount] = _kmeans(clusterCount);
	}
		
	std::vector< CountClusterPair > sortedScores;
	std::copy(clusterQualityScores.begin(), clusterQualityScores.end(), std::back_inserter(sortedScores));
	ScoreComparator comparator;
	std::sort(sortedScores.begin(), sortedScores.end(), comparator);

	report("Scores:");
	for (int i = 0; i < sortedScores.size(); i++) {
		report(sortedScores[i].first << " clusters: " << sortedScores[i].second.getScore());
	}
	
	report("Clustering with highest score: ");
	report(util::Indents(2) << "cluster count: " << sortedScores[0].first);
	report(util::Indents(2) << "aggregate score: " << sortedScores[0].second.getScore());
	report(util::Indents(2) << "detected clusters:");
	for (auto it = sortedScores[0].second.getClusters().begin(); it != sortedScores[0].second.getClusters().end(); ++it) {
		const Cluster& cluster = *it;
		report(util::Indents(4) << "position = " << cluster.position << ", elements = " << cluster.data.size());
	}
	
	result = sortedScores[0].second;
}

util::Clustering::~Clustering() {

}

util::Clustering::ClusterResult util::Clustering::_kmeans(unsigned k) {
	if (k == 0) {
		return ClusterResult();
	}
	
	const double threshold = 0.001;
	const unsigned maxIterations = 5000;
	srand((int)(*m_data.begin()));
	
	double delta = 0.0;
	unsigned iteration = 0;
	
	typedef std::vector< short > Membership;
	Membership membership(m_data.size(), -1);
	
	typedef std::vector< float > Distances;
	Distances distances(k, 0.0);
	
	ClusterVector clusters;
	clusters.resize(k);
	for (unsigned i = 0; i < k; i++) {
		clusters[i].position = m_data[rand() % m_data.size()];
	}
	
	// clustering main loop
	do {
		delta = 0.0;
		for (unsigned i = 0; i < k; i++) {
			clusters[i].data.clear();
		}
		
		for (unsigned i = 0; i < m_data.size(); i++) {
			float value = m_data[i];
			
			// compute which cluster lies at minimum distance to data point
			float dist = 0.0, minDist = 1e20;
			unsigned minIdx = 0;
			for (unsigned j = 0; j < k; j++) {
				if ((dist = _distance(value, clusters[j].position)) < minDist) {
					minDist = dist;
					minIdx = j;
				};
			}
			clusters[minIdx].data.push_back(value);
			
			// check if membership changed
			if (membership[i] != minIdx) {
				delta += 1.0;
				membership[i] = minIdx;
			}
		}
		
		for (unsigned i = 0; i < k; i++) {
			clusters[i].updatePosition();
		}
		
		delta /= m_data.size();
		iteration++;
	} while (delta > threshold && iteration < maxIterations);
	
	// compute score for the clustering
	for (unsigned i = 0; i < k; i++) {
		clusters[i].updateScore();
	}
	
	ClusterResult result;
	for (auto it = clusters.begin(); it != clusters.end(); ++it) {
		result.add(*it);
	}
	return result;
}