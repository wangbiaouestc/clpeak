/*
 * Clustering.hpp
 *
 *  Created on: Dec 1, 2013
 *      Author: michael
 */
#ifndef CLUSTERING_H_
#define CLUSTERING_H_

#include <vector>
#include <map>

namespace util {

	class Clustering {
		public:
			typedef std::vector<float> DataVector;
			
			class Cluster {
			public:
				Cluster() : position(0.0), score(0.0) { }
				Cluster(float initialPosition) : position(initialPosition), score(0.0) { }
				
			public:
				void updatePosition();
				void updateScore();
			
			public:
				float position;
				float score;
				DataVector data;
			};
			
			typedef std::vector<Cluster> ClusterVector;
			
			class ClusterResult {
			public:
				ClusterResult() : aggregateScore(0.0) { }
				
			public:
				void add(const Cluster& cluster) {
					clusters.push_back(cluster);
					aggregateScore += cluster.score;
				}
				
				const ClusterVector& getClusters() const {
					return clusters;
				}
				
				float getScore() const {
					return aggregateScore;// / clusters.size();
				}
				
				ClusterVector::const_iterator begin() const {
					return clusters.begin();
				} 
				
				ClusterVector::const_iterator end() const {
					return clusters.end();
				}
			
			private:
				ClusterVector clusters;
				float aggregateScore;
			};
			
			typedef std::map< unsigned, ClusterResult > ClusterMap;
			typedef std::pair< unsigned, ClusterResult > CountClusterPair;
			
			struct ScoreComparator {			
				bool operator()(const CountClusterPair& lhs, const CountClusterPair& rhs) {
					return lhs.second.getScore() < rhs.second.getScore();
				}
			};
		public:
			Clustering(const DataVector& _data, unsigned _max);
			~Clustering();
		
		public:
			const ClusterResult& getResult() const { return result; }
		private:
			ClusterResult _kmeans(unsigned k);
		
		private:
			DataVector m_data;
			unsigned m_maxClusters;
			
			ClusterResult result;
		};
		
}

#endif /* CLUSTERING_H_ */
