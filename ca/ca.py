from cluster import Clustering
from pca import PCA

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class HierarchicalCluster:

    def __init__(
            self,
            points: list, 
            choice='euclidean', 
            linkage='single'
        ) -> None:
        """Initializes a HierarchicalCluster object from a provided set of points. It requires
        a linkage method and a distance in order to perform the clustering.
        
        Atributes:
            history: (list) The clustering history.
            index: (list) The merging index history.
            choice: (str) A distance.
            linkage: (str) A linkage method.
        """
        self.history = [Clustering(points)]
        self.index = np.zeros(len(points)-1)
        self.choice = choice
        self.linkage = linkage
        
    def distance(
            self, 
            x: np.ndarray, 
            y: np.ndarray, 
            choice='euclidean'
        ) -> float:
        """Given to points it computes their distance in euclidean or Manhattan norm.

        Parameters:
            x: (np.ndarray) A point.
            y: (np.ndarray) Another point.
            choice: (str) A chosen norm.

        Returns:
            (float) Distance between x and y.
        """
        if choice == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif choice == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            raise ValueError("choice must be 'euclidean' or 'manhattan'")
        
    def fit(self):
        """Performs a Hierarchical Clustering to the given points using a linkage method and distance. 
        It saves the clustering history as well as the index history."""

        for k in range(len(self.history[0]) - 1):
            (best_i, cluster1), (best_j, cluster2) = min(
                combinations(enumerate(self.history[k]), 2),
                key=lambda pair: self._cluster_distance(pair[0][1], pair[1][1], self.linkage)
            )
            self.index[k+1] = self._cluster_distance(cluster1, cluster2, self.linkage)
            self._merge_clusters(k, best_i, best_j)

    def _cluster_distance(
            self, 
            cluster1: list, 
            cluster2: list, 
            linkage='single'
        ) -> float:
        """Given to clusters it computes their distance making use
        of a choosen linkage method including: single, complete, mean and Ward.
        
        Parameters:
            cluster1: (list) A cluster.
            cluster2: (list) Another cluster.
            linkage: (str) A linkage method.

        Returns:
            (float) Distance between cluster1 and cluster2. 
        """
        if linkage == 'single':
            return min(
            self.distance(p1, p2, self.choice)
            for p1 in cluster1
            for p2 in cluster2
        )
        elif linkage == 'complete':
            return max(
            self.distance(p1, p2, self.choice)
            for p1 in cluster1
            for p2 in cluster2
        )
        elif linkage == 'mean':
            centroid1 = np.mean(cluster1, axis=0)
            centroid2 = np.mean(cluster2, axis=0)
            return self.distance(centroid1, centroid2, self.choice)
        elif linkage == 'ward':
            centroid1 = np.mean(cluster1, axis=0)
            centroid2 = np.mean(cluster2, axis=0)
            centroid_merged = np.mean(cluster1 + cluster2, axis=0)
            variance1 = sum(np.sum((p - centroid1) ** 2) for p in cluster1)
            variance2 = sum(np.sum((p - centroid2) ** 2) for p in cluster2)
            variance_merged = sum(np.sum((p - centroid_merged) ** 2) for p in cluster1 + cluster2)
            return variance_merged - variance1 - variance2

    def _merge_clusters(
            self, 
            k: int, 
            i: int, 
            j: int
        ) -> None:
        """It adds a new clustering to the history created from the last clustering 
        but merging clusters C_i and C_j.
        
        Parameters:
            k: (int) Current position of the last clustering in self.history.
            i: (int) Index of one cluster in the last clustering.
            j: (int) Index of another cluster in the last clustering.
        """
        self.history.append(self.history[k].merge_clusters(i, j))

    def plot(
            self, 
            k: int
        ) -> None:
        pca = PCA(np.array([point for cluster in self.history[0] for point in cluster]))
        projected = pca.fit_transform(n_components=2)
        
        plt.figure(f'Clustering step {k}')
        for cluster_idx, cluster in enumerate(self.history[k]):
            indices = [i for i, point in enumerate(self.history[0].clustering) 
                    for p in cluster if np.array_equal(point[0], p)]
            plt.scatter(projected[indices, 0], projected[indices, 1], label=f'C{cluster_idx}')
        
        plt.title(f'Step {k}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()
