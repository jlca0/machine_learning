class Clustering:
    def __init__(
            self, 
            points: list, 
            wrap=True
        ) -> None:
        """Initializes a Clustering object. If wrap is True all the clusters are considered to be
        individual, it corresponds to an initializing state.
        
        Atributes:
            clustering: (list) A list for the clusters defining the clustering.
        """
        if wrap:
            self.clustering = [[point] for point in points]
        else:
            self.clustering = points

    def __len__(self) -> int:
        """Computes the number of clusters in the clustering."""
        return len(self.clustering)
    
    def __iter__(self):
        """Returns an iterator for the cluster list."""
        return iter(self.clustering)
    
    def __eq__(self, other: "Clustering") -> bool:
        """Checks if two clusterings are equal."""
        return self.clustering == other.clustering

    def merge_clusters(
            self, 
            index1: int, 
            index2: int
        ) -> "Clustering":
        """Given two clusters from the clustering it merges then into a new cluster.
        
        Parameters:
            index1: (int) Index of a cluster in the clustering.
            index2: (int) Index of another cluster.
        
        Returns:
            (Clustering) A new clustering with the selected clusters merged in the lowest index.
        """
        new_cluster = self.clustering[index1] + self.clustering[index2]
        new_clustering = [
            cluster for idx, cluster in enumerate(self.clustering)
            if idx != index1 and idx != index2
        ]
        new_clustering.insert(min(index1, index2), new_cluster)
        return Clustering(new_clustering, wrap=False)