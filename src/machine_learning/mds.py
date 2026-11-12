import numpy as np
import matplotlib.pyplot as plt

from .pca import PCA

class MDS:
    def __init__(self, distance: np.ndarray):
        self.distance = distance
        self.embedding = None

    def fit(self, n_components: int = 2):
        n_samples = self.distance.shape[0]
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        Q = -0.5 * H @ (self.distance ** 2) @ H
        pca = PCA(Q)
        self.embedding = pca.fit_transform(n_components)

distance_europe = np.array([
    [   0, 1055, 1870, 1365, 1480, 1315, 1810, 1770,  500, 1265],
    [1055,    0,  880, 1105,  430,  265, 1035,  885, 1450,  340],
    [1870,  880,    0, 1180,  575,  650,  525,  280, 2310,  930],
    [1365, 1105, 1180,    0, 1300, 1175,  765,  920, 1860, 1435],
    [1480,  430,  575, 1300,    0,  175,  940,  710, 1860,  360],
    [1315,  265,  650, 1175,  175,    0,  915,  715, 1710,  320],
    [1810, 1035,  525,  765,  940,  915,    0,  250, 2300, 1235],
    [1770,  885,  280,  920,  710,  715,  250,    0, 2220, 1030],
    [ 500, 1450, 2310, 1860, 1860, 1710, 2300, 2220,    0, 1585],
    [1265,  340,  930, 1435,  360,  320, 1235, 1030, 1585,    0]
])

if __name__ == "__main__":
    mds = MDS(distance_europe)
    mds.fit(n_components=2)
    cities = ["Madrid", "Paris", "Berlin", "Rome", "Amsterdam", "Brussels", "Wien", "Prague", "Lisbon", "London"]

    plt.scatter(mds.embedding[:, 0], mds.embedding[:, 1])

    for i, city in enumerate(cities):
        plt.annotate(city, (mds.embedding[i, 0], mds.embedding[i, 1]))

    plt.gca().invert_yaxis()
    plt.show()