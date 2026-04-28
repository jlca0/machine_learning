# machine-learning
Classic statistical models such as Multiple Linear Regression or Factorial Analysis lay the foundation for most of the widely used machine learning algorithms. For this Python implementation the focus is on actually understanding from first principles the building blocks of this fundamental algortihms, including not only their interpretation but their actual numerical resolution.

To that end, we include some **auxiliary libraries** which implement data structures or well-known numerical linear algebra methods:

- LINALG: It includes the main equation solving algorithms of the project including **QR decomposition** through Housholder reflexions, **Jacobi iterative method** for eigen decomposition and svd decomposition or **Cholesky decomposition** (remains to be implemented). Each method is matched with its own statistical inference problem.
- GRAPH: It includes a basic graph library including directed and weighted graphs. **BFS, DFS and Dijkstra** are implemented through the queue Python library. It is meant to be applied for the isomap algortihm, which remains to be implemented.
- CLUSTER: It includes the tools to create clusterings and merge clusters. Purposely coded for the classification algorithms.
- TEST: It includes several $\chi^2$ tests, as well as Bartlett's test or the KMO index. It is meant to provide the tools needed to verify a range of model hypothesis such as normality or significant correlation. Remains to be implemented. 

As for the models themselves we will make two distinctions: descriptive models and classification models. In the **descriptive model** subcategory one can find:

-MLR: The **Multiple Linear Regression** model is implemented. Coefficients are fitted solving the normal equations through the Housholder algorithm. It includes inference tools for the validation of the model as well as model hypothesis validation.
-PCA: The **Principal Component Analysis** model is implemented. The svd decomposition of the covariance matrix is computed via Jacobi algorithm. It includes a biplot visualization of the PC1-PC2 space.
-FA: The **Factorial Analysis** model is implemented. The factorial charge matrix is computed with the iterative factor method using a Cholesky decomposition. Remains to be implemented.
-MDS: The classic euclidean **Multidimensional Scaling** model is implemented. It reuses the PCA implementation in order to project the double-centered distance matrix. Remains to be implemented.
-KPCA: The **Kernel Principal Component Analysis** model is implemented. Providing some of the most commonly used kernels it reuses the PCA implementation in the high-order featured space. Remains to be implemented.
-ISOMAP: The Isomap model is implemented. It reuses CA to define an obect from GRAPH which will determine, by a search algorithm, the distance matrix to be processed by MDS. Remains to be implemented. 


In the **classification model** subcategory one can find:

-CA: Cluster analysis algorithms are implemented including the hierarchical agglomerative algorithm (with different linkages) and the k nearest neighbours algorithm (remains to be implemented). It makes use of the CLUSTER library.
-DA: A discriminant analysis algorithm remains to be implemented.

These algorithms, together with their iterated application, serve well the pedagogical purpose of introducing the basics of modern techniques in machine learning without losing the technical aspects of their implementation.
