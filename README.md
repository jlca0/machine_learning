# machine-learning

> *[English below](#english)*

---

## Español

Colección de modelos estadísticos multivariantes para descripción, reducción de dimensionalidad y exploración de datos. Pretende ser una construcción pedagógica y desde cero de los algoritmos básicos en ciencias de datos, con una atención especial a la implementación numérica de los mismos.

- **Módulo de álgebra lineal**: En `LINALG.py` se puede encontrar una implementación de la **descomposición QR** de una matriz usando **reflexiones de Householder** (regresión lineal) así como una implementación del **método de Jacobi** (análisis de componentes principales).

- **Módulo de regresión**: En `MLR.py` se implementa un modelo de Regresión Lineal Múltiple usando la descomposición QR en la resolución de las ecuaciones normales de la regresión. Incluye diagnósticos de las hipótesis del modelo como estimadores del condicionamiento de la matriz de diseño (colinealidad) y test de hipótesis sobre los estimadores como intervalos de confianza y distancias de Cook (observaciones atípicas).

- **Módulo descriptivo**: En `PCA.py` se implementa un modelo de Análisis de Componentes Principales. Se calcula la matriz de rotación mediante el método de Jacobi y se incluye la visualización del biplot en el espacio de las dos primeras componentes principales.

- **Módulo exploratorio**: En `CA.py` se implementa el método de Clustering Jerárquico. Se define una estructura de datos para clusterings en `CLUSTER.py` y se aplica la variante del algoritmo elegida por el usuario que depende de la distancia original (euclídea, Manhattan) y la hiperdistancia o tipo de enlace (simple, completo, promedio, Ward).

- **Módulo de grafos**: En `GRAPH.py` se implementan distintos tipos de grafos (simples, dirigidos, con pesos) y algoritmos de búsqueda clásicos (profundidad, anchura, Dijkstra) usando las estructuras de datos correspondientes.

### Roadmap

- [ ] **MLR.py** — Tests de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov)
- [ ] **FA.py** — Análisis Factorial + test de esfericidad (Bartlett)
- [ ] **MDS.py** — Escalamiento Multidimensional
- [ ] **DA.py** — Análisis Discriminante
- [ ] **KMEANS.py** — Clustering K-Means (método no jerárquico)
- [ ] **ISOMAP.py** — Isomap, reducción de dimensionalidad no lineal apoyada en `GRAPH.py` y `MDS.py`

---

## English <a name="english"></a>

A collection of multivariate statistical models for description, dimensionality reduction, and data exploration. It aims to be a pedagogical, from-scratch implementation of the core algorithms in data science, with special attention to their practical implementation.

- **Linear algebra module**: `LINALG.py` contains an implementation of **QR decomposition** of a matrix using **Householder reflections** (linear regression), as well as an implementation of the **Jacobi method** (principal component analysis).

- **Regression module**: `MLR.py` implements a Multiple Linear Regression model using QR decomposition to solve the normal equations of the regression. It includes diagnostics for the model's assumptions, such as estimators of the design matrix conditioning (collinearity), and hypothesis tests on the estimators including confidence intervals and Cook's distances (outlier detection).

- **Descriptive module**: `PCA.py` implements a Principal Component Analysis model. The rotation matrix is computed using the Jacobi method, and a biplot visualization in the space of the first two principal components is included.

- **Exploratory module**: `CA.py` implements Hierarchical Clustering. A data structure for clusterings is defined in `CLUSTER.py`, and the algorithm variant chosen by the user is applied, depending on the original distance metric (Euclidean, Manhattan) and the linkage type (single, complete, average, Ward).

- **Graph module**: `GRAPH.py` implements various types of graphs (simple, directed, weighted) and classic search algorithms (depth-first, breadth-first, Dijkstra) using their corresponding data structures.

### Roadmap

- [ ] **MLR.py** — Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
- [ ] **FA.py** — Factor Analysis + sphericity test (Bartlett)
- [ ] **MDS.py** — Multidimensional Scaling
- [ ] **DA.py** — Discriminant Analysis
- [ ] **KMEANS.py** — K-Means Clustering (non-hierarchical method)
- [ ] **ISOMAP.py** — Isomap, non-linear dimensionality reduction built on `GRAPH.py` and `MDS.py`
