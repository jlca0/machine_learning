# Machine Learning from First Principles

Implementación educativa y rigurosa de algoritmos clásicos de Machine Learning y álgebra lineal desarrollada desde los principios fundamentales (*First Principles*), utilizando únicamente librerías base de computación científica en Python, como **NumPy**, **SciPy** y **Matplotlib**.

## 📂 Estructura del Proyecto

```text
machine_learning/
├── README.md              # Documentación del proyecto
├── src/
│   └── machine_learning/
│       ├── pca.py         # Análisis de componentes principales (PCA).
│       ├── linalg.py      # Descomposiciones y herramientas de álgebra lineal (QR, SVD).
│       ├── mlr.py         # Regresión lineal múltiple.
│       ├── kNN.py         # Algoritmo de K-Vecinos más cercanos.
│       ├── mds.py         # Escalado multidimensional.
│       └── ca.py          # Análisis clúster
├── tests/
│   ├── test_linalg.py     # Pruebas unitarias de álgebra lineal.
│   └── test_models.py    # Pruebas unitarias de los modelos de ML.
└── examples/
    └── data/
        └── state.x77.csv
```

## ⚙️ Requisitos e Instalación

Este proyecto requiere **Python 3.9 o superior**. Se recomienda utilizar un entorno virtual para aislar las dependencias.

### 1. Clonar el repositorio

Clona el repositorio y sitúate en su directorio raíz.

### 2. Activar el entorno virtual

Activa el entorno virtual de Python que utilizarás para trabajar con el proyecto.

### 3. Instalar el paquete

Instala el paquete en modo editable (`-e`), lo que permite que los cambios realizados en el código fuente se reflejen directamente sin necesidad de reinstalarlo.

```bash
pip install -e .
```

## 🧪 Ejecución de Pruebas

El proyecto incluye una suite de pruebas unitarias configurada con **Pytest**, diseñada para verificar la precisión matemática de los algoritmos mediante su comparación con librerías de referencia como `scikit-learn` y `scipy`.

Para ejecutar todas las pruebas, utiliza el siguiente comando desde el directorio raíz:

```bash
pytest
```

## 🚀 Ejemplo de Uso

A continuación, se muestra un ejemplo de cómo importar y utilizar el modelo **PCA (Principal Component Analysis)**.

```python
import numpy as np
from machine_learning.pca import PCA

# 1. Generar datos sintéticos de prueba 
rng = np.random.default_rng(42)
X = rng.normal(size=(40, 5))

# 2. Ajustar el modelo PCA
model = PCA(X)
model.fit()

# 3. Consultar los resultados principales
print("Varianza explicada por componente:", model.explained_variance_)
print("Ratio de varianza explicada:", model.explained_variance_ratio_)

# 4. Proyectar los datos en el nuevo espacio de PCs.
X_reduced = model.transform()
print("Forma de los datos proyectados:", X_reduced.shape)
```

## 🛠️ Librerías Utilizadas

- **NumPy**: Operaciones matriciales eficientes y herramientas de álgebra lineal básica.
- **SciPy**: Herramientas complementarias de optimización y álgebra lineal avanzada (`scipy.linalg`).
- **Matplotlib**: Visualización de resultados, datos y proyecciones.

## 📝 Licencia

Este proyecto tiene fines estrictamente educativos y académicos.

Siéntete libre de utilizarlo, modificarlo o adaptarlo para tu propio aprendizaje.