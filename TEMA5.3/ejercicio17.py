from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# RUTA DE LA IMAGEN
ruta = Path(
    "/home/ubuntu/UA_PROCESAMIENTO/EJERCICIOS BOLETINES /"
    "imagenesusadas/cubo2.jpeg"
)

img_color = cv2.imread(str(ruta))

# NÚMERO DE GRUPOS
k = 12

# Convertir la imagen en una lista de píxeles BGR
pixeles = img_color.reshape((-1, 3))
pixeles = np.float32(pixeles)


# CRITERIOS DE PARADA
max_iteraciones = 100
epsilon = 0.2

criterios = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    max_iteraciones,
    epsilon
)

# Semilla para facilitar la comparación entre ejecuciones
cv2.setRNGSeed(42)


# APLICAR K-MEANS
compactacion, etiquetas, centros = cv2.kmeans(
    pixeles,
    k,
    None,
    criterios,
    10,
    cv2.KMEANS_PP_CENTERS
)


# RECONSTRUIR LA IMAGEN CON LOS COLORES REPRESENTATIVOS
centros_color = np.clip(centros, 0, 255).astype(np.uint8)

img_segmentada = centros_color[etiquetas.ravel()]
img_segmentada = img_segmentada.reshape(img_color.shape)


# MOSTRAR RESULTADOS
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2RGB))
plt.title(f"Segmentación con k-means\nNúmero de grupos: {k}")
plt.axis("off")

plt.tight_layout()
plt.show()
