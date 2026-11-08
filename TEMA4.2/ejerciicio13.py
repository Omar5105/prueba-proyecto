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

if img_color is None:
    raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")


# CONVERTIR A ESCALA DE GRISES
img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


# SUAVIZADO GAUSSIANO
tamano_filtro = 3
sigma = 1.0

img_filtrada = cv2.GaussianBlur(
    img_gris,
    (tamano_filtro, tamano_filtro),
    sigma
)


# DELIMITAR EL CUBO CON CUATRO CLICS
fig_seleccion = plt.figure(figsize=(9, 8))

plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title(
    "Marca las 4 esquinas exteriores del cubo en este orden:\n"
    "1. Superior izquierda    2. Superior derecha\n"
    "3. Inferior derecha     4. Inferior izquierda"
)
plt.axis("off")
plt.tight_layout()

puntos = plt.ginput(4, timeout=0)

plt.close(fig_seleccion)

if len(puntos) != 4:
    raise ValueError("Debes seleccionar las cuatro esquinas.")

vertices = np.round(puntos).astype(np.int32)

mascara_cubo = np.zeros_like(img_gris)
cv2.fillPoly(mascara_cubo, [vertices], 255)


# MOSTRAR LA ZONA SELECCIONADA SOBRE UNA COPIA
img_zona = img_color.copy()

cv2.polylines(
    img_zona,
    [vertices],
    isClosed=True,
    color=(0, 255, 0),
    thickness=2
)


# PARÁMETROS COMUNES PARA COMPARAR HARRIS Y SHI-TOMASI
max_esquinas = 100
calidad = 0.02
distancia_minima = 30
bloque = 5


# DETECTOR DE HARRIS
k = 0.04

esquinas_harris = cv2.goodFeaturesToTrack(
    img_filtrada,
    maxCorners=max_esquinas,
    qualityLevel=calidad,
    minDistance=distancia_minima,
    mask=mascara_cubo,
    blockSize=bloque,
    useHarrisDetector=True,
    k=k
)

img_harris = img_color.copy()
cantidad_harris = 0

if esquinas_harris is not None:
    cantidad_harris = len(esquinas_harris)

    for esquina in esquinas_harris:
        x, y = np.round(esquina.ravel()).astype(int)

        cv2.circle(
            img_harris,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# DETECTOR DE SHI-TOMASI
esquinas_shi = cv2.goodFeaturesToTrack(
    img_filtrada,
    maxCorners=max_esquinas,
    qualityLevel=calidad,
    minDistance=distancia_minima,
    mask=mascara_cubo,
    blockSize=bloque,
    useHarrisDetector=False
)

img_shi = img_color.copy()
cantidad_shi = 0

if esquinas_shi is not None:
    cantidad_shi = len(esquinas_shi)

    for esquina in esquinas_shi:
        x, y = np.round(esquina.ravel()).astype(int)

        cv2.circle(
            img_shi,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )


# DETECTOR FAST
umbral_fast = 25

detector_fast = cv2.FastFeatureDetector_create(
    threshold=umbral_fast,
    nonmaxSuppression=True
)

puntos_fast = detector_fast.detect(
    img_filtrada,
    mascara_cubo
)

img_fast = img_color.copy()
cantidad_fast = len(puntos_fast)

# Dibujar círculos iguales a los de los otros detectores
for punto in puntos_fast:
    x, y = np.round(punto.pt).astype(int)

    cv2.circle(
        img_fast,
        (x, y),
        4,
        (0, 0, 255),
        -1
    )


# MOSTRAR CANTIDADES EN LA TERMINAL
print("Harris:", cantidad_harris, "esquinas")
print("Shi-Tomasi:", cantidad_shi, "esquinas")
print("FAST:", cantidad_fast, "esquinas")


# MOSTRAR RESULTADOS
plt.figure(figsize=(16, 9))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_zona, cv2.COLOR_BGR2RGB))
plt.title("Imagen original y zona seleccionada")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(img_filtrada, cmap="gray", vmin=0, vmax=255)
plt.title(
    f"Imagen en grises y suavizada\n"
    f"Filtro: {tamano_filtro} × {tamano_filtro}, sigma={sigma}"
)
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(mascara_cubo, cmap="gray", vmin=0, vmax=255)
plt.title("Máscara: búsqueda dentro de la zona blanca")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
plt.title(
    f"Harris: {cantidad_harris} esquinas\n"
    f"Calidad={calidad}, distancia={distancia_minima} px"
)
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(img_shi, cv2.COLOR_BGR2RGB))
plt.title(
    f"Shi-Tomasi: {cantidad_shi} esquinas\n"
    f"Calidad={calidad}, distancia={distancia_minima} px"
)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
plt.title(
    f"FAST: {cantidad_fast} esquinas\n"
    f"Umbral={umbral_fast}"
)
plt.axis("off")

plt.tight_layout()
plt.show()
