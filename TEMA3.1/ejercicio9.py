import cv2
import numpy as np
import matplotlib.pyplot as plt


ruta = "EJERCICIO9/brazo.jpg"

img_gris = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
if img_gris is None:
    raise FileNotFoundError(f"No se pudo leer: {ruta}")
    
_, img_binaria= cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY_INV)

tamano = 9

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (tamano, tamano)
)

print(kernel)

# Reduce las regiones blancas
img_erosion = cv2.erode(img_gris, kernel, iterations=1)

# Amplía las regiones blancas
img_dilatacion = cv2.dilate(img_gris, kernel, iterations=1)

# Erosión seguida de dilatación
img_apertura = cv2.morphologyEx(
    img_gris, cv2.MORPH_OPEN, kernel
)

# Dilatación seguida de erosión
img_cierre = cv2.morphologyEx(
    img_gris, cv2.MORPH_CLOSE, kernel
)

# Diferencia entre dilatación y erosión
img_gradiente = cv2.morphologyEx(
    img_gris, cv2.MORPH_GRADIENT, kernel
)

# Original menos apertura
img_tophat = cv2.morphologyEx(
    img_gris, cv2.MORPH_TOPHAT, kernel
)

# Cierre menos original
img_blackhat = cv2.morphologyEx(
    img_gris, cv2.MORPH_BLACKHAT, kernel
)

imagenes = [
    img_gris, img_erosion, img_dilatacion, img_apertura,
    img_cierre, img_gradiente, img_tophat, img_blackhat
]

titulos = [
    "Original binaria", "Erosión", "Dilatación", "Apertura",
    "Cierre", "Gradiente", "Top-hat", "Black-hat"
]

plt.figure(figsize=(14, 8))

for i, (imagen, titulo) in enumerate(zip(imagenes, titulos)):
    plt.subplot(2, 4, i + 1)
    plt.imshow(imagen, cmap="gray", vmin=0, vmax=255)
    plt.title(f"{titulo}\nKernel: {tamano} × {tamano}")
    plt.axis("off")

plt.tight_layout()
plt.show()
