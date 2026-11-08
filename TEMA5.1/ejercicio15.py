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


# CONVERTIR A ESCALA DE GRISES
img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


# UMBRALIZACIÓN GLOBAL CON VALOR FIJO
umbral_fijo = 210

_, img_global = cv2.threshold(
    img_gris,
    umbral_fijo,
    255,
    cv2.THRESH_BINARY
)


# UMBRALIZACIÓN AUTOMÁTICA DE OTSU
umbral_otsu, img_otsu = cv2.threshold(
    img_gris,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Umbral fijo:", umbral_fijo)
print("Umbral calculado por Otsu:", umbral_otsu)


# MOSTRAR RESULTADOS
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(img_gris, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen en escala de grises")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(img_global, cmap="gray", vmin=0, vmax=255)
plt.title(f"Umbralización global\nUmbral fijo: {umbral_fijo}")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(img_otsu, cmap="gray", vmin=0, vmax=255)
plt.title(f"Umbralización de Otsu\nUmbral calculado: {umbral_otsu:.0f}")
plt.axis("off")

plt.tight_layout()
plt.show()
