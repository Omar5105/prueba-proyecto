import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


# Rutas de lectura y guardado
ruta_imagen = "puntosylineas.gif"
carpeta_resultados = "IMAGENES_EJERCICIO10/resultados"

Path(carpeta_resultados).mkdir(parents=True, exist_ok=True)


# Leer el GIF y convertirlo a escala de grises
try:
    imagen_pil = Image.open(ruta_imagen).convert("L")
    imagen_gris = np.array(imagen_pil)
except FileNotFoundError:
    print(f"No se encontró la imagen: {ruta_imagen}")
    raise SystemExit(1)


# Binarizar: los elementos blancos quedan en 255 y el fondo en 0
_, binaria = cv2.threshold(
    imagen_gris,
    127,
    255,
    cv2.THRESH_BINARY
)


# Crear un elemento estructurante elíptico de 9 por 9 píxeles
elemento = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)


# Aplicar apertura morfológica: erosión seguida de dilatación
erosion = cv2.erode(
    binaria,
    elemento,
    iterations=1
)

puntos = cv2.dilate(
    erosion,
    elemento,
    iterations=1
)


# Restar los puntos detectados para conservar las líneas
lineas = cv2.subtract(
    binaria,
    puntos
)


# Combinar puntos y líneas para comprobar la separación
reconstruccion = cv2.bitwise_or(
    puntos,
    lineas
)


# Crear una imagen a color para distinguir los resultados:
# puntos en verde y líneas en rojo
resultado_color = np.zeros(
    (binaria.shape[0], binaria.shape[1], 3),
    dtype=np.uint8
)

resultado_color[puntos > 0] = (0, 255, 0)
resultado_color[lineas > 0] = (0, 0, 255)


# Ampliar el elemento estructurante para visualizarlo
elemento_ampliado = cv2.resize(
    elemento,
    (180, 180),
    interpolation=cv2.INTER_NEAREST
)


# Guardar las imágenes de salida
cv2.imwrite(
    f"{carpeta_resultados}/puntos_separados.png",
    puntos
)

cv2.imwrite(
    f"{carpeta_resultados}/lineas_separadas.png",
    lineas
)

cv2.imwrite(
    f"{carpeta_resultados}/resultado_coloreado.png",
    resultado_color
)


# Mostrar las etapas del proceso
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(imagen_gris, cmap="gray")
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(binaria, cmap="gray")
plt.title("Imagen binarizada")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(elemento_ampliado, cmap="gray")
plt.title("Elemento estructurante")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(erosion, cmap="gray")
plt.title("Después de la erosión")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(puntos, cmap="gray")
plt.title("Puntos separados")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(lineas, cmap="gray")
plt.title("Líneas separadas")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(reconstruccion, cmap="gray")
plt.title("Puntos y líneas combinados")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(resultado_color, cv2.COLOR_BGR2RGB))
plt.title("Puntos verdes y líneas rojas")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    f"{carpeta_resultados}/proceso_ejercicio_10.png",
    dpi=150
)

print(f"Resultados guardados en: {carpeta_resultados}")

plt.show()
