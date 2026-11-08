import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ruta_imagen = Path("CUBO/cubo4.jpeg")
carpeta_resultados = Path("IMAGENES_EJERCICIO_CUBO/resultados")
carpeta_resultados.mkdir(parents=True, exist_ok=True)

# Leer la imagen
imagen = cv2.imread(str(ruta_imagen))

if imagen is None:
    print(f"No se pudo leer la imagen: {ruta_imagen}")
    raise SystemExit(1)

alto, ancho = imagen.shape[:2]

# Recortar un margen pequeño: en esta foto, la cara ocupa casi toda la imagen
margen_x = int(ancho * 0.04)
margen_y = int(alto * 0.04)

cara = imagen[
    margen_y:alto - margen_y,
    margen_x:ancho - margen_x
].copy()

alto_cara, ancho_cara = cara.shape[:2]
paso_x = ancho_cara / 3
paso_y = alto_cara / 3

# Convertir la cara a HSV para analizar sus colores
cara_hsv = cv2.cvtColor(cara, cv2.COLOR_BGR2HSV)

imagen_clasificada = cara.copy()
matriz_colores = []

for fila in range(3):
    fila_colores = []

    for columna in range(3):
        # Límites de la casilla
        x0 = int(columna * paso_x)
        x1 = int((columna + 1) * paso_x)
        y0 = int(fila * paso_y)
        y1 = int((fila + 1) * paso_y)

        # Tomar solo la zona central para evitar bordes y separaciones
        ancho_casilla = x1 - x0
        alto_casilla = y1 - y0

        muestra = cara_hsv[
            y0 + int(alto_casilla * 0.30):y1 - int(alto_casilla * 0.30),
            x0 + int(ancho_casilla * 0.30):x1 - int(ancho_casilla * 0.30)
        ]

        # Usar el valor mediano de HSV de la muestra
        h, s, v = np.median(
            muestra.reshape(-1, 3),
            axis=0
        ).astype(int)

        # Clasificar el color según el tono y la saturación
        if s < 90 and v > 90:
            color = "BLANCO"
        elif h < 3 or h >= 170:
            color = "ROJO"
        elif h < 18:
            color = "NARANJA"
        elif h < 35:
            color = "AMARILLO"
        elif h < 85:
            color = "VERDE"
        elif h < 135:
            color = "AZUL"
        else:
            color = "OTRO"
        fila_colores.append(color)

        # Dibujar la cuadrícula y el nombre detectado
        cv2.rectangle(
            imagen_clasificada,
            (x0, y0),
            (x1, y1),
            (0, 0, 0),
            2
        )

        centro_x = x0 + ancho_casilla // 2
        centro_y = y0 + alto_casilla // 2

        cv2.putText(
            imagen_clasificada,
            color,
            (centro_x - 45, centro_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            2
        )

        print(
            f"Casilla ({fila + 1}, {columna + 1}): "
            f"{color} | HSV = ({h}, {s}, {v})"
        )

    matriz_colores.append(fila_colores)

print("\nMatriz de colores:")
for fila in matriz_colores:
    print(fila)

# Mostrar la foto original y la clasificación
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imagen_clasificada, cv2.COLOR_BGR2RGB))
plt.title("Colores clasificados")
plt.axis("off")

plt.tight_layout()
plt.savefig(carpeta_resultados / "cubo2_colores.png", dpi=150)
plt.show()
