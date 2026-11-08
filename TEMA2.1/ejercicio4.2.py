import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Rutas de lectura y guardado
ruta_imagen = "CUBO/cubo2.jpeg"
carpeta_resultados = "IMAGENES_EJERCICIO/resultados"

Path(carpeta_resultados).mkdir(parents=True, exist_ok=True)


# Leer la imagen
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"No se pudo leer la imagen: {ruta_imagen}")
    raise SystemExit(1)


# Recortar un pequeño margen de la fotografía
alto, ancho = imagen.shape[:2]
margen_x = int(ancho * 0.04)
margen_y = int(alto * 0.04)

cara = imagen[
    margen_y:alto - margen_y,
    margen_x:ancho - margen_x
].copy()


# Paleta de referencia en formato BGR
colores_bgr = {
    "ROJO": (0, 0, 255),
    "NARANJA": (0, 165, 255),
    "AMARILLO": (0, 255, 255),
    "VERDE": (0, 255, 0),
    "AZUL": (255, 0, 0),
    "BLANCO": (255, 255, 255)
}


# Tomar muestras del centro de cada una de las nueve casillas
alto_cara, ancho_cara = cara.shape[:2]
paso_x = ancho_cara / 3
paso_y = alto_cara / 3

muestras_kmeans = []
rangos_casillas = []
coordenadas_casillas = []
muestras_visual = np.zeros_like(cara)

inicio = 0

for fila in range(3):
    for columna in range(3):
        x0 = int(columna * paso_x)
        x1 = int((columna + 1) * paso_x)
        y0 = int(fila * paso_y)
        y1 = int((fila + 1) * paso_y)

        ancho_casilla = x1 - x0
        alto_casilla = y1 - y0

        # Usar solo el 40 % central de la casilla
        sx0 = x0 + int(ancho_casilla * 0.30)
        sx1 = x1 - int(ancho_casilla * 0.30)
        sy0 = y0 + int(alto_casilla * 0.30)
        sy1 = y1 - int(alto_casilla * 0.30)

        muestra = cara[sy0:sy1, sx0:sx1]

        # Reducir la cantidad de píxeles muestreando uno de cada cuatro
        pixeles = muestra[::4, ::4].reshape(-1, 3).astype(np.float32)

        muestras_kmeans.append(pixeles)

        fin = inicio + len(pixeles)
        rangos_casillas.append((inicio, fin))
        coordenadas_casillas.append((x0, y0, x1, y1, sx0, sy0, sx1, sy1))
        inicio = fin

        # Mostrar la zona central utilizada
        muestras_visual[sy0:sy1, sx0:sx1] = muestra


# Unir las nueve muestras para aplicar K-means
datos = np.vstack(muestras_kmeans).astype(np.float32)

K = 6

criterio = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

compactitud, etiquetas, centros = cv2.kmeans(
    datos,
    K,
    None,
    criterio,
    10,
    cv2.KMEANS_PP_CENTERS
)


# Asignar un nombre a cada grupo comparándolo con la paleta de referencia
nombres_grupos = {}

for numero_grupo in range(K):
    centro = centros[numero_grupo]

    nombre_mas_cercano = min(
        colores_bgr,
        key=lambda nombre: np.linalg.norm(
            centro - np.array(colores_bgr[nombre], dtype=np.float32)
        )
    )

    nombres_grupos[numero_grupo] = nombre_mas_cercano

    print(
        f"Grupo {numero_grupo}: "
        f"centro BGR = {centro.astype(int)}, "
        f"clasificado como {nombre_mas_cercano}"
    )


# Determinar el color dominante de cada casilla
resultado = cara.copy()
mapa_colores = np.zeros((3, 3, 3), dtype=np.uint8)
matriz_colores = []

for indice_casilla, (fila, columna) in enumerate(
    [(f, c) for f in range(3) for c in range(3)]
):
    inicio, fin = rangos_casillas[indice_casilla]
    etiquetas_casilla = etiquetas[inicio:fin].flatten()

    cantidades = np.bincount(etiquetas_casilla, minlength=K)
    grupo_dominante = int(np.argmax(cantidades))
    color = nombres_grupos[grupo_dominante]

    fila_matriz = fila
    columna_matriz = columna

    if columna_matriz == 0:
        matriz_colores.append([])

    matriz_colores[fila_matriz].append(color)

    x0, y0, x1, y1, sx0, sy0, sx1, sy1 = (
        coordenadas_casillas[indice_casilla]
    )

    # Dibujar el borde de la casilla y de la muestra central
    cv2.rectangle(resultado, (x0, y0), (x1, y1), (0, 0, 0), 2)
    cv2.rectangle(resultado, (sx0, sy0), (sx1, sy1), (255, 255, 255), 2)

    # Escribir el color asignado
    cv2.putText(
        resultado,
        color,
        (x0 + 8, y0 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2
    )

    # Crear el mapa de colores 3x3
    mapa_colores[fila, columna] = colores_bgr[color]

    print(
        f"Casilla ({fila + 1}, {columna + 1}): "
        f"{color} | grupo K-means: {grupo_dominante}"
    )


# Imprimir la matriz detectada
print("\nMatriz de colores detectada:")
for fila in matriz_colores:
    print(fila)


# Ampliar la matriz para visualizarla mejor
mapa_ampliado = cv2.resize(
    mapa_colores,
    (300, 300),
    interpolation=cv2.INTER_NEAREST
)


# Separar los canales B, G y R para visualizar las características usadas
canal_b, canal_g, canal_r = cv2.split(cara)


# Mostrar los pasos del procesamiento
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(cara, cv2.COLOR_BGR2RGB))
plt.title("Cara recortada")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(canal_b, cmap="gray")
plt.title("Canal azul B")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(canal_g, cmap="gray")
plt.title("Canal verde G")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(canal_r, cmap="gray")
plt.title("Canal rojo R")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(muestras_visual, cv2.COLOR_BGR2RGB))
plt.title("Muestras centrales")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
plt.title("Clasificación de las casillas")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(mapa_ampliado, cv2.COLOR_BGR2RGB))
plt.title("Matriz visual de colores")
plt.axis("off")

plt.tight_layout()

# Guardar los resultados en la carpeta establecida
plt.savefig(
    f"{carpeta_resultados}/cubo4_kmeans.png",
    dpi=150
)

cv2.imwrite(
    f"{carpeta_resultados}/cubo4_matriz_kmeans.png",
    mapa_ampliado
)

print(f"\nResultados guardados en: {carpeta_resultados}")

plt.show()
