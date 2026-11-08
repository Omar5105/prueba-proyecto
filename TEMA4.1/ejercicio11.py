import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# Ruta de la imagen y carpeta para los resultados
ruta_imagen = "cubo1.jpeg"
carpeta_resultados = "IMAGENES_EJERCICIO11/resultados"

Path(carpeta_resultados).mkdir(parents=True, exist_ok=True)


# Parámetros ajustables
tipo_filtro = "promedio"   # Escribe "gaussiano" o "promedio"
ancho_filtro = 7
alto_filtro = 7
desviacion_estandar = 1.5

umbral_bajo = 30
umbral_alto = 130


# El tamaño del filtro debe ser impar
if ancho_filtro % 2 == 0:
    ancho_filtro += 1

if alto_filtro % 2 == 0:
    alto_filtro += 1


# Leer la imagen directamente en escala de grises
imagen_gris = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen_gris is None:
    print(f"No se pudo leer la imagen: {ruta_imagen}")
    raise SystemExit(1)


# Aplicar el filtro seleccionado
if tipo_filtro == "gaussiano":
    imagen_gaussiana = cv2.GaussianBlur(
        imagen_gris,
        (ancho_filtro, alto_filtro),
        desviacion_estandar
    )
    imagen_promedio = cv2.blur(
        imagen_gris,
        (ancho_filtro, alto_filtro)
    )
    imagen_suavizada = imagen_gaussiana

elif tipo_filtro == "promedio":
    imagen_promedio = cv2.blur(
        imagen_gris,
        (ancho_filtro, alto_filtro)
    )
    imagen_gaussiana = cv2.GaussianBlur(
        imagen_gris,
        (ancho_filtro, alto_filtro),
        desviacion_estandar
    )
    imagen_suavizada = imagen_promedio

else:
    print("El tipo de filtro debe ser 'gaussiano' o 'promedio'.")
    raise SystemExit(1)


# Detectar bordes mediante Canny
bordes = cv2.Canny(
    imagen_suavizada,
    umbral_bajo,
    umbral_alto
)


# Buscar contornos en la imagen de bordes
contornos_externos, _ = cv2.findContours(
    bordes.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contornos_todos, _ = cv2.findContours(
    bordes.copy(),
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)


# Dibujar los contornos detectados
imagen_contornos_externos = cv2.cvtColor(
    imagen_gris,
    cv2.COLOR_GRAY2BGR
)

imagen_contornos_todos = cv2.cvtColor(
    imagen_gris,
    cv2.COLOR_GRAY2BGR
)

cv2.drawContours(
    imagen_contornos_externos,
    contornos_externos,
    -1,
    (0, 255, 0),
    2
)

cv2.drawContours(
    imagen_contornos_todos,
    contornos_todos,
    -1,
    (0, 255, 0),
    2
)


print(f"Filtro seleccionado: {tipo_filtro}")
print(f"Tamaño del filtro: {ancho_filtro} x {alto_filtro}")
print(f"Umbral bajo de Canny: {umbral_bajo}")
print(f"Umbral alto de Canny: {umbral_alto}")
print(f"Contornos externos: {len(contornos_externos)}")
print(f"Todos los contornos: {len(contornos_todos)}")


# Mostrar las etapas del procesamiento
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.imshow(imagen_gris, cmap="gray")
plt.title("Imagen en escala de grises")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(imagen_gaussiana, cmap="gray")
plt.title("Filtro Gaussiano")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(imagen_promedio, cmap="gray")
plt.title("Filtro promedio")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(imagen_suavizada, cmap="gray")
plt.title(f"Filtro usado: {tipo_filtro}")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(bordes, cmap="gray")
plt.title("Bordes con Canny")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(imagen_contornos_externos, cv2.COLOR_BGR2RGB))
plt.title("Contornos externos")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(imagen_contornos_todos, cv2.COLOR_BGR2RGB))
plt.title("Todos los contornos")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.imshow(bordes, cmap="gray")
plt.title(
    f"Canny: bajo={umbral_bajo}, alto={umbral_alto}"
)
plt.axis("off")

plt.tight_layout()

# Guardar los resultados
cv2.imwrite(
    f"{carpeta_resultados}/imagen_suavizada.png",
    imagen_suavizada
)

cv2.imwrite(
    f"{carpeta_resultados}/bordes_canny.png",
    bordes
)

cv2.imwrite(
    f"{carpeta_resultados}/contornos.png",
    imagen_contornos_todos
)

plt.savefig(
    f"{carpeta_resultados}/proceso_ejercicio_11.png",
    dpi=150
)

print(f"Resultados guardados en: {carpeta_resultados}")

plt.show()
