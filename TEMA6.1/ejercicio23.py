from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# CAMBIA ESTAS RUTAS SEGÚN TUS CARPETAS
CARPETA_BASE = Path("/home/ubuntu/UA_PROCESAMIENTO/EJERCICIOS BOLETINES /imagenesusadas/base_datos_letra")
RUTA_PRUEBA = Path("/home/ubuntu/UA_PROCESAMIENTO/EJERCICIOS BOLETINES /imagenesusadas/A2.jpeg")

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}

# Parámetros para considerar una letra reconocida.
# Si rechaza letras correctas, prueba bajando MINIMO_EMPAREJAMIENTOS.
MINIMO_EMPAREJAMIENTOS = 3
DIFERENCIA_MINIMA = 1

# ORB detecta puntos de interés y calcula sus descriptores binarios
orb = cv2.ORB_create(
    nfeatures=1000,
    fastThreshold=5
)

# BFMatcher usa distancia Hamming para comparar descriptores ORB
comparador = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def preparar_imagen(ruta):
    """Binariza, recorta la letra y la centra en un fondo negro."""

    imagen = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")

    _, binaria = cv2.threshold(
        imagen,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Procura que la letra quede blanca y el fondo negro
    if np.mean(binaria) > 127:
        binaria = cv2.bitwise_not(binaria)

    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        raise ValueError(f"No se encontró una letra en: {ruta}")

    contorno = max(contornos, key=cv2.contourArea)
    x, y, ancho, alto = cv2.boundingRect(contorno)
    recorte = binaria[y:y + alto, x:x + ancho]

    # Centrar la letra en una imagen común de 400 x 400
    lienzo = np.zeros((400, 400), dtype=np.uint8)
    margen = 40
    escala = min(
        (400 - 2 * margen) / ancho,
        (400 - 2 * margen) / alto
    )

    nuevo_ancho = max(1, int(ancho * escala))
    nuevo_alto = max(1, int(alto * escala))

    letra = cv2.resize(
        recorte,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_AREA
    )

    x_inicio = (400 - nuevo_ancho) // 2
    y_inicio = (400 - nuevo_alto) // 2

    lienzo[
        y_inicio:y_inicio + nuevo_alto,
        x_inicio:x_inicio + nuevo_ancho
    ] = letra

    return imagen, lienzo


def obtener_descriptores(ruta):
    """Prepara la imagen y obtiene puntos y descriptores ORB."""

    imagen_original, imagen_procesada = preparar_imagen(ruta)
    puntos, descriptores = orb.detectAndCompute(imagen_procesada, None)

    return imagen_original, imagen_procesada, puntos, descriptores


def contar_buenos_emparejamientos(desc_prueba, desc_referencia):
    """Compara dos conjuntos de descriptores y cuenta las coincidencias buenas."""

    if desc_prueba is None or desc_referencia is None:
        return [], float("inf")

    if len(desc_prueba) < 2 or len(desc_referencia) < 2:
        return [], float("inf")

    pares = comparador.knnMatch(
        desc_prueba,
        desc_referencia,
        k=2
    )

    buenos = []

    # Filtro de Lowe: conserva coincidencias más claras
    for par in pares:
        if len(par) == 2:
            mejor, segundo = par

            if mejor.distance < 0.75 * segundo.distance:
                buenos.append(mejor)

    distancia_promedio = (
        np.mean([m.distance for m in buenos])
        if buenos
        else float("inf")
    )

    return buenos, distancia_promedio


# Obtener los descriptores de la imagen de prueba
imagen_original, procesada_prueba, puntos_prueba, desc_prueba = (
    obtener_descriptores(RUTA_PRUEBA)
)

if desc_prueba is None:
    raise ValueError(
        "ORB no encontró suficientes características en la imagen de prueba."
    )

# Comparar la prueba con cada imagen de referencia
resultados_por_letra = {"A": [], "B": []}
mejor_emparejamiento = None

for letra in ("A", "B"):
    carpeta_letra = CARPETA_BASE / letra

    if not carpeta_letra.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta_letra}")

    archivos = sorted(
        archivo for archivo in carpeta_letra.iterdir()
        if archivo.suffix.lower() in EXTENSIONES
    )

    if not archivos:
        raise ValueError(f"No hay imágenes dentro de: {carpeta_letra}")

    for archivo in archivos:
        _, procesada_ref, puntos_ref, desc_ref = obtener_descriptores(archivo)

        buenos, distancia = contar_buenos_emparejamientos(
            desc_prueba,
            desc_ref
        )

        resultados_por_letra[letra].append({
            "archivo": archivo,
            "imagen": procesada_ref,
            "puntos": puntos_ref,
            "descriptores": desc_ref,
            "buenos": buenos,
            "cantidad": len(buenos),
            "distancia": distancia
        })

        print(
            f"{archivo.name} ({letra}): "
            f"{len(buenos)} emparejamientos buenos"
        )

# Para cada letra, conservar la referencia con más emparejamientos
mejor_por_letra = {}

for letra, resultados in resultados_por_letra.items():
    mejor_por_letra[letra] = max(
        resultados,
        key=lambda resultado: resultado["cantidad"]
    )

cantidad_a = mejor_por_letra["A"]["cantidad"]
cantidad_b = mejor_por_letra["B"]["cantidad"]

if cantidad_a > cantidad_b:
    candidata = "A"
    diferencia = cantidad_a - cantidad_b
elif cantidad_b > cantidad_a:
    candidata = "B"
    diferencia = cantidad_b - cantidad_a
else:
    candidata = None
    diferencia = 0

if (
    candidata is not None
    and mejor_por_letra[candidata]["cantidad"] >= MINIMO_EMPAREJAMIENTOS
    and diferencia >= DIFERENCIA_MINIMA
):
    resultado = candidata
else:
    resultado = "No reconocida"

print("\nMejor comparación con A:", cantidad_a, "emparejamientos")
print("Mejor comparación con B:", cantidad_b, "emparejamientos")
print("Resultado:", resultado)

# Mostrar los emparejamientos con la mejor referencia encontrada
if candidata is not None:
    mejor = mejor_por_letra[candidata]
else:
    mejor = (
        mejor_por_letra["A"]
        if cantidad_a >= cantidad_b
        else mejor_por_letra["B"]
    )

imagen_emparejamientos = cv2.drawMatches(
    procesada_prueba,
    puntos_prueba,
    mejor["imagen"],
    mejor["puntos"],
    mejor["buenos"][:40],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(procesada_prueba, cmap="gray")
plt.title("Letra de prueba procesada")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imagen_emparejamientos, cmap="gray")
plt.title(
    f"Coincidencias con referencia {candidata or 'más cercana'}"
)
plt.axis("off")

plt.suptitle(f"Resultado: {resultado}")
plt.tight_layout()
plt.show()
