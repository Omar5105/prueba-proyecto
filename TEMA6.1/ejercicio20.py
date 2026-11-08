import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RUTA_IMAGEN = Path("/home/ubuntu/UA_PROCESAMIENTO/EJERCICIOS BOLETINES /imagenesusadas/B1.jpeg")


CARPETA_BASE = Path(__file__).resolve().parent / "/home/ubuntu/UA_PROCESAMIENTO/EJERCICIOS BOLETINES /imagenesusadas/base_datos_letra"


UMBRAL_RECHAZO = 9


def preparar_letra(imagen):
    # Acepta imágenes en color o en escala de grises
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen.copy()

    # Binarización automática
    _, binaria = cv2.threshold(
        gris,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # La letra debe quedar blanca y el fondo negro
    if np.count_nonzero(binaria) > binaria.size * 0.5:
        binaria = cv2.bitwise_not(binaria)

    # Encontrar el contorno exterior más grande para recortar la letra
    contornos, _ = cv2.findContours(
        binaria.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None, None, None, None

    contorno_principal = max(contornos, key=cv2.contourArea)
    x, y, ancho, alto = cv2.boundingRect(contorno_principal)

    # Recortar la letra; la máscara conserva sus partes interiores
    recorte = binaria[y:y + alto, x:x + ancho]

    # Colocar la letra centrada en un cuadro cuadrado
    lado = max(ancho, alto) + 20
    cuadro = np.zeros((lado, lado), dtype=np.uint8)

    x_inicio = (lado - ancho) // 2
    y_inicio = (lado - alto) // 2

    cuadro[y_inicio:y_inicio + alto, x_inicio:x_inicio + ancho] = recorte

    # Todas las letras quedan del mismo tamaño
    normalizada = cv2.resize(cuadro, (200, 200))

    # Calcular los momentos de Hu sobre la máscara binaria
    momentos = cv2.moments(normalizada, binaryImage=True)
    hu = cv2.HuMoments(momentos).flatten()

    # Transformación para comparar los valores de Hu
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    return normalizada, hu, contorno_principal, (x, y, ancho, alto)


# Leer la imagen de prueba
imagen = cv2.imread(str(RUTA_IMAGEN))

if imagen is None:
    raise SystemExit(f"No se pudo leer la imagen: {RUTA_IMAGEN}")

mascara_prueba, hu_prueba, contorno_prueba, caja_prueba = preparar_letra(imagen)

if mascara_prueba is None:
    raise SystemExit("No se encontró una letra en la imagen de prueba.")

contornos_mascara, jerarquia = cv2.findContours(
    mascara_prueba.copy(),
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

huecos = 0
if jerarquia is not None:
    huecos = sum(1 for elemento in jerarquia[0] if elemento[3] != -1)

print("Huecos interiores:", huecos)

# Cargar las referencias de A y B
nombres = []
momentos_referencia = []
mascaras_referencia = {"A": None, "B": None}

for letra in ["A", "B"]:
    carpeta_letra = CARPETA_BASE / letra

    if not carpeta_letra.exists():
        raise SystemExit(f"No existe la carpeta de referencias: {carpeta_letra}")

    archivos = [
        archivo for archivo in carpeta_letra.iterdir()
        if archivo.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ]

    print(f"\nBuscando imágenes en: {carpeta_letra}")
    print(f"Archivos encontrados: {len(archivos)}")

    for archivo in archivos:
        print(f"  - {archivo.name}")

    if not archivos:
        raise SystemExit(f"No hay imágenes dentro de: {carpeta_letra}")

    for archivo in archivos:
        imagen_ref = cv2.imread(str(archivo))

        if imagen_ref is None:
            print(f"No se pudo leer la referencia: {archivo}")
            continue

        mascara_ref, hu_ref, _, _ = preparar_letra(imagen_ref)

        if mascara_ref is None:
            print(f"No se encontró una letra en: {archivo}")
            continue

        nombres.append(letra)
        momentos_referencia.append(hu_ref)

        print(f"Referencia cargada correctamente: {archivo}")

        # Guardar una referencia para mostrarla en los subplots
        if mascaras_referencia[letra] is None:
            mascaras_referencia[letra] = mascara_ref

if not nombres:
    raise SystemExit("No se pudo cargar ninguna referencia.")

# Calcular la distancia mínima a cada letra
distancias_por_letra = []

for letra in ["A", "B"]:
    distancias = [
        np.linalg.norm(hu_prueba - hu_ref)
        for nombre, hu_ref in zip(nombres, momentos_referencia)
        if nombre == letra
    ]

    if distancias:
        distancias_por_letra.append(min(distancias))
    else:
        distancias_por_letra.append(float("inf"))

indice = int(np.argmin(distancias_por_letra))
letra_detectada = ["A", "B"][indice]
distancia_minima = distancias_por_letra[indice]

print("\nDistancia a cada imagen de referencia:")

for nombre, hu_ref in zip(nombres, momentos_referencia):
    distancia = np.linalg.norm(hu_prueba - hu_ref)
    print(f"Letra {nombre}: {distancia:.2f}")

# Aplicar rechazo solo si luego se configura un umbral
if UMBRAL_RECHAZO is not None and distancia_minima > UMBRAL_RECHAZO:
    resultado = "No es A ni B"
else:
    resultado = letra_detectada

if letra_detectada == "A" and huecos != 1:
    resultado = "No es A ni B"
elif letra_detectada == "B" and huecos != 2:
    resultado = "No es A ni B"
else:
    resultado = letra_detectada

print("Distancia a A:", distancias_por_letra[0])
print("Distancia a B:", distancias_por_letra[1])
print("Resultado:", resultado)

# Dibujar el contorno encontrado en la imagen original
imagen_contorno = imagen.copy()

cv2.drawContours(
    imagen_contorno,
    [contorno_prueba],
    -1,
    (0, 255, 0),
    2
)

# Mostrar resultados
plt.figure(figsize=(15, 8))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 4, 2)
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
plt.imshow(gris, cmap="gray")
plt.title("Escala de grises")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(mascara_prueba, cmap="gray")
plt.title("Letra binarizada")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(imagen_contorno, cv2.COLOR_BGR2RGB))
plt.title("Letra localizada")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(mascaras_referencia["A"], cmap="gray")
plt.title("Referencia A")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(mascaras_referencia["B"], cmap="gray")
plt.title("Referencia B")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.bar(["A", "B"], distancias_por_letra, color=["royalblue", "orange"])
plt.title("Distancia a cada letra")
plt.ylabel("Distancia")

plt.subplot(2, 4, 8)
plt.text(
    0.5,
    0.5,
    f"Letra reconocida:\n{resultado}",
    ha="center",
    va="center",
    fontsize=18
)
plt.title("Resultado")
plt.axis("off")

plt.tight_layout()
plt.show()
