import cv2
import matplotlib.pyplot as plt

ruta = "imagenes/ejercicio2/letrero.jpg"

print("Buscando imagen en:", ruta)

img_color = cv2.imread(str(ruta))

if img_color is None:
    print("No se pudo leer la imagen.")
    print("Comprueba que existe en esta ruta:", ruta)

else:
    # IMAGEN EN GRISES
    img_gris = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_gris, cmap="gray", vmin=0, vmax=255)
    plt.title("Imagen en grises")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(hist, color="black")
    plt.title("Histograma")
    plt.xlabel("Nivel de gris (0-255)")
    plt.ylabel("Número de píxeles")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()

# IMAGEN EN COLOR
    colores = ("b", "g", "r")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title("Imagen en color")
    plt.axis("off")

    plt.subplot(1, 2, 2)

    for i, color in enumerate(colores):
        hist = cv2.calcHist([img_color], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)

    plt.title("Histograma por canal")
    plt.xlabel("Intensidad (0-255)")
    plt.ylabel("Número de píxeles")
    plt.xlim([0, 255])

    plt.tight_layout()
    plt.show()
