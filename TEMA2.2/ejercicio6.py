import cv2
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path


ruta = "EJERCICIO6/letrero.jpg"
img_color = cv2.imread(ruta)

if img_color is None:
  print("No se pudo leer la imagen:", ruta)

# Obtener las dimensiones de la imagen
alto, ancho = img_color.shape[:2]

#######TRANSLACIÒN#######

# Desplazamiento en píxeles
dx = 150 # Positivo: derecha; negativo: izquierda
dy = 130 # Positivo: abajo; negativo: arriba

# Matriz de traslación
matriz = np.float32([
    [1, 0, dx],
    [0, 1, dy]
])

# Aplicar la transformación
img_trasladada = cv2.warpAffine(img_color, matriz, (ancho, alto))

####ROTACIÒN######

# Dimensiones y centro de la imagen
alto, ancho = img_color.shape[:2]
centro = (ancho / 2, alto / 2)

# Ángulo positivo: giro antihorario; negativo: horario
angulo = 45

# Matriz de rotación: escala 1.0 mantiene el tamaño
matriz_rotacion = cv2.getRotationMatrix2D(centro, angulo, 1.0)

# Aplicar la rotación
img_rotada = cv2.warpAffine(img_color, matriz_rotacion, (ancho, alto))

####REFLEXIÓN#####

# Reflexión horizontal: intercambia izquierda y derecha
img_reflejada = cv2.flip(img_color, -1)

###ESCALADO DE IMAGEN######

# Factores de escala: mayor que 1 amplía; entre 0 y 1 reduce
escala_x = 1.5
escala_y = 1.5

# Aplicar el escalado
img_escalada = cv2.resize(
    img_color,
    None,
    fx=escala_x,
    fy=escala_y,
    interpolation=cv2.INTER_LINEAR
)


####SESGADO#####

# Factores de sesgado (0 significa sin inclinación)
sesgo_x = 0.3
sesgo_y = 0.0

# Matriz de sesgado
matriz_sesgado = np.float32([
    [1, sesgo_x, 0],
    [sesgo_y, 1, 0]
])

# Aplicar el sesgado
img_sesgada = cv2.warpAffine(
    img_color, matriz_sesgado, (ancho, alto)
)


#####SIMILITUD#######

# Dimensiones y centro de la imagen
alto, ancho = img_color.shape[:2]
centro = (ancho / 2, alto / 2)

# Parámetros
angulo_sim = 30
escala_sim = 0.8
dx_sim = 40
dy_sim = 20

# Matriz de rotación y escalado uniforme
matriz_similitud = cv2.getRotationMatrix2D(
    centro, angulo_sim, escala_sim
)

# Agregar traslación
matriz_similitud[0, 2] += dx_sim
matriz_similitud[1, 2] += dy_sim

# Aplicar la transformación
img_similitud = cv2.warpAffine(
    img_color, matriz_similitud, (ancho, alto)
)

# Guardar en la carpeta EJERCICIO6
carpeta_salida = Path(r"EJERCICIO6")
carpeta_salida.mkdir(parents=True, exist_ok=True)

# Guardar la imagen

cv2.imwrite(str(carpeta_salida / "imagen_trasladada.jpg"), img_trasladada)
cv2.imwrite(str(carpeta_salida / "imagen_rotada.jpg"), img_rotada)
cv2.imwrite(str(carpeta_salida / "imagen_reflejada.jpg"), img_reflejada)
cv2.imwrite(str(carpeta_salida / "imagen_escalada.jpg"), img_escalada)
cv2.imwrite(str(carpeta_salida / "imagen_sesgada.jpg"), img_sesgada)
cv2.imwrite(str(carpeta_salida / "imagen_similitud.jpg"), img_similitud)

# Mostrar el resultados
plt.figure(figsize=(15, 10))



plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("Imagen original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(cv2.cvtColor(img_trasladada, cv2.COLOR_BGR2RGB))
plt.title("Imagen trasladada")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(img_rotada, cv2.COLOR_BGR2RGB))
plt.title(f"Imagen rotada: {angulo}º")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(img_reflejada, cv2.COLOR_BGR2RGB))
plt.title("Imagen con reflexión")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(img_escalada, cv2.COLOR_BGR2RGB))
plt.title(f"Original: {img_color.shape[1]} × {img_color.shape[0]} px\n"
f"Escalada: {img_escalada.shape[1]} × {img_escalada.shape[0]} px")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(img_sesgada, cv2.COLOR_BGR2RGB))
plt.title("Imagen sesgada")
plt.axis("off")


plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(img_similitud, cv2.COLOR_BGR2RGB))
plt.title("Imagen aplicada similitud")
plt.axis("off")

plt.tight_layout()
plt.show()
