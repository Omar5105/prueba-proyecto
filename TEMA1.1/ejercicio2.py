import os
import cv2
import matplotlib.pyplot as plt

# 1. Leer la imagen directamente desde la ruta especificada
ruta = "imagenes/ejercicio1/trapesio.jpeg"
img_bgr = cv2.imread(ruta)

if img_bgr is None:
  print("No se pudo leer la imagen:", ruta)
else:

  base, ext = os.path.splitext(ruta)
  
  # 2. Convertir a gris
  gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  # 3. Guardar la imagen en gris con imwrite
  ruta_salida = "imagenes/ejercicio1"
  cv2.imwrite(base + "_gray" + ext, gris)    
  print("Imagen en gris guardada en:", ruta_salida)
  
  # 4. Convertir a blanco y negro
  _, bn = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
  
  # 5. Guarla la imagen en blanco y negro
  ruta_salida = "imagenes/ejercicio1"
  cv2.imwrite(base + "_bn" + ext, bn)
  print("Imagen en blanco y negro guardada en:", ruta_salida)
  
  # 4. Preparar la imagen para Matplotlib (convertir BGR a RGB)
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  

  # 5. Mostrar original y gris en un plot
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 3, 1)
  plt.imshow(img_rgb)
  plt.title("Original: brazo.jpg")
  plt.axis("off")

  plt.subplot(1, 3, 2)
  plt.imshow(gris, cmap="gray")
  plt.title("Grises")
  plt.axis("off")
  
  plt.subplot(1, 3, 3)
  plt.imshow(bn, cmap="gray")
  plt.title("Blanco y Negro")
  plt.axis("off")


  plt.tight_layout()
  plt.show()
