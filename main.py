

import cv2
import numpy as np
import matplotlib.pyplot as plt

def obtener_mascara_segmentacion(imagen):
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de valores para el color rojo
    rango_rojo_bajo = np.array([0, 100, 100])
    rango_rojo_alto = np.array([10, 255, 255])

    # Crear la máscara de segmentación
    mascara_segmentacion = cv2.inRange(imagen_hsv, rango_rojo_bajo, rango_rojo_alto)

    return mascara_segmentacion

def encontrar_contorno_planta(mascara_segmentacion):
    # Encontrar los contornos
    contornos, _ = cv2.findContours(mascara_segmentacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (la planta segmentada)
    contorno_planta = max(contornos, key=cv2.contourArea)

    return contorno_planta

def obtener_puntos_extremos(contorno):
    # Obtener los puntos extremos del contorno
    puntos_extremos = tuple(contorno[contorno[:, :, 0].argmin()][0]), tuple(contorno[contorno[:, :, 0].argmax()][0])

    return puntos_extremos

# Cargar la imagen de ejemplo
imagen = cv2.imread(r"C:\Users\Usuario\Desktop\pythonProject\ImagenesSEGMENTADAS\segmentado_4_13_03.JPG")

# Obtener la máscara de segmentación
mascara_segmentacion = obtener_mascara_segmentacion(imagen)

# Encontrar el contorno de la planta segmentada
contorno_planta = encontrar_contorno_planta(mascara_segmentacion)

# Obtener los puntos extremos dentro del contorno
puntos_extremos = obtener_puntos_extremos(contorno_planta)

# Calcular el ancho de la planta (en píxeles)
ancho_px = abs(puntos_extremos[1][0] - puntos_extremos[0][0])

# Calcular la altura de la planta (en píxeles)
altura_px = abs(puntos_extremos[1][1] - puntos_extremos[0][1])

# Calcular el área total de la planta (en píxeles cuadrados)
area_px2 = cv2.contourArea(contorno_planta)

# Suponemos una relación aproximada entre píxeles y centímetros
# Esta relación puede necesitar ajustes según la escala real de la imagen
# Por ejemplo, si hay un objeto de longitud conocida en la imagen, se puede utilizar para una calibración más precisa.
pixeles_por_cm = 100 # Por ejemplo, suponemos que 10 píxeles representan aproximadamente 1 cm

# Convertir el ancho y la altura de píxeles a centímetros
ancho_cm = ancho_px / pixeles_por_cm
altura_cm = altura_px / pixeles_por_cm

# Convertir el área de píxeles cuadrados a centímetros cuadrados
area_cm2 = area_px2 / (pixeles_por_cm ** 2)

# Mostrar la imagen con el rectángulo y las medidas calculadas
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.plot([puntos_extremos[0][0], puntos_extremos[1][0]], [puntos_extremos[0][1], puntos_extremos[1][1]], color='red')
plt.text(puntos_extremos[1][0], puntos_extremos[1][1], f'Altura: {altura_cm:.2f} cm', fontsize=12, color='red', ha='right', va='bottom')
plt.text((puntos_extremos[0][0] + puntos_extremos[1][0]) / 2, puntos_extremos[0][1] - 10, f'Ancho: {ancho_cm:.2f} cm', fontsize=12, color='red', ha='center', va='top')
plt.text((puntos_extremos[0][0] + puntos_extremos[1][0]) / 2, (puntos_extremos[0][1] + puntos_extremos[1][1]) / 2, f'Área: {area_cm2:.2f} cm^2', fontsize=12, color='red', ha='center', va='center')
plt.axis('off')
plt.show()

'''


#######################################################################
############### CODIGO DE ALTURA BUENO ################################
#######################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

def obtener_mascara_segmentacion(imagen):
    # Convertir la imagen a espacio de color HSV
    imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Definir el rango de valores para el color rojo
    rango_rojo_bajo = np.array([0, 100, 100])
    rango_rojo_alto = np.array([10, 255, 255])

    # Crear la máscara de segmentación
    mascara_segmentacion = cv2.inRange(imagen_hsv, rango_rojo_bajo, rango_rojo_alto)

    return mascara_segmentacion

def encontrar_contorno_planta(mascara_segmentacion):
    # Encontrar los contornos
    contornos, _ = cv2.findContours(mascara_segmentacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (la planta segmentada)
    contorno_planta = max(contornos, key=cv2.contourArea)

    return contorno_planta

def obtener_puntos_extremos(contorno):
    # Obtener los puntos extremos del contorno
    puntos_extremos = tuple(contorno[contorno[:, :, 1].argmin()][0]), tuple(contorno[contorno[:, :, 1].argmax()][0])

    return puntos_extremos

# Cargar la imagen de ejemplo
imagen = cv2.imread(r"C:\Users\Usuario\Desktop\pythonProject\ImagenesSEGMENTADAS\segmentado_4_13_03.JPG")

# Obtener la máscara de segmentación
mascara_segmentacion = obtener_mascara_segmentacion(imagen)

# Encontrar el contorno de la planta segmentada
contorno_planta = encontrar_contorno_planta(mascara_segmentacion)

# Obtener los puntos extremos dentro del contorno
puntos_extremos = obtener_puntos_extremos(contorno_planta)

# Calcular la altura de la planta (en píxeles)
altura_px = abs(puntos_extremos[1][1] - puntos_extremos[0][1])

# Suponemos una relación aproximada entre píxeles y centímetros
# Esta relación puede necesitar ajustes según la escala real de la imagen
# Por ejemplo, si hay un objeto de longitud conocida en la imagen, se puede utilizar para una calibración más precisa.
pixeles_por_cm = 100 # Por ejemplo, suponemos que 10 píxeles representan aproximadamente 1 cm

# Convertir la altura de píxeles a centímetros
altura_cm = altura_px / pixeles_por_cm

# Mostrar la imagen con el rectángulo y la altura calculada
plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
plt.plot([puntos_extremos[0][0], puntos_extremos[1][0]], [puntos_extremos[0][1], puntos_extremos[1][1]], color='red')
plt.text(puntos_extremos[1][0], puntos_extremos[1][1], f'Altura: {altura_cm:.2f} cm', fontsize=12, color='red', ha='right', va='bottom')
plt.axis('off')
plt.show()

#######################################################################################################
#######################################################################################################






import cv2
import os

def calcular_altura_plantas(ruta_imagenes_segmentadas):
    # Obtener la lista de nombres de archivos de la carpeta
    archivos = os.listdir(ruta_imagenes_segmentadas)

    # Bucle sobre cada archivo de imagen en la carpeta
    for archivo in archivos:
        # Construir la ruta completa de la imagen
        ruta_imagen = os.path.join(ruta_imagenes_segmentadas, archivo)

        # Leer la imagen desde la ruta
        imagen = cv2.imread(ruta_imagen)

        # Convertir la imagen a escala de grises
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicar umbral para obtener la máscara de la planta segmentada
        _, mascara = cv2.threshold(imagen_gris, 240, 255, cv2.THRESH_BINARY)

        # Encontrar contornos en la máscara
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar el contorno más grande
        contorno_mas_grande = max(contornos, key=cv2.contourArea)

        # Calcular el rectángulo delimitador del contorno más grande
        x, y, w, h = cv2.boundingRect(contorno_mas_grande)

        # Dibujar el contorno más grande en la imagen original
        cv2.drawContours(imagen, [contorno_mas_grande], -1, (0, 255, 0), 2)

        # Redimensionar la imagen para que se ajuste a la pantalla
        alto, ancho, _ = imagen.shape
        factor_escala = 600 / alto  # Redimensionar a una altura de 600 píxeles
        imagen_redimensionada = cv2.resize(imagen, (int(ancho * factor_escala), int(alto * factor_escala)))

        # Mostrar la imagen con el contorno más grande
        cv2.imshow("Contorno más grande", imagen_redimensionada)
        cv2.waitKey(0)

        # Calcular la altura de la planta (altura del rectángulo delimitador)
        altura_planta = h

        print(f"Altura de la planta en {archivo}: {altura_planta} píxeles")

if __name__ == "__main__":
    ruta_imagenes_segmentadas = input("Por favor, ingresa la ruta donde tienes las imágenes segmentadas de tus plantas de rábano: ")
    calcular_altura_plantas(ruta_imagenes_segmentadas)







################################################################################################
####################### CODIGO PARA LA SEGMENTACIÓN EN TIEMPO REAL #############################
################################################################################################

# importar librerias
from ultralytics import YOLO
import cv2

#leer nuestro modelo
# codigo para tiempo real videocaptura
modelo= YOLO("best.pt")

cap = cv2.VideoCapture(1)

#bucle
while True:
    #leer fotogramas
    ret, frame = cap.read()

    resultados = modelo.predict(frame, imgsz = 640)

    anotaciones = resultados[0].plot()

    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
######################################################################################

#######################################################################################
############## CODIGO PARA INGRESAR LAS IMAGENES (SEGMENTACIÓN)  ######################
#######################################################################################
from ultralytics import YOLO
import cv2
import os

# Leer nuestro modelo
modelo = YOLO("best.pt")

# Ruta a la carpeta que contiene las imágenes
carpeta_imagenes = "imagenesSegmentación"

# Obtener la lista de nombres de archivos de la carpeta
archivos = os.listdir(carpeta_imagenes)

# Bucle sobre cada archivo de imagen en la carpeta
for archivo in archivos:
    # Construir la ruta completa de la imagen
    ruta_imagen = os.path.join(carpeta_imagenes, archivo)

    # Leer la imagen desde la ruta
    frame = cv2.imread(ruta_imagen)

    # Predecir utilizando el modelo
    resultados = modelo.predict(frame, imgsz=640, conf=0.8)

    # Obtener las anotaciones
    anotaciones = resultados[0].plot()

    # Ajustar tamaño de la ventana a las dimensiones de la imagen
    cv2.namedWindow("DETECCION Y SEGMENTACION", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DETECCION Y SEGMENTACION", frame.shape[1], frame.shape[0])

    # Mostrar las anotaciones
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Esperar una tecla para cerrar la ventana
    if cv2.waitKey(0) == 27:
        break

# Cerrar todas las ventanas
cv2.destroyAllWindows()

########################################################################
'''


