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


