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
    anotaciones = resultados.pred[0]

    # Dibujar las cajas delimitadoras en la imagen
    for caja in anotaciones:
        coordenadas = caja[:4].int().cpu().numpy()
        x_min, y_min, x_max, y_max = coordenadas
        frame = cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    # Ajustar tamaño de la ventana a las dimensiones de la imagen
    cv2.namedWindow("DETECCION Y SEGMENTACION", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DETECCION Y SEGMENTACION", frame.shape[1], frame.shape[0])

    # Mostrar la imagen con las cajas delimitadoras
    cv2.imshow("DETECCION Y SEGMENTACION", frame)

    # Esperar una tecla para cerrar la ventana
    if cv2.waitKey(0) == 27:
        break

# Cerrar todas las ventanas
cv2.destroyAllWindows()







