import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
#Ingreso de la carpeta con las imagenes a procesar
image_folder = '/content/prub'
images = [cv2.imread(os.path.join(image_folder, filename)) for filename in os.listdir(image_folder)]

#Kernels que se van aplicar
kernel_1 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])#kernel a 45 grados

kernel_2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])#kernel a 135 grados

kernel_3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])# kernel a 90 grados

kernel_4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])#kernel a cero grados

kernel_5 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])#kernel a 180 grados

kernel_6 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])#kernel a 270 grados

#ciclo "for" principal para el proceso de filtrado
for filename in os.listdir(image_folder):
    # Construir la ruta de la imagen completa
    image_path = os.path.join(image_folder, filename)
    # leer la imagen en BGR format
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #print(image_rgb)
    #filtro bilateral
    bilateral_filtered_image1 = cv2.bilateralFilter(image_bgr, d=15, sigmaColor=75, sigmaSpace=75)
    bilateral_filtered_image = cv2.bilateralFilter(image_rgb, d=15, sigmaColor=75, sigmaSpace=75)
    # Convertir la imagen a escala de grises
    image_gray = cv2.cvtColor(bilateral_filtered_image1, cv2.COLOR_BGR2GRAY)
 # Aplicación de los kernels
    image_convoluted1 = cv2.filter2D(image_gray, -1, kernel_1)
    image_convoluted2 = cv2.filter2D(image_gray, -1, kernel_2)
    image_convoluted3 = cv2.filter2D(image_gray, -1, kernel_3)
    image_convoluted4 = cv2.filter2D(image_gray, -1, kernel_4)
    image_convoluted5 = cv2.filter2D(image_gray, -1, kernel_5)
    image_convoluted6 = cv2.filter2D(image_gray, -1, kernel_6)
    combined_image = cv2.addWeighted(image_convoluted1, 1, image_convoluted2, 1, 0)
    combined_image = cv2.addWeighted(combined_image, 1, image_convoluted3, 1, 0)
    combined_image = cv2.addWeighted(combined_image, 1, image_convoluted4, 1, 0)
    combined_image = cv2.addWeighted(combined_image, 1, image_convoluted5, 1, 0)
    combined_image5 = cv2.addWeighted(combined_image, 1, image_convoluted6, 1, 0)
    #sobel_threshold_image = cv2.adaptiveThreshold(combined_image5, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #Aplicacion de threshole
    threshold_value = 35
    sobel_threshold_image = cv2.threshold(combined_image5, threshold_value, 255, cv2.THRESH_BINARY)[1]
    imagefull = cv2.cvtColor(sobel_threshold_image, cv2.COLOR_GRAY2BGR)
    merged_image = cv2.addWeighted(bilateral_filtered_image, 1, imagefull, 0.5, 0.0)
    #merged_image22 = cv2.cvtColor(merged_image, cv2.COLOR_BGR2RGB)
    plt.imshow(merged_image, cmap="gray")
    plt.axis("off")
    plt.show()