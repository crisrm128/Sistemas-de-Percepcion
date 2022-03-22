import numpy
import cv2
from cv2 import aruco
import glob
import numpy as np
from cmath import sqrt
import math

# Parámetros del tablero ChAruco 
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 5 
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)

squareLength = 0.04   # Tamaño del lado de los cuadrados de ajedrez. (Importante para los puntos 3D grabados a fuego en el código)

# Crear constantes para pasarlas a OpenCV y los métodos de Aruco
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=0.04,
        markerLength=0.02,
        dictionary=ARUCO_DICT)

# Crear los vectores y variables que se usarán para guardar información como esquinas y los IDs de las imágenes procesadas
corners_all = [] # Esquinas detectadas en todas las imágenes procesadas
ids_all = [] # Aruco ids ccorrespondeintes a las esquinas detectadas 
image_size = None # Determinado en tiempo de ejecución

###################################################################################################

#Puntos obtenidos empíricamente para las esquinas de los cuadrados pertecencientes al tablero de ajedrez (puntos azules).

objp_corners = np.zeros((24, 3), np.float32)
objp_corners[0] = (squareLength, squareLength, 0)
objp_corners[1] = (2*squareLength, squareLength, 0)
objp_corners[2] = (3*squareLength, squareLength, 0)
objp_corners[3] = (4*squareLength, squareLength, 0)
objp_corners[4] = (squareLength, 2*squareLength, 0)
objp_corners[5] = (2*squareLength, 2*squareLength, 0)
objp_corners[6] = (3*squareLength, 2*squareLength, 0)
objp_corners[7] = (4*squareLength, 2*squareLength, 0)
objp_corners[8] = (squareLength, 3*squareLength, 0)
objp_corners[9] = (2*squareLength, 3*squareLength, 0)
objp_corners[10] = (3*squareLength, 3*squareLength, 0)
objp_corners[11] = (4*squareLength, 3*squareLength, 0)
objp_corners[12] = (squareLength, 4*squareLength, 0)
objp_corners[13] = (2*squareLength, 4*squareLength, 0)
objp_corners[14] = (3*squareLength, 4*squareLength, 0)
objp_corners[15] = (4*squareLength, 4*squareLength, 0)
objp_corners[16] = (squareLength, 5*squareLength, 0)
objp_corners[17] = (2*squareLength, 5*squareLength, 0)
objp_corners[18] = (3*squareLength, 5*squareLength, 0)
objp_corners[19] = (4*squareLength, 5*squareLength, 0)
objp_corners[20] = (squareLength, 5*squareLength, 0)
objp_corners[21] = (2*squareLength, 5*squareLength, 0)
objp_corners[22] = (3*squareLength, 5*squareLength, 0)
objp_corners[23] = (4*squareLength, 5*squareLength, 0)

objpoints = [] # Puntos 3D en el espacio del mundo real.

###################################################################################################


# Conjunto de imágenes para calibrar (del mismo tamaño al ser tomadas por la misma cámara)
images = glob.glob('cris_charuco_home_antiguas/*.jpg')

# Para todas las imágenes obtenidas
for iname in images:
    # Abrir la imagen
    img = cv2.imread(iname)
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Buscar los marcadores aruco en la imagen
    corners, ids, rejected = aruco.detectMarkers(
            image=gray,
            dictionary=ARUCO_DICT)

    # Resaltar los bordes de los marcadores aruco encontrados en la imagen
    img = aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    # Obtener las esquinas del charuco y sus ids a partir de los marcadores aruco detectados
    response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD)

    #print(charuco_corners) #Detecta los corners azules de los ids

    #print(response)
    # Si se detectó un tablero charuco, se van a recolectar los puntos de esquina de la imagen
    # Se requiere al menos haber detectado 20 cuadrados (varía en función del tablero)
    if response >23:
        #Añade las esquinas y sus ids a los vectores de calibración
        objpoints.append(objp_corners)
        corners_all.append(charuco_corners) #son los imgpoints
        ids_all.append(charuco_ids)

        # Dibuja el tablero charuco para mostrar que el tablero se ha detectado adecuadamente
        img = aruco.drawDetectedCornersCharuco(
                image=img,
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids)
       
        # Si el tamaño de la imagen es desconocido, guardarlo ahora
        if not image_size:
            image_size = gray.shape[::-1]

        cv2.imshow('Charuco board', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a charuco board in image: {}".format(iname))

cv2.destroyAllWindows()


# Para asegurarse de que al menos se encontró una imagen
if len(images) < 1:
    # Calibración fallida al no haber imágenes, avisar al usuario
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    # Exit por fallo
    exit()


# Asegurarse de que se pudo calibrar al menos un tablero charuco a partir de si se ha determinado el tamalo de la imagen
if not image_size:
    # Calibración fallida porque no se detectó ningún tablero charuco con los parámetros especificados
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit por fallo
    exit()


# Ahora que hemos visto todas las imágenes se calibra la cámara en función de los puntos detectados
calibration, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

#Una vez la cámara está calibrada, calculamos el error de reproyección.
mean_error = 0
for i in range(len(objpoints)):
    #Proyectamos los puntos 3D, rvecs y tvecs son 2 vectores de 3x1 tanto de rotación como de traslación, pero la rotación
    #está implementada en el formato Rodrigues.
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    #Norma L2: Distancia euclídea entre los puntos del patrón y los que hemos proyectado nosotros
    error = cv2.norm(corners_all[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    #print(corners_all[i])
    #print(imgpoints2)
    #print(error)
    mean_error += error
mean_error = mean_error/len(objpoints)

#Se normaliza el error al hacer un cambio de escala.
resol = (640, 480)

res = sqrt(math.pow(resol[0],2) + math.pow(resol[1],2))
normal = mean_error/res
mean_error = normal.real


# Imprimir matriz de parametros intrinsecos y coeficientes de distorsion
print(cameraMatrix)
print(distCoeffs)
print(calibration)

cv2.imshow('Vis', img)
cv2.waitKey(0)
    
#Me guardo todos los parámetros de la calibración.
calib_info = dict = {'mtx':cameraMatrix, 'dist':distCoeffs,'rvecs':rvecs,'tvecs':tvecs,'mean_error':mean_error}

np.savez('calib_charuco_parameters.npz', **calib_info)  # data is a dict here
    
# Imprimir por consola que la calibración se ha llevado a cabo correctamente
print( "total error: {}".format(mean_error) )
print("Calibration",calib_info)