from cmath import sqrt
from email import header
import numpy as np
import cv2 as cv
import glob 
import math

# termination criteria -> tiene que ver con el proceso de minimización, en algún momento hay que cortar.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

###################################################################################################

# Preparando los objpoints, teniendo en cuenta que la distancia entre los centros de los círculos es 72 mm (en realidad,
# este número podría modificarse por cualquier otro, pero no va a afectar). De todas formas, el tamaño real del círculo 
# es innecesario mientras se están calculando los parámetros de calibración.
objp = np.zeros((44, 3), np.float32)
objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 72 , 0)
objp[2]  = (0  , 144, 0)
objp[3]  = (0  , 216, 0)
objp[4]  = (36 , 36 , 0)
objp[5]  = (36 , 108, 0)
objp[6]  = (36 , 180, 0)
objp[7]  = (36 , 252, 0)
objp[8]  = (72 , 0  , 0)
objp[9]  = (72 , 72 , 0)
objp[10] = (72 , 144, 0)
objp[11] = (72 , 216, 0)
objp[12] = (108, 36,  0)
objp[13] = (108, 108, 0)
objp[14] = (108, 180, 0)
objp[15] = (108, 252, 0)
objp[16] = (144, 0  , 0)
objp[17] = (144, 72 , 0)
objp[18] = (144, 144, 0)
objp[19] = (144, 216, 0)
objp[20] = (180, 36 , 0)
objp[21] = (180, 108, 0)
objp[22] = (180, 180, 0)
objp[23] = (180, 252, 0)
objp[24] = (216, 0  , 0)
objp[25] = (216, 72 , 0)
objp[26] = (216, 144, 0)
objp[27] = (216, 216, 0)
objp[28] = (252, 36 , 0)
objp[29] = (252, 108, 0)
objp[30] = (252, 180, 0)
objp[31] = (252, 252, 0)
objp[32] = (288, 0  , 0)
objp[33] = (288, 72 , 0)
objp[34] = (288, 144, 0)
objp[35] = (288, 216, 0)
objp[36] = (324, 36 , 0)
objp[37] = (324, 108, 0)
objp[38] = (324, 180, 0)
objp[39] = (324, 252, 0)
objp[40] = (360, 0  , 0)
objp[41] = (360, 72 , 0)
objp[42] = (360, 144, 0)
objp[43] = (360, 216, 0)
###################################################################################################
# Arrays para almacenar los object points y los image points de todas las imágenes.
objpoints = [] # Puntos 3D en el espacio del mundo real.
imgpoints = [] # Puntos 2D en el plano de la imagen.
###################################################################################################
images = glob.glob('circulos/*.jpg')
#TIP: Todas estas líneas comentadas sirven para, en vez de calibrar utilizando imágenes estáticas de la
#carpeta (ver línea 72), se abra una pestaña de vídeo y se vea en directo la calibración y los puntos detectados.
#cap = cv.VideoCapture(0)
#if (cap.isOpened()== False): 
#print("Error opening video  file")
#while(cap.isOpened()):
        #ret, img = cap.read()
        #if ret == False:
        #        break
for fname in images:
        img = cv.imread(fname)
        img = cv.resize(img,tuple(np.array([1280,720])))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findCirclesGrid(gray, (4,11), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid
        # If found, add object points, image points (after refining them)
        if ret == True:
                objpoints.append(objp)
                #El error estaba aquí, no hace falta calcular la mejor esquina calculada en una vecindad cercana.
                corners2 = corners
                #corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, (4,11), corners2, ret)
                cv.imshow('Vis', img)
                cv.waitKey(500)
        #cv.imshow('Vis', img)
        #if cv.waitKey(25) & 0xFF == ord('q'):
                #break
        
#cap.release()
cv.destroyAllWindows()
###################################################################################################
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#Una vez la cámara está calibrada, calculamos el error de reproyección.
mean_error = 0

for i in range(len(objpoints)):
#Proyectamos los puntos 3D, rvecs y tvecs son 2 vectores de 3x1 tanto de rotación como de traslación, pero la rotación
#está implementada en el formato Rodrigues.
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#Norma L2: Distancia euclídea entre los puntos del patrón y los que hemos proyectado nosotros
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
mean_error = mean_error/len(objpoints)

#Me guardo todos los parámetros de la calibración.
calib_info = dict = {'mtx':mtx, 'dist':dist,'rvecs':rvecs,'tvecs':tvecs,'mean_error':mean_error}

np.savez('calib_circle_parameters.npz', **calib_info)  # data is a dict here

print( "total error: {}".format(mean_error) )
print("Calibration",calib_info)