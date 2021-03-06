from cmath import sqrt
from email import header
import math
from cv2 import mean
import numpy as np
import cv2 as cv
import glob
import pickle as pkl
import csv


#https://johnnn.tech/q/opencv-number-of-object-and-image-points-must-be-equal-expected-numberofobjectpoints-numberofimagepoints/
#Para probar con video en vez de con imagenes, el profe ha dicho que hacer un video que cada 60 frames haga una foto.

#Necesitamos que correpsondan los puntos 2D con los 3D. En 3D tenemos de cada imagen tomada un total de 9x6 puntos
#3D, mientras que en 2D necesitamos el mismo número de puntos, que son 9x6.

header = ['Normalized_error', 'Resolution']
f = open('csv_file.csv', 'w')
writer = csv.writer(f)
writer.writerow(header)
f.close()


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('cris_chessboard_class/*.jpg')

for fname in images:
    img = cv.imread(fname)
    img = cv.resize(img,tuple(np.array([1280,720])))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None) #El None es para poner flags de operación, que no se usan normalmente

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # Good calibration is all about precision. 
        # To get good results it is important to obtain the location of corners with sub-pixel level of accuracy. 
            
        #OpenCV’s function cornerSubPix takes in the original image, and the location of corners, 
        # and looks for the best corner location inside a small neighborhood of the original location. 
        # The algorithm is iterative in nature and therefore we need to specify the termination criteria 
        # ( e.g. number of iterations and/or the accuracy )
            
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('Vis', img)
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

mean_error = mean_error/len(objpoints)

res = sqrt(math.pow(1280,2) + math.pow(720,2))
mean_error = mean_error/res
mean_error = mean_error.real

mean_error = "{:.8f}".format(float(mean_error))

calib_info = dict = {'mtx':mtx, 'dist':dist,'rvecs':rvecs,'tvecs':tvecs,'mean_error':mean_error}

print( "total error: {}".format(mean_error))
print("Calibration",calib_info)

np.savez('calib_parameteres.npz', **calib_info) 

