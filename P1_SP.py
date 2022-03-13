import numpy as np
import cv2 as cv
import glob
import pickle as pkl

#https://johnnn.tech/q/opencv-number-of-object-and-image-points-must-be-equal-expected-numberofobjectpoints-numberofimagepoints/
#Para probar con video en vez de con imagenes, el profe ha dicho que hacer un video que cada 60 frames haga una foto.

#Necesitamos que correpsondan los puntos 2D con los 3D. En 3D tenemos de cada imagen tomada un total de 9x6 puntos
#3D, mientras que en 2D necesitamos el mismo número de puntos, que son 9x6.


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('Callibration_frames3.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
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
    
calib_info = dict = {'mtx':mtx, 'dist':dist,'rvecs':rvecs,'tvecs':tvecs,'mean_error':mean_error}

print( "total error: {}".format(mean_error/len(objpoints)) )
print("Calibration",calib_info)

f = open("calib_parameteres.pkl", "wb")
pkl.dump(calib_info,f)

