
import numpy as np
import cv2 as cv
import glob
import math

with np.load('calib_parameteres.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

axis = np.float32([[4,2,-2], [6,2,-2], [4,2,-4], [6,2,-4], [4-1.24233,2,-4-1.8592], [7,2,-2], [5,2,-4], [5,2,-4.7]]).reshape(-1,3)

def draw(img, corners, imgpts):
    corner1 = tuple(corners[21].ravel())
    corner2 = tuple(corners[25].ravel())

    img = cv.line(img, corner1, tuple(imgpts[0].ravel()), (0,0,255), 5)                     #Pierna izq
    img = cv.line(img, corner2, tuple(imgpts[1].ravel()), (0,0,255), 5)                     #Pierna der
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)    #Entrepierna
    img = cv.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 5)    #Cuerpo izq
    img = cv.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)    #Cuerpo der
    img = cv.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)    #Entrehombros
    img = cv.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)    #Brazo der
    img = cv.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[5].ravel()), (0,0,255), 5)    #Brazo izq
    img = cv.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), (0,0,255), 5)    #Cuello
    # Using cv2.circle() method
    tam = imgpts[3].ravel() - imgpts[2].ravel()
    tam = abs(tam[0])
    tam = int(tam/3)

    t =  tuple(map(lambda i, j: i - j, tuple(imgpts[7].ravel()), (0, tam/3 + tam/4)))
    t1 = t[0]
    t2 = t[1]
    t = (int(t1), int(t2))

    #print(t)
    img = cv.circle(img, t, tam, (0, 0, 255), 5)

    return img

cap = cv.VideoCapture("http://192.168.31.177:8080/video")

if (cap.isOpened()== False): 
  print("Error opening video  file")


while(cap.isOpened()):

    ret, img = cap.read()
    if ret == False:
        break

    #img = cv.imread(fname)
    #img = cv.resize(img,tuple(np.array([1280,720])))
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6),None)
    if ret == True:
    
        #print(axis_change)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.

        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)

    
    cv.imshow('img',img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()