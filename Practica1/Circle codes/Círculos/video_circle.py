import numpy as np
import cv2 as cv
import glob

with np.load('calib_circle_parameters.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

#NO ES NECESARIA PORQUE SE UTILZA PARA LA FUNCION CORNERSUBPIX
# termination criteria -> tiene que ver con el proceso de minimización, en algún momento hay que cortar.
#criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points:
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

#Cambiando los valores cambiamos las longitudes de los ejes (ejemplo de eje central).
#Para proyectar los ejes
axis = np.float32([[2*72,0,0], [0,2*72,0], [0,0,2*72]]).reshape(-1,3)

def draw_axis(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    corners = np.int32(corners).reshape(-1,2)

    origin = corners[0]

    img = cv.line(img, origin, tuple(imgpts[0].ravel()), (255,0,0), 2) #Eje x
    img = cv.line(img, origin, tuple(imgpts[1].ravel()), (0,255,0), 2) #Eje y
    img = cv.line(img, origin, tuple(imgpts[2].ravel()), (0,0,255), 2) #Eje z

    return img


#Para proyectar el muñeco
stickman = np.float32([[2*72+10,2*72,54], [3*72-10,2*72,54], [2*72+10,2*72,108], [3*72-10,2*72,108],[2*72-10,2*72,54], [3*72+10,2*72,54], [180,2*72,108], [180,2*72,108+18]]).reshape(-1,3)

def draw_stickman(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    corners = np.int32(corners).reshape(-1,2)

    origin = tuple(corners[18].ravel())

    origin2 = tuple(corners[26].ravel())
    
    imgpts1 = tuple(imgpts[0].ravel())
    
    imgpts2 = tuple(imgpts[1].ravel())
    
    imgpts3 = tuple(imgpts[2].ravel())

    imgpts4 = tuple(imgpts[3].ravel())

    imgpts5 = tuple(imgpts[4].ravel())

    imgpts6 = tuple(imgpts[5].ravel())

    imgpts7 = tuple(imgpts[6].ravel())

    imgpts8 = tuple(imgpts[7].ravel())
    
    img = cv.line(img, origin, imgpts1, (255,0,0), 5) #Pierna izq
    img = cv.line(img, origin2, imgpts2, (255,0,0), 5) #Pierna der
    img = cv.line(img, imgpts1, imgpts2, (255,0,0), 5) #Entrepierna
    img = cv.line(img, imgpts1, imgpts3, (255,0,0), 5) #Lado izq
    img = cv.line(img, imgpts2, imgpts4, (255,0,0), 5) #Lado der
    img = cv.line(img, imgpts3, imgpts4, (255,0,0), 5) #Linea sup
    img = cv.line(img, imgpts3, imgpts5, (255,0,0), 5) #Linea sup
    img = cv.line(img, imgpts4, imgpts6, (255,0,0), 5) #Linea sup
    img = cv.line(img, imgpts7, imgpts8, (255,0,0), 5) #Linea sup

    # Using cv2.circle() method
    tam = imgpts[4].ravel() - imgpts[3].ravel()
    tam = abs(tam[0])
    tam = int(tam/3)

    t =  tuple(map(lambda i, j: i - j, tuple(imgpts[7].ravel()), (0, tam/3 + tam/4)))
    t1 = t[0]
    t2 = t[1]
    t = (int(t1), int(t2))

    #print(t)
    img = cv.circle(img, t, tam, (255, 0, 0), 5)

    return img

cap = cv.VideoCapture(0)


if (cap.isOpened()== False): 
  print("Error opening video  file")

while(cap.isOpened()):

    ret, img = cap.read()
    if ret == False:
        break

#for fname in glob.glob('circle_calibration/*.jpg'):
    #img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findCirclesGrid(gray, (4,11), None, flags = cv.CALIB_CB_ASYMMETRIC_GRID)   # Find the circle grid
    if ret == True:
        corners2 = corners
        #corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Otener vectores de rotación y traslación.
        #ret,rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist) #Para usar RANSAC pero no afecta
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # Proyectar los puntos 3D en la imagen del plano (el primer parámetro se debe modificar en función de lo que se quiera dibujar)
        imgpts, _ = cv.projectPoints(stickman, rvecs, tvecs, mtx, dist)     
        #img = draw_axis(img,corners2,imgpts) #Para proyectar los ejes
        img = draw_stickman(img,corners2,imgpts) #Para proyectar el muñeco
        
        #cv.imshow('img',img)
        #k = cv.waitKey(0) & 0xFF
        #if k == ord('s'):
            #cv.imwrite(fname[:6]+'.png', img)
    
    cv.imshow('Vis', img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()