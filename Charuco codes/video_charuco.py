import cv2
from cv2 import aruco
#import cv2.aruco as A
import numpy as np
import time
import glob

with np.load('calib_charuco_parameters.npz') as X:
    camera_matrix, dist_coeffs, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_250 )

squareLength = 0.04   # Here, our measurement unit is centimetre. (LA RELACIÓN PARA DIBUJAR ES A PARTIR DE ESTE PARÁMETRO)
markerLength = 0.02   # Here, our measurement unit is centimetre.
board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict)

arucoParams = aruco.DetectorParameters_create()


#axis = np.float32([[0.01,-0.01,0],[0.01,-0.01,0],[-0.01,-0.01,0],[-0.01,0.01,0], [0.01,0.01,0],[-0.01,-0.01,0],[-0.01,0.01,0],[0.01,0.01,0]]) 
axis = np.float32([[2*squareLength,2.5*squareLength,0],[2*squareLength,3*squareLength,squareLength],
                    [2*squareLength,4.5*squareLength,0], [2*squareLength,4*squareLength,squareLength], 
                    [2*squareLength,3*squareLength,2*squareLength], [2*squareLength,4*squareLength,2*squareLength], 
                    [2*squareLength,2.5*squareLength,squareLength], [2*squareLength,4.5*squareLength,squareLength],
                    [2*squareLength,3.5*squareLength,2*squareLength],[2*squareLength,3.5*squareLength,2.25*squareLength]]).reshape(-1,3)

#def draw(img, esquinas, imgpts):
#    imgpts = np.int32(imgpts).reshape(-1,2)

 #   for i,j in zip(range(4),range(4,8)):
#        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255,255,0),3)
   
#    return img

def draw(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)
    #corners = np.int32(imgpts).reshape(-1,2)
    
    #Conforme va apareciendo la oclusión, se va cambiando el punto 4 porque no se corresponde con el id = 4, se corresponde con el cuarto valor 
    #obtenido que se haya proyectado, por eso va cambiando de sitio en función del id que desaparece.

    #Si se hace todo con imgpts en vez de con corners ya no sucede ese problema.

    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,255,0), 5)     #Pierna izq
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,255,0), 5)     #Pierna der
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,255,0), 5)     #Entrepierna
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[4].ravel()), (0,255,0), 5)     #Cuerpo izq
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[5].ravel()), (0,255,0), 5)     #Cuerpo der
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0,255,0), 5)     #Entrehombros
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[6].ravel()), (0,255,0), 5)    #Brazo izq
    img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[7].ravel()), (0,255,0), 5)    #Brazo der
    img = cv2.line(img, tuple(imgpts[8].ravel()), tuple(imgpts[9].ravel()), (0,255,0), 5)    #Cuello
    # Using cv2.circle() method
    tam = imgpts[5].ravel() - imgpts[4].ravel()
    tam = abs(tam[0])
    tam = int(tam/3)

    t =  tuple(map(lambda i, j: i - j, tuple(imgpts[9].ravel()), (0, tam/3 + tam/4)))
    t1 = t[0]
    t2 = t[1]
    t = (int(t1), int(t2))

    #HEAD
    img = cv2.circle(img, t, tam, (0,255,0), 5)

    return img

# Arrays para almacenar puntos de objeto y puntos de imagen de todas las imágenes.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://192.168.31.177:8080/video")

if (cap.isOpened()== False): 
  print("Error opening video  file")

#time.sleep(3)

#while(cap.isOpened()):

#images = glob.glob('charuco_frames_2/*.jpg')

# Loop through images glob'ed
#for iname in images:

while(True):
    ret, frame = cap.read() # Capture frame-by-frame
    if ret == True:
        #frame = cv2.imread(iname)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
        aruco.refineDetectedMarkers(frame_gray, board, corners, ids, rejectedImgPoints)
        frame = aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 0, 255))
        
        #print(len(ids))

        #if ids[0] != None: #Nmero de arucos en el patrón
        #    print(len(ids))

        im_with_charuco_board = frame.copy()
        #if ids is not None and len(ids) > 10: # if there is at least one marker detected
        if np.all(ids!=None):
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, frame_gray, board) #charucoretval is the number of squares detected (24 correct)

            if charucoretval is not None and charucoretval > 1: #Numero de esquinas chessboard detectadas (id=0 ... id=23), numero de esquinas que como minimo debe detectar (como minimo necesita las de la cruz del sist. de coord.)

                im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (255,0,0))
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, rvecs, tvecs)  # posture estimation from a charuco board
                if retval == True:
                    imgpts,_ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                    im_with_charuco_board = draw(im_with_charuco_board, charucoCorners, imgpts)
                    #im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # axis length 100 can be changed according to your requirement
        #else:
        #    im_with_charuco_left = frame
        #    cv2.imshow("charucoboard left", im_with_charuco_left)

        cv2.imshow("charucoboard", im_with_charuco_board)   
        #cv2.waitKey(0)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break


#cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()