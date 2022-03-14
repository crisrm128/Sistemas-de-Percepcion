import cv2
from cv2 import aruco
import numpy as np
import glob

with np.load('calib_charuco_parameters.npz') as X:
    camera_matrix, dist_coeffs, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_250 )

squareLength = 0.04   # Tamaño del lado de los cuadrados de ajedrez. (LA RELACIÓN PARA DIBUJAR ES A PARTIR DE ESTE PARÁMETRO)
markerLength = 0.02   # Tamaño del lado de los cuadrados de arucos.
board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict) #Creación del tablero

arucoParams = aruco.DetectorParameters_create() #Se crean los parámetros del tablero, obtenidos dela web de OpenCV


#Los puntos se han obtenido de forma empírica de acuerdo a la relación en el eje de coordenadas (origen: esquina superior izquierda)
points = np.float32([[2*squareLength,2.5*squareLength,0],[2*squareLength,3*squareLength,squareLength],
                    [2*squareLength,4.5*squareLength,0], [2*squareLength,4*squareLength,squareLength], 
                    [2*squareLength,3*squareLength,2*squareLength], [2*squareLength,4*squareLength,2*squareLength], 
                    [2*squareLength,2.5*squareLength,squareLength], [2*squareLength,4.5*squareLength,squareLength],
                    [2*squareLength,3.5*squareLength,2*squareLength],[2*squareLength,3.5*squareLength,2.25*squareLength]]).reshape(-1,3)


def draw(img, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)
    
    #Conforme va apareciendo la oclusión, se va cambiando el punto 4 porque no se corresponde con el id = 4, se corresponde con el cuarto valor 
    #obtenido que se haya proyectado, por eso va cambiando de sitio en función del id que desaparece.

    #Si se hace todo con imgpts en vez de con corners ya no sucede ese problema.

    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 5)     #Pierna izq
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)     #Pierna der
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)     #Entrepierna
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 5)     #Cuerpo izq
    img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[5].ravel()), (0,0,255), 5)     #Cuerpo der
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0,0,255), 5)     #Entrehombros
    img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 5)     #Brazo izq
    img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[7].ravel()), (0,0,255), 5)     #Brazo der
    img = cv2.line(img, tuple(imgpts[8].ravel()), tuple(imgpts[9].ravel()), (0,0,255), 5)     #Cuello
    
    # Using cv2.circle() method 
    tam = imgpts[5].ravel() - imgpts[4].ravel()
    tam = abs(tam[0])
    tam = int(tam/3)

    t =  tuple(map(lambda i, j: i - j, tuple(imgpts[9].ravel()), (0, tam/3 + tam/4)))
    t1 = t[0]
    t2 = t[1]
    t = (int(t1), int(t2))

    #HEAD
    img = cv2.circle(img, t, tam, (0, 0, 255), 5)

    return img

cap = cv2.VideoCapture(0)

if (cap.isOpened()== False): 
  print("Error opening video  file")


#images = glob.glob('cris_charuco_home/*.jpg')

# Bucle a través de las imágenes
#for iname in images:

while(True):
    ret, frame = cap.read() # Capturar frame por frame
    if ret == True:
        #frame = cv2.imread(iname)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requiere una imagen en escala de grises

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)  # Primero detectar los marcadores
        aruco.refineDetectedMarkers(frame_gray, board, corners, ids, rejectedImgPoints)
        frame = aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 0, 255))
        
        im_with_charuco_board = frame.copy()
        if np.all(ids!=None): # Si se han detectado ids en el tablero
            
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, frame_gray, board) #charucoretval es el numero de esquinas detectadas en los cuadrados del tablero de ajedrez (24 correcto)

            if charucoretval is not None and charucoretval > 1: #Numero de esquinas chessboard detectadas (id=0 ... id=23), numero de esquinas que como minimo debe detectar (como minimo necesita las de la cruz del sist. de coord.)

                im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (255,0,0))
                retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, rvecs, tvecs)  # estimacion de pose para un tablero charuco
                if retval == True:
                    #Se obtienen los puntos proyectados del tablero:
                    imgpts,_ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
                    #Para dibujar el monigote:
                    im_with_charuco_board = draw(im_with_charuco_board, imgpts)
                    #Para dibujar únicamente los ejes de coordenadas:
                    #im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # El último parámetro es la longitud de los ejes

        cv2.imshow("charucoboard", im_with_charuco_board)   
        #cv2.waitKey(0)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break


#cap.release()   # Cuando se ha terminado, terminar la captura
cv2.destroyAllWindows()