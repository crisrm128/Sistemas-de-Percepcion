import cv2
from cv2 import aruco
#import cv2.aruco as A
import numpy as np
import time
import glob

with np.load('calib_charuco_parameters.npz') as X:
    camera_matrix, dist_coeffs, rvec, tvec = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

aruco_dict = aruco.Dictionary_get( aruco.DICT_6X6_250 )

squareLength = 0.04   # Here, our measurement unit is centimetre.
markerLength = 0.02   # Here, our measurement unit is centimetre.
board = aruco.CharucoBoard_create(5, 7, squareLength, markerLength, aruco_dict)

arucoParams = aruco.DetectorParameters_create()

#cap = cv2.VideoCapture(0)

#if (cap.isOpened()== False): 
#  print("Error opening video  file")

#while(cap.isOpened()):

images = glob.glob('calibration_charuco/*.jpg')

# Loop through images glob'ed
for iname in images:

#while(True):
    #ret, frame = cap.read() # Capture frame-by-frame
    #if ret == True:
    frame = cv2.imread(iname)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
    aruco.refineDetectedMarkers(frame_gray, board, corners, ids, rejectedImgPoints)
    frame = aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 0, 255))
    
    print(ids)

    if ids[0] != None:
        print(len(ids))

    if ids is not None and len(ids) > 10: # if there is at least one marker detected
        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, frame_gray, board) #charucoretval is the number of squares detected (24 correct)
        
        if charucoretval is not None and charucoretval > 20:

            im_with_charuco_board = aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0,0,255))
            retval = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, dist_coeffs, rvec, tvec)  # posture estimation from a charuco board
            if retval == True:
                im_with_charuco_board = aruco.drawAxis(im_with_charuco_board, camera_matrix, dist_coeffs, rvec, tvec, 100)  # axis length 100 can be changed according to your requirement
            cv2.imshow("charucoboard", im_with_charuco_board)
    else:
        im_with_charuco_left = frame
        cv2.imshow("charucoboard", frame)

        
    cv2.waitKey(0)
    #if cv2.waitKey(2) & 0xFF == ord('q'):
    #    break


#cap.release()   # When everything done, release the capture
cv2.destroyAllWindows()