import re
import numpy
import os
import cv2
import base64
import requests
import cognitive_face as CF
from github import Github


# CONTROLS
#	SPACE BAR:	Take Picture
#	ESC:		Exit Program

def faceRecog():
    
    return

#Microsoft Azure Key, distributed from Azure account
KEY = ''
CF.Key.set(KEY)

BASE_URL = 'https://eastus.api.cognitive.microsoft.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)

#Github Access Token, distributed from github account
gT = ""

#Create Github main class gH
gH = Github(gT)

#Get Facial Recognition Repository
gR = gH.get_user().get_repo("FacialRecognition")

headers = {'Content-Type': 'application/octet-stream', 
			'Ocp-Apim-Subscription-Key': KEY}
url = 'https://api.projectoxford.ai/face/v1.0/detect'

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
CF.BaseUrl.set(BASE_URL)

img_counter = 0

if vc.isOpened(): #Get the first frame from webcam
    rval, frame = vc.read()
else:
    rval = False

print("-+-+-Controls: -+-+-")
print("-+-+-Exit: ESC Key-+-+-")
print("-+-+-Take Picture: Space Bar-+-+-")

while rval:
	cv2.imshow("preview", frame)
	rval, frame = vc.read()
	key = cv2.waitKey(20)
	if key == 27: #ESC - exit program
		break
	elif key == 32: #Space bar - Take Picture
		img_name = "facePic.png"
		cv2.imwrite(img_name, frame)
		print("Face Picture written!")
		data = open('facePic.png', 'rb')
        #Encode image as Base64
		data_read = data.read()
		data_64 = base64.encodestring(data_read)
        #Get face file SHA
		gCon = gR.get_contents('/facePic.png')
		gSha = gCon.sha
        #Update face file
		gR.update_file('/facePic.png','', data_64, gSha)
		#Detect faces
		face1 = CF.face.detect('https://raw.githubusercontent.com'
			'/SR860835/FacialRecognition/master/facePic.png')
		face2 = CF.face.detect('https://raw.githubusercontent.com'
			'/SR860835/FacialRecognition/master/facePicBase.png')
		#Get face ID's
		face1ID = face1['faceId']
		face2ID = face2['faceId']
		#Verify faces
		faceComp = CF.face.verify(face1ID, face2ID)
		#If match is found, print that they are a match
        if faceComp['isIdentical']:
			print('These faces are a match!')

cv2.destroyWindow("preview")
vc.release()
