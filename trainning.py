import cv2
import os.path
import numpy as np
from PIL import Image
import easygui

facecascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
recongniger = cv2.face.LBPHFaceRecognizer_create()
path = 'DataSet'


def getImageWithmssv(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImg, 'uint8')
        print(faceNP)
        Id = int(imagePath.split('\\')[1].split('.')[1])
        facez = facecascade.detectMultiScale(faceNP, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in facez: 
            faces.append(faceNP[y:y+h, x:x+w])
            IDs.append(Id)
        cv2.imshow('trainning', faceNP)
        cv2.waitKey(10)
    return faces, IDs

faces, IDs = getImageWithmssv(path)
recongniger.train(faces, np.array(IDs))
easygui.msgbox("Trainning Thành Công", title="Result")
if not os.path.exists('recongniger'):
    os.makedirs('recongniger')
recongniger.write('recongniger\\trainningData.yml')

cv2.destroyAllWindows()
