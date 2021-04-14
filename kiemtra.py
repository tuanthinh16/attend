import cv2
import os
import numpy as np
from PIL import Image
import dlib
import sqlite3
import datetime
import easygui
import time


hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1(
    'E:\\diemdanh\\mmod_human_face_detector.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
now = datetime.datetime.now()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('E:\\diemdanh\\recongniger\\trainningData.yml')


def getProfile(mssv):
    conn = sqlite3.connect('E:\\diemdanh\\sv.db')
    query = "SELECT * FROM sinhvien WHERE MSSV="+str(mssv)
    curror = conn.execute(query)
    profile = None
    for row in curror:
        profile = row
    conn.close()
    # print(str(profile[1]))
    return profile


# left = 10
# right = 10
# top = 10
# bottom = 10
index = None
cap = cv2.VideoCapture(0)
fontFace = cv2.FONT_ITALIC
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # faces_cnn = cnn_face_detector(frame, 1)
    # for face in faces_cnn:
    #     x = face.rect.left()
    #     y = face.rect.top()
    #     w = face.rect.right() - x
    #     h = face.rect.bottom() - y
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        # frame = frame[y-top:y+h+bottom, x-left:x+w+right]
        roi_gray = gray[y:y+h, x:x+w]
        MSSV, confidence = recognizer.predict(roi_gray)
        check = False
        
        # face_landmarks_list = face_recognition.face_landmarks(frame)
        
        # dets  = detector(frame)
        # faces = dlib.full_object_detections()
        # for detection in dets:
        #     faces.append(predictor(frame, detection))
        # if index == 'OK':
        #     continue
        if confidence <= 45:
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(
                frame,
                str(confidence),
                (x+5, y+h-5),
                fontFace,
                1,
                (255, 255, 0),
                1
            )
            profile = getProfile(MSSV)
            if(profile != None):
                check = True
                cv2.putText(
                    frame, ""+str(profile[1])+","+str(profile[2]), (x+10, y+h+30), fontFace, 1, (0, 255, 0), 2)
            conn = sqlite3.connect('E:\\diemdanh\\sv.db')
            query = "SELECT * FROM diemdanh WHERe MSSV="+str(MSSV)
            curr = conn.execute(query)
            isRecordExit = 0
            for row in curr:
                isRecordExit = 1
            if(isRecordExit == 0):
                query = "INSERT INTO diemdanh VALUES("+str(MSSV) + \
                    ",'"+str(now)+"','" + \
                    str(check) + "')"
                cv2.putText(frame, "Done !", (x, y+h-60),
                            fontFace, 1, (241, 175, 0), 3)
            else:

                #    index = easygui.msgbox("Đã Được Điểm Danh Vào Lúc :"+str(now), title="Result")
                cv2.putText(frame, "Attendanced", (x, y+h+60),
                            fontFace, 1, (72, 150, 32), 2)

            conn.execute(query)
            conn.commit()
            conn.close()
            # index = easygui.msgbox("Thành Công", title="Result")
            # print(str(now)+"----"+str(check) +
            #           "----"+str(MSSV)+"----"+str(query))
        else:
            confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(frame, "Unknow", (x+10, y+h+30),
                        fontFace, 1, (0, 0, 255), 2)

        #  start = time.time()

        #  end = time.time()
        # print("CNN Execution time: " + str(end-start))

        # Vẽ một đường bao đỏ xung quanh các khuôn mặt được xác định bởi CNN
    frame = cv2.resize(frame, (900, 680))
    cv2.imshow('image', frame)
    if cv2.waitKey(1) == ord('q'):
        break
if now.hour > 12:
    easygui.msgbox("Bạn Đã Đi Học Muộn ! Vui Lòng Đến Sớm Hơn", title="Result")
cap.release()
cv2.destroyAllWindows()
