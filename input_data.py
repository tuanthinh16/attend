import cv2
import os.path
import sqlite3
from PIL import Image
import dlib


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


def InsertOrUpdate(mssv, name, lop, email):
    conn = sqlite3.connect('D:\\diemdanh\\sv.db')
    query = "SELECT * FROM sinhvien where MSSV="+str(mssv)
    curr = conn.execute(query)
    isRecordExit = 0
    for row in curr:
        isRecordExit = 1
    if(isRecordExit == 0):
        query = "INSERT INTO sinhvien VALUES("+str(mssv) + \
            ",'"+str(name)+"','"+str(lop)+"','"+str(email)+"')"
    else:
        query = "UPDATE sinhvien SET Name = '" + \
            str(name)+"',Class ='"+str(lop)+"',Email ='" + \
            str(email)+"' WHERE MSSV ="+str(mssv)
    conn.execute(query)
    conn.commit()
    conn.close()


# input data
mssv = input("Nhap MSSV: ")
name = input("Nhap Ten SV: ")
lop = '18ct3'
email = 'hhhh'
InsertOrUpdate(mssv, name, lop, email)

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video = cv2.VideoCapture(0)
smlNum = 0
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_dec = dlib.get_frontal_face_detector()
    faces = face_dec(frame)
    left = 10
    right = 10
    top = 10
    bottom = 10
    # faces = face_cascade.detectMultiScale(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(img = frame, pt1 =(x1, y1), pt2 =(x2, y2), color=(0, 255, 0), thickness =3)
        face_feature = predictor(frame, face)
        for n in range(0, 68):
            x = face_feature.part(n).x
            y = face_feature.part(n).y
            # draw dot
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
        
        if not os.path.exists('Dataset'):
            os.makedirs('Dataset')
        smlNum += 1
        frame_crop  = gray[y1-top:y1+(y1-y2)+bottom, x1-left:x1+(x2-x1)+right]
        cv2.imwrite('Dataset/'+str(name)+'.'+str(mssv) +'.'+str(smlNum)+'.jpg', frame_crop)
    if ret:
        cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
