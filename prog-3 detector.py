import cv2

face1=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


names = ['John', 'Jack']  #Change the names according to your id's
cam=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingdata.yml")

#Text Font parameters
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
fontscale=1
fontcolor=255,50,10
id=0

while True:
    ret,frame=cam.read()  #Capturing frames
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face1.detectMultiScale(gray,1.33,6) #Detecting faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        id,conf = rec.predict(gray[y:y+h,x:x+w])  #Predicting faces
        print (id, conf)
        if conf<65:
            id = names[id-1]
        else:
            id="Unknown"
        cv2.putText(frame, str(id), (x+h,y), font, fontscale, fontcolor)
    cv2.imshow("Face",frame)

    if cv2.waitKey(1)== ord('q'):
        break
cv2.destroyAllWindows()
cam.release()


