import cv2
face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
filename = "2.mp4"
cam = cv2.VideoCapture(filename)

cam.set(3, 512)
cam.set(4, 512)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.33, 5)
    j = "Face Not Detected"
    k = 0
    for (x, y, h, w) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 201, 150), 2)
        j = "Face Detected"

    cv2.imshow("Face", frame)
    cv2.imwrite("")
    print(j)
    if cv2.waitKey(1) == 32:
        break

cv2.destroyAllWindows()

cam.release()


