import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()     #Creating a recognizer for recognize captured face
path = "dataSet"

recognizer_path = 'recognizer'
if not os.path.isdir(recognizer_path):
    os.mkdir(recognizer_path)

def getImg(path):
    imgpath = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []    #defining an empty list for storing the faces
    ids = []       # defining an empty list for storing the id's
    for imagepath in imgpath:
        faceimg = Image.open(imagepath) #Opening captured images
        facenp = np.array(faceimg, "uint8")    #convert to numpy array bcz cv2 only works with numpy array

        id = int(os.path.split(imagepath)[-1].split(".")[1])   #spliting the id number
        print(id)
        faces.append(facenp)   #storing the numpy array faces to the empty list
        print(id)
        ids.append(id)    #storing the id's to the empty list
        #cv2.imshow("Training",facenp)
        cv2.waitKey(1)
    return faces, ids   #this will return all faces and ids


faces, ids = getImg(path)
recognizer.train(faces,np.array(ids))       #training the recognizer
recognizer.write(recognizer_path + "/trainingdata.yml")        #creating a training file
getImg(path)

