from PIL import Image
import glob
import numpy as np
from HomeBotics import KnnClassification as KC
from HomeBotics import CnnClassification as CC
import cv2


class FR:
    
    #video_capture = cv2.VideoCapture(0)
    def __init__(self, cascPath, facesDataPath):
        self.cascPath = cascPath
        self.faceCascade = cv2.CascadeClassifier(self.cascPath)
        self.imageList=[] #image list stored in dataset
        self.labels=[] #the labels for images to classify images
    
        #paths for images
        self.faceDataPath = facesDataPath
        self.path1= facesDataPath + "/1/*.jpg"
        self.path2= facesDataPath + "/2/*.jpg"
        self.path3= facesDataPath + "/3/*.jpg"
        self.path4= facesDataPath + "/4/*.jpg"
    
    #convert image
    def convertImage(self, img, width=80, height=80):
        img=img.resize((width,height),Image.BILINEAR)
        img=img.convert("L")#convert to gray scale
        img_ndarray = np.asarray(img, dtype='float32') / 255
        face = np.ndarray.flatten(img_ndarray)
        face = face.reshape(-1,80*80)
        return face
        
    #read image to image list
    def readImages(self):
        #read images of face1
        for imageFile in glob.glob(self.path1):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(0)
        #read images of face2
        for imageFile in glob.glob(self.path2):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(1)
        #read images of face3
        for imageFile in glob.glob(self.path3):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(2)
        #read images of face4
        for imageFile in glob.glob(self.path4):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(3)
        return self.imageList,self.labels
    
    #count=0 #number of total recognised face
    #cnt1=0 #number of total recognised face1
    #cnt2=0 #number of total recognised face2
    #cnt3=0 #number of total recognised face3
    #cnt4=0 #number of total recognised face4
    def recogniseFaces(self, frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert to gray scale
        #find faces
        faces=self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(20,20)
        )
        cls = []
        for(x, y, w, h) in faces:
            #Draw rectangles around the faces
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            #Get the frame of face area
            faceFrame = frame[y:y+h, x:x+w]
            
            #write faceimage as jpg file
            cv2.imwrite(self.faceDataPath + "/face.jpg", faceFrame)
            
            #read face image
            faceImage=Image.open(self.faceDataPath + "/face.jpg")
            
            #convert face image
            img = self.convertImage(faceImage)
            
            #classify the face
            #cls.append(KC.knnClassify(img,np.array(self.imageList),self.labels,7))
            cls.append(CC.cnn_pre(img))
        return cls
        

