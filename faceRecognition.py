from PIL import Image
import glob
import numpy as np
from HomeBotics import KnnClassification as KC
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
    def convertImage(self, img, width=30, height=30):
        img=img.resize((width,height),Image.BILINEAR)
        img=img.convert("L")#convert to gray scale
        img=np.reshape(img, (1,width*height))#convert to 1x(width*height) matrix
        return img[0]
        
    #read image to image list
    def readImages(self):
        #read images of face1
        for imageFile in glob.glob(self.path1):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(1)
        #read images of face2
        for imageFile in glob.glob(self.path2):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(2)
        #read images of face3
        for imageFile in glob.glob(self.path3):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(3)
        #read images of face4
        for imageFile in glob.glob(self.path4):
            img=Image.open(imageFile)
            self.imageList.append(self.convertImage(img))
            self.labels.append(4)
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
            cls.append(KC.knnClassify_cosSimilarity(img,np.array(self.imageList),self.labels,7))
        return cls
        
    #while True:
        #ret, frame=video_capture.read()
        #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert to gray scale
        ##find faces
        #faces=faceCascade.detectMultiScale(
            #gray,
            #scaleFactor=1.4,
            #minNeighbors=5,
            #minSize=(30,30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        #)
        
        #for(x, y, w, h) in faces:
            ##Draw rectangles around the faces
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            ##Get the frame of face area
            #faceFrame = frame[y:y+h, x:x+w]
            
            ##write faceimage as jpg file
            #cv2.imwrite("face.jpg", faceFrame)
            
            ##read face image
            #faceImage=Image.open("face.jpg")
            
            ##convert face image
            #img = convertImage(faceImage)
            
            ##classify the face
            #cls = KC.knnClassify(img,np.array(imageList),labels,7)
            
            ##increase count
            #count+=1
            #if cls==1:
                #print "Found Juntao!"
                #cnt1+=1;
            #elif cls==2:
                #print "Found Wentao!"
                #cnt2+=1;
            #elif cls==3:
                #print "Found Xiaotong"
                #cnt3+=1;
            #elif cls==4:
                #print "Found Yi"
                #cnt4+=1;
            
            #if count > 20:
                ##The possibilities for the face to belong to a person
                #pos = {}
                #pos["Juntao"] = (cnt1+0.0)/count
                #pos["Wentao"] = (cnt2+0.0)/count
                #pos["Xiaotong"] = (cnt3+0.0)/count
                #pos["Yi"] = (cnt4+0.0)/count
                #maxName = ''#name with max possibility
                #maxPos = 0.0
                #print pos
                
                #for (k,v) in pos.items():
                    #if v > maxPos:
                        #maxName = k
                        #maxPos = v
                #print 'Hello! ' + maxName + '!'
                    
        #cv2.imshow('Vedio', frame)
        #if cv2.waitKey(1)&0xFF == ord('q'):
            #break
    
    #video_capture.release()
    #cv2.destroyAllWindows()
