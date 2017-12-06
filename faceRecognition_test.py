import cv2
import sys

sys.path.append('face_recognition')

from HomeBotics.faceRecognition import FR

video_capture = cv2.VideoCapture(0)
cascPath = "haarcascades/haarcascade_frontalface_alt.xml"
facesDataPath = "facesData"
faceRec = FR(cascPath, facesDataPath)

#read images from faces datasets
faceRec.readImages()
count=0 #number of total recognised face
cnt1=0 #number of total recognised face1
cnt2=0 #number of total recognised face2
cnt3=0 #number of total recognised face3
cnt4=0 #number of total recognised face4
cnt5=0
while True:
    ret, frame=video_capture.read()
    cls = faceRec.recogniseFaces(frame)
    if len(cls) > 0:
        #increase count
        count+=1
        if cls[0]==0:
            print ("Found Juntao!")
            cnt1+=1;
        elif cls[0]==1:
            print ("Found Wentao!")
            cnt2+=1;
        elif cls[0]==2:
            print ("Found Xiaotong")
            cnt3+=1;
        elif cls[0]==3:
            print ("Found Yi")
            cnt4+=1;   
        else:
            cnt5+=1;
            print ("Reject")
        if count > 20:
            #The possibilities for the face to belong to a person
            pos = {}
            pos["Juntao"] = (cnt1+0.0)/count
            pos["Wentao"] = (cnt2+0.0)/count
            pos["Xiaotong"] = (cnt3+0.0)/count
            pos["Yi"] = (cnt4+0.0)/count
            pos["Reject"]= (cnt5+0.0)/count
            maxName = ''#name with max possibility
            maxPos = 0.0
            print (pos)
            
            for (k,v) in pos.items():
                if v > maxPos:
                    maxName = k
                    maxPos = v
            if maxName=="Reject":
                print ("Reject")
            else:
                print ('Hello! ' + maxName + '!')
    cv2.imshow('Vedio', frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()