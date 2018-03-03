import cv2
from mtcnn.mtcnn import MTCNN


video_capture = cv2.VideoCapture(0)
detect=MTCNN()
count=0
faceDataPath='facesData/negative/'

while True:
    ret, frame=video_capture.read()
    faces = detect.detect_faces(frame)
    for face in faces:
        # Draw rectangles around the faces
        x, y, w, h = face['box']
        faceFrame = frame[y:y + h, x:x + w]
        # write faceimage as jpg file
        cv2.rectangle(frame,(x,y),(x+w,y+h),[255,155,0],1)
        cv2.imshow('Vedio', frame)
        name=str(count)
        cv2.imwrite(faceDataPath + name+".jpg", faceFrame)
        count+=1
    if cv2.waitKey(1) & 0xFF == ord('q') or count==200:
        break
video_capture.release()
cv2.destroyAllWindows()