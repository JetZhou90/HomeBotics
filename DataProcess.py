import pickle
from PIL import Image
import numpy as np
import os
import pylab


i=0
w=80
h=80
faces=np.empty((792,w*h)) # 792, w * h faces
faces_label=[]   # labels
cls=4

# one-hot

def one_hot(index,n):
    lb=np.zeros(n,dtype=np.int)
    lb[index]=1
    return lb



# read image as num

for filename in os.listdir(r'facesData/1'):
    if(filename!='Thumbs.db'):
        basedir = 'facesData/1/'
        image = Image.open(basedir + filename)
        image = image.resize((w, h), Image.BILINEAR)
        image = image.convert("L")
        img_ndarray = np.asarray(image, dtype='float32')/255
        faces[i]=np.ndarray.flatten(img_ndarray)
        label = one_hot(0,cls)
        faces_label.append(label)
        i = i + 1

for filename in os.listdir(r'facesData/2'):
    if(filename!='Thumbs.db'):
        basedir = 'facesData/2/'
        image = Image.open(basedir + filename)
        image = image.resize((w, h), Image.BILINEAR)
        image = image.convert("L")
        img_ndarray = np.asarray(image, dtype='float32')/255
        faces[i]=np.ndarray.flatten(img_ndarray)
        label = one_hot(1,cls)
        faces_label.append(label)
        i = i + 1

for filename in os.listdir(r'facesData/3'):
    if(filename!='Thumbs.db'):
        basedir = 'facesData/3/'
        image = Image.open(basedir + filename)
        image = image.resize((w, h), Image.BILINEAR)
        image = image.convert("L")
        img_ndarray = np.asarray(image, dtype='float32')/255
        faces[i]=np.ndarray.flatten(img_ndarray)
        label = one_hot(2,cls)
        faces_label.append(label)
        i = i + 1

for filename in os.listdir(r'facesData/4'):
    if(filename!='Thumbs.db'):
        basedir = 'facesData/4/'
        image = Image.open(basedir + filename)
        image = image.resize((w, h), Image.BILINEAR)
        image = image.convert("L")
        img_ndarray = np.asarray(image, dtype='float32')/255
        faces[i]=np.ndarray.flatten(img_ndarray)
        label = one_hot(3,cls)
        faces_label.append(label)
        i = i + 1
faces_label=np.asarray(faces_label,dtype=np.int)

print (faces_label)
## image show

# img0=faces[200].reshape(w,h)
# pylab.imshow(img0)
# pylab.gray()
# pylab.show()

## save images

# write_file=open('Faces.pkl','wb')
# pickle.dump(faces,write_file,-1)
# pickle.dump(faces_label,write_file,-1)
# write_file.close()

## load images

# read_file=open('faces.pkl','rb')
# faces=pickle.load(read_file)
# label=pickle.load(read_file)
# read_file.close()
# print (faces[800])