import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import PIL
from PIL import Image

image_path = "standardleaves1"

def loadImages(path):
    img_files = sorted([os.path.join(path, file)
                for file in os.listdir(path) if file.endswith('.jpg')])
    return img_files

#calls all preprocessing methods included
def preprocessing(data):
    resize(data)
    removeNoise(data)
    segAndMorph(data)
    

#make all the images 1200 by 900 using PIL
def resize(data):
    basewidth = 1200
    for i in data:
        img = Image.open(i)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save(i)

#removing noise by adding gaussian blur using OpenCV
def removeNoise(data):
    for i in data:
        img = cv2.imread(i)
        blur = cv2.GaussianBlur(img,(5,5),0)
        cv2.imwrite(i, blur)

#segmenting and morhping images
def segAndMorph(data):
    for i in data:
        img = cv2.imread(i)
        #segmentation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #further noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,
                                   iterations=2)
        #background
        back = cv2.dilate(opening, kernel, iterations=3)
        #foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, fore = cv2.threshold(dist_transform, 0.7*dist_transform.max(),
                                  255, 0)
        #unknown region
        fore = np.uint8(fore)
        unknown = cv2.subtract(back, fore)

        ret, marker = cv2.connectedComponents(fore)
        marker = marker + 1
        marker[unknown == 255] = 0
        marker = cv2.watershed(img, marker)
        img[marker == -1] = [255, 0, 0]

        #saving marker array as an image
        mpimg.imsave(i, marker)

        
# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    
def main():
    test_img = loadImages(image_path)
    preprocessing(test_img)
