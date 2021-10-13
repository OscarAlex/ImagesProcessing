# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:10:12 2021

@author: Oscar
"""
import numpy as np
import cv2

#%%
img_size = (200,200)
img = np.ones(img_size) * 255

# polar equation
theta = np.linspace(0, np.pi, 1000)
r = 1 / (np.sin(theta) - np.cos(theta))

# polar to cartesian
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

x,y = polar2cart(r, theta)
x1, x2, y1, y2 = x[0], x[1], y[0], y[1]

# line equation y = f(X)
def line_eq(X):
    m = (y2 - y1) / (x2 - x1)
    return m * (X - x1) + y1

line = np.vectorize(line_eq)

x = np.arange(0, img_size[0])
y = line(x).astype(np.uint)

cv2.line(img, (x[0], y[0]), (x[-1], y[-1]), (0,0,0))
cv2.imshow("foo",img)
cv2.waitKey()


#%%
width, height = 500, 300
x1, y1 = 0, 0
x2, y2 = 500, 300
image = np.zeros((height, width, 3), dtype=np.uint8)
#Background
image[:,:,:] = (255, 0, 0)

line_thickness = 50
#cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 30)
cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=line_thickness)
cv2.line(image, (x2, y1), (x1, y2), (255, 255, 255), thickness=line_thickness)
cv2.imshow("foo",image)
cv2.waitKey()

#%%
img1 = cv2.imread('C:/Users/Oscar/Pictures/MJ/1.png')
img2 = cv2.imread('C:/Users/Oscar/Pictures/MJ/2.png')

dst = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.circle(img,(447,63), 63, (0,0,255), -1)

cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

cv2.imshow("foo",img)
cv2.waitKey()
