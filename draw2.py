# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:06:53 2021

@author: Oscar
"""
import cv2 as cv
import numpy as np

img= cv.imread('C:/Users/Oscar/Pictures/Random/tesis.jpg')
img2= cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#%%
#Total pixels
img_shape= img.shape
pixels= img.size
pixels2= img2.size

print(pixels)
print(img_shape[0]*img_shape[1]*img_shape[2])

#Show image (Name, image)
cv.imshow('Original', img)
#cv.imshow('Gray', img2)
cv.waitKey(0)

#BGR
xp= 40
yp= 6
blue= img[yp, xp, 0]
green= img[yp, xp, 1]
red= img[yp, xp, 2]
print((blue, green, red))
#BGR
bgr= img[yp, xp]
print(bgr)
#Change pixel value
green2= img[6, 40, 1]
img[6, 40, 1]= 255

#Top
top_left_corner= img[0:500, 0:500]
cv.imshow("Top-left corner", top_left_corner)
cv.waitKey(0)
top_right_corner= img[0:500, 500:-1]
cv.imshow("Top-right corner", top_right_corner)
cv.waitKey(0)

#Bottom
bottom_left_corner= img[500:-1, 0:500]
cv.imshow("Bottom-left corner", bottom_left_corner)
cv.waitKey(0)
bottom_right_corner= img[500:-1, 500:-1]
cv.imshow("Bottom-right corner", bottom_right_corner)
cv.waitKey(0)

#Change section color
img[0:500, 0:500]= (255, 0, 0)
cv.imshow("Top-left change", img)
cv.waitKey(0)


#%%
rgb= cv.imread('C:/Users/Oscar/Documents/Python Scripts/Imagenes/circles.jpeg')
rgb2= cv.imread('C:/Users/Oscar/Pictures/half-life-2-animated.jpg')

def split_rgb(img):
    #Split
    blue= img[:, :, 0]
    green= img[:, :, 1]
    red= img[:, :, 2]
    return blue, green, red
    
def gray_scale(blue, green, red):
    #Blue
    cv.imshow('Blue', blue)
    cv.waitKey(0)
    #Green
    cv.imshow('Green', green)
    cv.waitKey(0)
    #Red
    cv.imshow('Red', red)
    cv.waitKey(0)
    
def merge(blue, green, red):
    zeros= np.zeros(blue.shape, np.uint8)
    
    #Merges
    blueBGR= cv.merge((blue, zeros, zeros))
    greenBGR= cv.merge((zeros, green, zeros))
    redBGR= cv.merge((zeros, zeros, red))
    #Blue
    cv.imshow('blue BGR', blueBGR)
    cv.waitKey(0)
    #Green
    cv.imshow('green BGR', greenBGR)
    cv.waitKey(0)
    #Red
    cv.imshow('red BGR', redBGR)
    cv.waitKey(0)
        
b, g, r= split_rgb(rgb)
gray_scale(b, g, r)
merge(b, g, r)
