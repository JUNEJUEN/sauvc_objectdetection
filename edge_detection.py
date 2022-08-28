from cgi import MiniFieldStorage
import cv2
from cv2 import threshold
from cv2 import GaussianBlur
from cv2 import Canny
import numpy as np
import array as arr

def salt(img,n):
    for k in range(n):
        i = int(np.random.random()*img.shape[1]);
        j = int(np.random.random()*img.shape[0]);
        if img.ndim == 2:
            img[j,i] ==5
        elif img.ndim == 3:
            img[j,i,0] = 255
            img[j,i,1] = 255
            img[j,i,2] = 255
    return img

path = './OutputImages/gate_LowComplexityDCP.jpg'
frame = cv2.imread(path)
scale_percent = 40 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
frame = cv2.resize(frame, (width, height))

cv2.imshow('Original',frame)
cv2.waitKey(0)
cv2.destroyWindow('Original')


# Color filtering
Min = np.array([0, 19,  0])
Max = np.array([106, 255, 255])
mask = cv2.inRange(frame, Min, Max)
result = cv2.bitwise_and(frame, frame, mask=mask)

cv2.imshow('Color filtering', result)
cv2.waitKey(0)
cv2.destroyWindow('Color filtering')

# Thresholding
thresh = 0
maxValue = 255
processed_frame, thresholding_image = cv2.threshold(result, thresh, maxValue, cv2.THRESH_BINARY)

cv2.imshow('Thresholding', thresholding_image)
cv2.waitKey(0)
cv2.destroyWindow('Thresholding')



# Blur and Canny the image
cv2.namedWindow('Edges_median',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Edges_gaussian',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('ksize','Edges_median',1, 51, lambda x: None)
cv2.createTrackbar('threshold1_m', 'Edges_median', 10, 1000, lambda x: None)
cv2.createTrackbar('threshold2_m', 'Edges_median', 500, 1000, lambda x: None)
cv2.createTrackbar('k1','Edges_gaussian',1, 51, lambda x: None)
cv2.createTrackbar('k2', 'Edges_gaussian', 1, 51, lambda x: None)
cv2.createTrackbar('sigmaX','Edges_gaussian',0, 10, lambda x: None)
cv2.createTrackbar('threshold1_g', 'Edges_gaussian', 10, 1000, lambda x: None)
cv2.createTrackbar('threshold2_g', 'Edges_gaussian', 500, 1000, lambda x: None)

while True:
    ksize = cv2.getTrackbarPos('ksize','Edges_median')
    threshold1_m = cv2.getTrackbarPos('threshold1_m','Edges_median') 
    threshold2_m = cv2.getTrackbarPos('threshold2_m','Edges_median') 

    k1 = cv2.getTrackbarPos('k1','Edges_gaussian')
    k2 = cv2.getTrackbarPos('k2','Edges_gaussian')
    sigmaX = cv2.getTrackbarPos('sigmaX','Edges_gaussian')
    threshold1_g = cv2.getTrackbarPos('threshold1_g','Edges_median') 
    threshold2_g = cv2.getTrackbarPos('threshold2_g','Edges_median') 

    if ksize % 2 == 0:
        ksize += 1
    if k1 %2 == 0:
        k1 += 1
    if k2 %2 == 0:
        k2 +=1
    
    
    median_blur = cv2.medianBlur(thresholding_image, ksize)
    gaussian_blur = cv2.GaussianBlur(thresholding_image,(k1, k2),sigmaX=sigmaX)
    edges_median = cv2.Canny(image=median_blur, threshold1=threshold1_m, threshold2=threshold2_m)
    edges_gaussian = cv2.Canny(image=gaussian_blur, threshold1=threshold1_g,threshold2=threshold2_g)

    cv2.imshow('Edges_median',edges_median)
    cv2.imshow('Edges_gaussian',edges_gaussian)

    key = cv2.waitKey(1)
    if key == 27:
            break

cv2.destroyAllWindows()

# Line detection
cv2.namedWindow('lines',cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('threshold', 'lines', 0, 100, lambda x: None)
cv2.createTrackbar('minLineLength', 'lines', 10, 200, lambda x: None)
cv2.createTrackbar('maxLineGap','lines',5, 100, lambda x: None)

while True:
    threshold = cv2.getTrackbarPos('threshold','lines')
    minLineLength = cv2.getTrackbarPos('minLineLength','lines')
    maxLineGap = cv2.getTrackbarPos('maxLineGap','lines')
    rho = 1
    theta = np.pi /180


    lines = cv2.HoughLinesP(edges_gaussian, rho, theta, threshold, np.array([]),minLineLength, maxLineGap)

    lines_edges = frame.copy()

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lines_edges,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('lines',lines_edges)
    key = cv2.waitKey(1)
    if key == 27:
            break
    
cv2.destroyAllWindows()