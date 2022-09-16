from cmath import rect
import re
import cv2
import numpy as np
import array as arr
import os
import natsort

def RecoverHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        #sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance


class Rectangle():
    def __init__(self, contours) -> None:
        self.rect = cv2.minAreaRect(contours)
        self.box = cv2.boxPoints(self.rect)
        self.box = np.int0(self.box)
        self.left_top_pt = self.box[1]
        self.right_bottom_pt = self.box[3]
        self.line_length = []
        self.max_length = {"length": 0, "coordinate" : [0,0,0,0]}
        
        for point in range(4):
            if point < 3:
                x1 = self.box[point][0]
                y1 = self.box[point][1]
                x2 = self.box[point+1][0]
                y2 = self.box[point+1][1]
                
            else :
                x1 = self.box[point][0]
                y1 = self.box[point][1]
                x2 = self.box[0][0]
                y2 = self.box[0][1]

            length = ((x1 - x2)**2 + (y1 - y2)**2) ** (1/2)
            self.line_length.append(length)
            if length > self.max_length["length"]:
                    self.max_length["length"] = length
                    self.max_length["coordinate"] = [x1, y1, x2, y2]

        self.area = max(self.line_length) * min(self.line_length)

        if self.max_length["coordinate"][0] -  self.max_length["coordinate"][2] == 0:
            self.slope = -1 
        else:
            self.slope = abs((self.max_length["coordinate"][1] -  self.max_length["coordinate"][3]) / (self.max_length["coordinate"][0] -  self.max_length["coordinate"][2]) )

        
    def drawing(self,img):
        for point in range(4):
            if point < 3:
                x1 = self.box[point][0]
                y1 = self.box[point][1]
                x2 = self.box[point+1][0]
                y2 = self.box[point+1][1]
            else :
                x1 = self.box[point][0]
                y1 = self.box[point][1]
                x2 = self.box[0][0]
                y2 = self.box[0][1]

            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
        return img



def find_rectangle_left(img, frame):
    original = frame.copy()

    contours, _hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("number of countours: %d" % len(contours))

    rectangle_list = []

    for cnt in contours:
        rectangle_list += [Rectangle(cnt)]

    # Sorting by their area.
    rectangle_list.sort(key=lambda x : x.area, reverse = True)      

    # Validation check : Not enough rectangle
    if len(rectangle_list) < 2:
        return None

    possible_gate = []
    if len(rectangle_list) >= 4:
        for i in range(4):      # Only consider the largest 4 rectangles
            if rectangle_list[i].slope == -1 or rectangle_list[i].slope > 11:               #tan(85) = 11  
                possible_gate += [rectangle_list[i]]
    else:
        possible_gate = rectangle_list

    # Validation check : Not enough rectangle
    if len(possible_gate) < 2:  
        return None

    # The lower/larger, the foremost (the possibility that it is underwater is higher)
    possible_gate.sort(key=lambda x : x.left_top_pt[1], reverse=True)       
    #print("number of possible recantagle for gate: ", len(possible_gate))

    # Validation check : Prevent that one rectangle is on the top (shadow) and one rectangle is  underwater
    if abs(possible_gate[0].left_top_pt[1] - possible_gate[1].left_top_pt[1]) > 150 :
        return None

    most_possible_gate = []
    for i in range(2):
        most_possible_gate += [possible_gate[i]]

    most_possible_gate.sort(key=lambda x : x.left_top_pt[0]) 
    
    for  rectangle in most_possible_gate:
        #print(rectangle.box)
        original = rectangle.drawing(original)  
    cv2.imshow('left_rectangle',original)
    cv2.waitKey(1)

    return most_possible_gate

def find_rectangle_right(img, frame):
    original = frame.copy()

    contours, _hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("number of countours: %d" % len(contours))

    rectangle_list = []

    for cnt in contours:
        rectangle_list += [Rectangle(cnt)]

    # Sorting by their area.
    rectangle_list.sort(key=lambda x : x.area, reverse = True)      

    # Validation check : Not enough rectangle
    if len(rectangle_list) < 2:
        return None

    possible_gate = []
    if len(rectangle_list) >= 4:
        for i in range(4):      # Only consider the largest 4 rectangles
            if rectangle_list[i].slope == -1 or rectangle_list[i].slope > 11:               #tan(85) = 11  
                possible_gate += [rectangle_list[i]]
    else:
        possible_gate = rectangle_list

    # Validation check : Not enough rectangle
    if len(possible_gate) < 2:  
        return None

    # The lower/larger, the foremost (the possibility that it is underwater is higher)
    possible_gate.sort(key=lambda x : x.left_top_pt[1], reverse=True)       
    #print("number of possible recantagle for gate: ", len(possible_gate))

    # Validation check : Prevent that one rectangle is on the top (shadow) and one rectangle is  underwater
    if abs(possible_gate[0].left_top_pt[1] - possible_gate[1].left_top_pt[1]) > 150 :
        return None

    most_possible_gate = []
    for i in range(2):
        most_possible_gate += [possible_gate[i]]

    most_possible_gate.sort(key=lambda x : x.left_top_pt[0]) 
    
    for  rectangle in most_possible_gate:
        #print(rectangle.box)
        original = rectangle.drawing(original)  
    cv2.imshow('left_rectangle',original)
    cv2.waitKey(1)

    return most_possible_gate


path = "/home/kyapo/Desktop/navigation/stereo_calibration/pooltesting_stereo_video_27_8/q_gate3/"
files = os.listdir(path)
files =  natsort.natsorted(files)

for i in range(len(files)):
    file = files[i]
    img = cv2.imread(path+file)
    frame = RecoverHE(img)
    
    scale_percent = 100 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height))

    cv2.imshow('Original',frame)
    cv2.waitKey(1)

    # Convert color space
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Color filtering
    Min = np.array([ 0,  0, 83])
    Max = np.array([ 64, 143, 255])
    mask = cv2.inRange(frame, Min, Max)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Color filtering', result)
    cv2.waitKey(1)

    # Thresholding
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = 0
    maxValue = 255
    processed_frame, thresholding_image = cv2.threshold(result, thresh, maxValue, cv2.THRESH_BINARY)

    #thresholding_image = salt(thresholding_image, 10)
    thresholding_image= cv2.dilate(thresholding_image,(9,9), iterations=10)

    cv2.imshow('Thresholding',thresholding_image )
    cv2.waitKey(1)

    left_frame = frame[0:720, 0: 1280]
    thresholding_left = thresholding_image[0:720, 0: 1280]
    right_frame = frame[0:720, 1280:2560]
    thresholding_right = thresholding_image[0:720, 1280:2560]

    #gate = find_rectangle(thresholding_image, frame)
    gate_left = find_rectangle_left(thresholding_left, left_frame)
    gate_right = find_rectangle_right(thresholding_right, right_frame)

    
    if gate_left is not None:
        gate_left_location = (gate_left[0].left_top_pt, gate_left[1].right_bottom_pt)
        left_frame = cv2.rectangle(left_frame, gate_left_location[0], gate_left_location[1], (255, 0,0), 2)
        cv2.imshow('gate_left', left_frame)
        cv2.waitKey(1)

    if gate_right is not None:
        gate_right_location = (gate_right[0].left_top_pt, gate_right[1].right_bottom_pt)
        right_frame = cv2.rectangle(right_frame, gate_right_location[0], gate_right_location[1], (255, 0,0), 2)
        cv2.imshow('gate_right', right_frame)
        cv2.waitKey(1)
@