from cmath import rect
from lib2to3.pgen2.pgen import generate_grammar
from queue import Empty
import re
from reprlib import recursive_repr
from turtle import pos
import cv2
import numpy as np
import array as arr
import os
import natsort


# Image preprocessing
def RecoverHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        #sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

def img_preprocessing(img):
    frame = RecoverHE(img)
    
    #scale_percent = 100 # percent of original size
    #width = int(frame.shape[1] * scale_percent / 100)
    #height = int(frame.shape[0] * scale_percent / 100)
    #frame = cv2.resize(frame, (width, height))

    #cv2.imshow('Original',frame)
    #cv2.waitKey(0)

    # Convert color space
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Color filtering
    Min = np.array([  0, 170, 0])
    Max = np.array([ 255, 255, 255])
    mask = cv2.inRange(frame, Min, Max)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #cv2.imshow('Color filtering', result)
    #cv2.waitKey(0)

    # Thresholding
    #result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = 0
    maxValue = 255
    processed_frame, thresholding_image = cv2.threshold(result, thresh, maxValue, cv2.THRESH_BINARY)

    thresholding_image = cv2.cvtColor(thresholding_image, cv2.COLOR_BGR2GRAY)
    thresholding_image= cv2.dilate(thresholding_image,(9,9), iterations=10)

    cv2.imshow('Thresholding',thresholding_image )
    cv2.waitKey(0)

    return thresholding_image

def order_points(pts):
    ''' sort rectangle points by clockwise '''
    sort_x = pts[np.argsort(pts[:, 0]), :]
    
    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:,1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:,1]), :]
    
    return np.concatenate((Left, Right), axis=0)

class Rectangle():
    def __init__(self, contours) -> None:
        self.rect = cv2.minAreaRect(contours)
        self.box = cv2.boxPoints(self.rect)
        self.box = np.int0(self.box)
        #self.box = order_points(box)

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


def find_rectangle(thres_img, l_r_frame, w,h):

    original = l_r_frame.copy()
    indepent = False

    contours, _hierarchy = cv2.findContours(thres_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("number of countours: %d" % len(contours))

    rectangle_list = []
    for cnt in contours:
        rectangle_list += [Rectangle(cnt)]

    # Filtering : the slope of the triangles should more than 11 (almost vertical) , the area should not be too small
    possible_gate = []
    for i in range(len(rectangle_list))  :
        if (rectangle_list[i].slope == -1 or rectangle_list[i].slope > 11) and rectangle_list[i].area > 2000:               #tan(85) = 11  
            possible_gate += [rectangle_list[i]]
    
    # Validation check : possible_gate is empty
    if len(possible_gate) == 0:
        print("Validation check failed : Cannot find any retangle ")
        return None

    # Validation check : Not enough rectangle
    if len(possible_gate) == 1:
        print("Validation check failed : Not enought rectangle")
        #for  rectangle in possible_gate:
        #    original = rectangle.drawing(original) 
        #cv2.imshow('rectangle',original)
        #cv2.waitKey(0)
        indepent = True
        gate_info = (indepent, possible_gate[0].left_top_pt, possible_gate[0].right_bottom_pt, None, None)
        return gate_info

    # Sorting by their area.
    possible_gate.sort(key=lambda a : a.area, reverse = True)

    # Validation check : the difference of area of two largest rectangle are too large
    if possible_gate[0].area / possible_gate[1].area >= 4 :
        print("Validation check failed : the difference of area of two largest rectangle are too large")
        indepent = True
        gate_info = (indepent, possible_gate[0].left_top_pt, possible_gate[0].right_bottom_pt, None, None)
        return gate_info

    # Only consider the largest 4 rectangles
    more_possible_gate = []
    if len(possible_gate) < 4:
        more_possible_gate = possible_gate
    else :
        for i in range(4):
            more_possible_gate += [possible_gate[i]]

    # The lower/larger, the foremost (the possibility that it is underwater is higher)
    #more_possible_gate.sort(key=lambda y : y.left_top_pt[1], reverse=True)    
    more_possible_gate.sort(key=lambda y : y.right_bottom_pt[1], reverse=True)     
    #print("number of possible recantagle for gate: ", len(more_possible_gate))

    # Validation check : Prevent that one rectangle is on the top (shadow) and one rectangle is  underwater
    if abs(more_possible_gate[0].left_top_pt[1] - more_possible_gate[1].left_top_pt[1]) > 150  and abs(more_possible_gate[0].right_bottom_pt[1] - more_possible_gate[1].right_bottom_pt[1]) > 300 :
        #for  rectangle in more_possible_gate:
        #    original = rectangle.drawing(original)  
        #    cv2.imshow('Validation check',original)
        #    cv2.waitKey(0)
        print("Validation check failed : one rectangle is on the top (shadow) and one rectangle is  underwater :", more_possible_gate[0].left_top_pt, more_possible_gate[1].left_top_pt, more_possible_gate[0].right_bottom_pt,more_possible_gate[1].right_bottom_pt)
        indepent = True
        gate_info = (indepent, more_possible_gate[0].left_top_pt, more_possible_gate[0].right_bottom_pt, None, None)
        return gate_info

    
    
    most_possible_gate = []
    for i in range(2):
        print("Rectangle area:", more_possible_gate[i].area)
        most_possible_gate += [more_possible_gate[i]]

    # Sorting them accoring to x coordinate
    most_possible_gate.sort(key=lambda x : x.left_top_pt[0]) 
    
    # Validation check : whether the gate area is too small
    length = (most_possible_gate[1].right_bottom_pt[0] - most_possible_gate[0].left_top_pt[0]) #x
    width = (most_possible_gate[1].right_bottom_pt[1] - most_possible_gate[0].left_top_pt[1]) #y
    gate_area = length * width 
    if gate_area <= 50000:
        print("Validation check failed : The gate area is too small ")
        return None

    gate_mid_pt = (most_possible_gate[0].left_top_pt[0] + int(length / 2), most_possible_gate[0].left_top_pt[1] + int(width / 2))
    gate_info = (indepent, most_possible_gate[0].left_top_pt, most_possible_gate[1].right_bottom_pt, gate_area, gate_mid_pt)
    
    # Drawing the rectangles and point
    middle_w = int(w/2)
    middle_h = int(h/2)
    for  rectangle in most_possible_gate:
        original = rectangle.drawing(original) 
    original = cv2.circle(original, gate_mid_pt, radius=0, color=(255, 0, 255), thickness=6)
    original = cv2.circle(original, (middle_w, middle_h), radius=0, color=(125, 0, 125), thickness=6)
    cv2.imshow('rectangle',original)
    cv2.waitKey(0)

    return gate_info


class Gate():
    def __init__(self, gate_l, gate_r, w, h) -> None:
        self.gate_left = gate_l
        self.gate_right = gate_r

        # w = 1280  h = 720
        self.width = w
        self.hight = h
        self.middle_w = w/2      #640
        self.middle_h = h/2
        '''
        8 : Both are none gate  : cannot see any gate
        7 : Left is none gate, right is a single gate on the right
        6 : Left is a single gate, right is none gate on the left
        5 : Both are single gate, but dont know it is going to turn left or not.
        4 : Both are single gate, but one is on the most left, but one is on the most right
        3 : Left is single gate, right is full gate
        2 : Left is full gate, right is single gate
        1 : Both are full gate
        0 : Bad detection
        '''
        if (gate_l is None and gate_r is None) :
            self.status = 8
        elif (gate_l is None and gate_r[0] == False):
            self.status = 0
        elif (gate_l is None and gate_r[0] == True):
            if gate_r[1][0] > w-self.middle_w/4:
                self.status = 7
            else :
                self.status = 0
        elif (gate_l[0] == False and gate_r is None):
            self.status = 0
        elif (gate_l[0] == True and gate_r is None):
            if gate_l[1][0] < self.middle_w/4:
                self.status = 6
            else :
                self.status = 0
        elif (gate_l[0] == True and gate_r[0] == True):
            if (gate_l[1][0] > gate_r[1][0]) and abs(gate_l[1][1] - gate_r[1][1]) < 50:
                self.status = 5
            elif (gate_l[1][0] < gate_r[1][0]) and ((w - gate_l[1][0]) + gate_r[1][0])  > 2200:
                self.status = 4
            else :
                self.status = 0
        elif (gate_l[0] == True and gate_r[0] == False):
            if gate_l[1][0] > gate_r[1][0] and abs(gate_l[1][1] - gate_r[1][1]) < 50 and abs(gate_r[4][0]-self.middle_w)>30 :
                self.status = 3
            else :
                self.status = 0
        elif (gate_l[0] == False and gate_r[0] == True):
            if gate_l[2][0] > gate_r[2][0] and abs(gate_l[2][1] - gate_r[2][1]) < 50 and abs(gate_l[4][0]-self.middle_w)>30 :
                self.status = 2
            else : 
                self.status = 0
        else :
            self.status = 1
    
        self.behaviour = None
        self.distance = None


    def set_behaviour(self, dir):
        self.behaviour = dir
        if (dir == 'left'):
            self.turn_left()
        elif (dir == 'right'):
            self.turn_right()
        elif (dir == 'front'):
            self.front()
        elif (dir == 'center_front'):
            self.center_front()
        elif (dir == 'stop'):
            self.stop()
    
    def turn_left(self):
        if self.status == 2:
            self.distance = abs(self.middle_w - self.gate_left[4][0])
        elif self.status == 6:
            self.distance = abs(self.width - self.gate_left[1][0] * 2 )     # 總寬 - 左上角坐標 x 2 
        elif self.status == 7:
            self.distance = abs(self.width - self.gate_right[1][0])  # 總寬 - 左上角坐標
        elif self.status == 5:
            self.distance = abs(self.width - self.gate_left[1][0])
        print("turn_left: ", - self.distance)
        #return self.distance

    def turn_right(self):
        if self.status == 3:
            self.distance = abs(self.gate_right[4][0] - self.middle_w)
        elif self.status == 6 :
            self.distance = abs(self.gate_left[1][0])
        elif self.status == 7:
            self.distance = abs(self.width - (self.width - self.gate_right[1][0])*2 )     
        elif self.status == 5:
            self.distance = abs(self.gate_left[1][0])
        print("turn_right: ", self.distance)
        #return self.distance

    def front(self):
        if self.status == 1:
            self.distance = self.middle_w - self.gate_left[4][0]
            self.distance = -self.distance      # -(+) : left  -(-) : right
        print("front: ", self.distance)
        #return self.distance

    def center_front(self):
        self.distance = 0
        print("center_front: ", self.distance)
        #return self.distance

    def stop(self):
        print("stop")


def check_global_status(gate_object: Gate,  w : int):
    global gate_count,  gate_dataset, ready_pass, frame, c_gate, detec_err_count

    detection_err = False
    s = gate_object.status
    middle_w = w/2

    if gate_count != 0:
        ps = gate_dataset[-1].status
        print("precious status: ", ps)
        print("status: ", s )
    else:
         ps = -1

    if s != 0 :
        if gate_count == 0:     #第一張合理的的image
            print("The first valid image detected")
            if s == 8:
                gate_object.set_behaviour('center_front')
            else:
                print("You hv seen a gate")
                if s == 2 :
                    gate_object.set_behaviour('left')
                elif s == 3 :
                    gate_object.set_behaviour('right')
                elif s == 5:      # Not sure status , from previous experience, the auv always drifts to the right.
                    gate_object.set_behaviour('left')
                elif s == 1:
                    gate_object.set_behaviour('front')
                if s != 7 or s != 6:
                    c_gate = True
            
        if ready_pass == True:     # 確定會即將過gate之後
            print("You r ready to pass the gate")
            if s == 8 :         
                count = 0
                for i in range(1, 6):   # -1 to -5
                    if gate_dataset[-i].status == 8 :
                        count += 1
                    else :
                        gate_object.set_behaviour('front')          # 繼續直走 還差一點就過gate
                        break
                if count == 10:      # 連續十張都是None gate
                    gate_object.set_behaviour('stop')

            elif ps == s:       # 4 == 4  6 == 6  7 ==7
                # 繼續正中走
                # 如果發現還是有向左偏移 則繼續矯正
                # 如果發現還是有向右偏移 則繼續矯正
                gate_object.set_behaviour(gate_dataset[-1].behaviour)
            
            elif s == 6 :
                gate_object.set_behaviour('right')
            elif s == 7 :
                gate_object.set_behaviour('left')
            elif s == 4 :
                gate_object.set_behaviour('center_front') 
            elif s == 5:
                if gate_object.gate_right[1][0] < middle_w/4:
                    gate_object.set_behaviour('right')
                elif gate_object.gate_left[1][0] > (w -  middle_w/4 ):
                    gate_object.set_behaviour('left')

            #else:
            #    detection_err = True


        if ready_pass == False and gate_count != 0 :
            print("Detecting")

            # 當之前都沒有看到過任何gate，第一次看到後。
            if ps == 8 and c_gate == False and s != 8 :
                if s == 1:
                    gate_object.set_behaviour('front')
                elif s == 2 or s == 5:        
                    gate_object.set_behaviour('left')       # 初始5 default 偏右 根據經驗
                elif s == 3 :
                    gate_object.set_behaviour('right')
                if s == 6 or s == 7:
                    detection_err = True
                else :
                    print("You hv seen a gate")
                    c_gate = True

            # 正在矯正中/前進
            elif ps == 1 and s == 1:
                gate_object.set_behaviour("front")
            elif (ps == 2 and s == 2) :
                gate_object.set_behaviour("left")
            elif (ps == 6 and s == 6):
                gate_object.set_behaviour("left")
            elif (ps == 3 and s == 3) :
                gate_object.set_behaviour("right")
            elif (ps == 7 and s == 7):
                gate_object.set_behaviour("right")
            elif (ps == 5 and s == 5):
                gate_object.set_behaviour(gate_dataset[-1].behaviour)     
            elif (ps == 8 and ps == 8):
                gate_object.set_behaviour('center_front') 

            # 1, 2 ,3
            elif ps == 1 and s == 2:
                gate_object.set_behaviour('left')     # 有向右偏移, 導致只有左邊相機能看到完整的gate
            elif ps == 1 and s == 3:
                gate_object.set_behaviour('right')    # 有向左偏移，導致只有右邊相機能看到完整的gate
            elif (ps == 2 or ps == 3) and s == 1:
                gate_object.set_behaviour('front')    # 矯正成功
            elif ps == 2 and s == 3:
                gate_object.set_behaviour('right')    # 過分矯正
            elif ps == 3 and s == 2:
                gate_object.set_behaviour('left')     # 過分矯正

            # 5
            elif (ps == 6 or ps == 7 or ps == 5 or ps == 4) and s == 1:
                gate_object.set_behaviour('front')      # 上一個gate 的 status(5) 是 detection error
                gate_dataset.remove(gate_dataset[-1])
            elif ps == 5 and gate_dataset[-1].behaviour == 'left'and  s == 2:
                gate_object.set_behaviour('left')       # 矯正成功, 繼續矯正
            elif ps == 5 and gate_dataset[-1].behaviour == 'right' and s == 3:
                gate_object.set_behaviour("right")
            elif ps == 5 and gate_dataset[-1].behaviour == 'left'and s == 7:
                # 繼續向左偏移中（矯正失敗) 由於如果初始gate是5，則並不清楚應該偏左偏右，默認偏右（行動向左），但如果實際情況是偏左，向左只會令其越偏，導致7
                gate_object.set_behaviour('right')
                # 建議平移而不往前
            elif ps == 6 and s == 5:
                gate_object.set_behaviour("left")   # 矯正成功, 繼續矯正
            elif ps == 7 and s == 5:
                gate_object.set_behaviour("right")   # 矯正成功, 繼續矯正

            # Ready to pass gate case
            elif (gate_count> 20) and ( ps == 4 and gate_dataset[-2].status == 4 )and s == 4  :
                gate_object.set_behaviour('center_front')
                print ("Ready to pass gate")      
                ready_pass = True

            else :
                if c_gate == True :
                    c_full_gate_count = 0
                    for i in range (3):
                        if gate_dataset[-i].status == 1:
                            c_full_gate_count += 1
                    if c_full_gate_count == 3 and ( s == 8  or s == 4) :
                        # 前幾次都能看到full gate， 這次只剩兩個柱子在各自的鏡頭裏， 代表即將過gate，而且是正中過 
                        # 前幾次都能看到full gate， 忽然看不大代表即將過gate
                        gate_object.set_behaviour('center_front')
                        print ("Ready to pass gate")      
                        ready_pass = True

                    elif c_full_gate_count == 3 and s == 6  :
                        gate_object.set_behaviour('right')      # 前幾次都能看到full gate, 忽然左邊有柱子， 代表即將過gate 但會偏左
                        print ("Ready to pass gate")
                        ready_pass = True
                    
                    elif c_full_gate_count == 3 and s == 7 :
                        gate_object.set_behaviour('left')       # 前幾次都能看到full gate, 忽然右邊有柱子， 代表即將過gate 但會偏右
                        print ("Ready to pass gate")
                        ready_pass = True
                    
                    elif c_full_gate_count == 3 and s == 5:
                        if gate_object.gate_right[1][0] < middle_w/4:
                            gate_object.set_behaviour('right')
                            ready_pass = True
                        elif gate_object.gate_left[1][0] > (w -  middle_w/4 ):
                            gate_object.set_behaviour('left')
                            ready_pass = True

            '''
            # Detection error case
            elif (ps == 2 or ps == 3) and (s == 5 or s == 6 or s == 7):         
                detection_err = True                        # 2 或 3 不會朝相反方向走
                print("Detection error : (ps == 2 or ps == 3) and (s == 5 or s == 6 or s == 7)")
            elif (ps == 6 or ps == 7) and (s == 2 or s == 3):        
                detection_err = True                        # 6 或者 7會先經歷 5, 再到 2 or 3
                print("Detection error : (ps == 6 or ps == 7) and (s == 2 or a == 3)")
            elif (ps == 1 and s == 5) :
                detection_err = True                        # 1 會先經歷 2 或者 3 , 直接變成5不應該發生 
                print("Detection error : (ps == 1 and s == 5)") 
            elif ps == 5 and  gate_dataset[-1].behaviour == 'left' and s == 6:
                #gate_object.set_behaviour('left')          # 繼續向右偏移中 (矯正失敗) 不應該發生
                print("Detection error : ps == 5 and  gate_dataset[-1].behaviour == 'left' and s == 6")
                detection_err = True
            elif ps == 5  and s == 4:   
                detection_err = True    
                print("You have Detection error : ps == 5  and s == 4 ")                    
            elif ps == 4 and (s == 5 or s == 3 or s == 2) :
                detection_err = True
                print("You have Detection error : ps == 4 and (s == 5 or s == 3 or s == 2) ")
            '''

        #if detection_err:
        #    detec_err_count += 1            # Accumulate number of detection error 

        if detection_err == False and gate_object.behaviour is not None:
            detec_err_count = 0
            gate_dataset += [gate_object]
            gate_count += 1

    
    print("behaviour: ", gate_object.behaviour)
    print("distance:", gate_object.distance)
    print()

    # Create an empty placeholder for displaying the values
    placeholder = np.zeros((frame.shape[0],500,3),dtype=np.uint8)
    # fill the placeholder with the values of color spaces
    cv2.putText(placeholder, "previous status:  {}".format(ps), (20, 70), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "status: {} ".format(gate_object.status), (20, 140), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "behaviour:  {}".format(gate_object.behaviour), (20, 210), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "distance:  {}".format(gate_object.distance), (20, 280), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(placeholder, "detection error:  {}".format(detection_err), (20, 350), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "ready to pass gate:  {}".format(ready_pass), (20, 420), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "Have seen gate:  {}".format(c_gate), (20, 490), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
    combinedResult = np.hstack([frame,placeholder])
    combinedResult = np.hstack([frame,placeholder])
    cv2.imshow('frame',combinedResult)
    cv2.waitKey(0)

    


#path = "/home/kyapo/Desktop/navigation/stereo_calibration/pooltesting_stereo_video_27_8/q_gate4/"
#path = '.\\q_gate3\\'
path = '.\\q_gate1\\q_gate1\\'
files = os.listdir(path)
files =  natsort.natsorted(files)

height = 720
middle = 1280
width = 2560

gate_count = 0
ready_pass = False
c_gate = False
detec_err_count = False
gate_dataset = []

cv2.namedWindow("frame", cv2.WINDOW_NORMAL )
cv2.resizeWindow("frame", 1280, 360)

cv2.namedWindow('rectangle', cv2.WINDOW_NORMAL )
cv2.resizeWindow('rectangle', 640, 180)

cv2.namedWindow('Thresholding',cv2.WINDOW_NORMAL )
cv2.resizeWindow("Thresholding", 1280, 360)

for i in range(len(files)):
    file = files[i]
    frame = cv2.imread(path+file)
    #cv2.imshow('frame', frame)
    #n cv2.waitKey(0)
    print("Now processing :", file)

    thresholding_image = img_preprocessing(frame)

    left_frame = frame[0:height, 0: middle]
    thresholding_left = thresholding_image[0:height, 0: middle]
    right_frame = frame[0:height, middle:width]
    thresholding_right = thresholding_image[0:height, middle:width]


    #gate = find_rectangle(thresholding_image, frame)
    gate_left = find_rectangle(thresholding_left, left_frame, middle, height)
    gate_right = find_rectangle(thresholding_right, right_frame, middle, height)


    #if gate_left is not None:
    #    left_frame = cv2.rectangle(left_frame, gate_left[1], gate_left[2], (255, 0,0), 2)
    #    cv2.imshow('gate_left', left_frame)
    #    cv2.waitKey(0)

    #if gate_right is not None:
    #    right_frame = cv2.rectangle(right_frame, gate_right[1], gate_right[2], (255, 0,0), 2)
    #    cv2.imshow('gate_right', right_frame)
    #    cv2.waitKey(0)

    
    print("gate_info_left :", gate_left)
    print("gate_infor_right :", gate_right)
    gate = Gate(gate_left, gate_right, middle, height)
    
    check_global_status(gate, middle)

    
    
