# encoding: utf-8
# oni_ocv_sample.py
# 2013-9-13
# Eiichiro Momma

import cv2
import cv2.cv as cv
import sys
import math

from kinect import *
from meanshift import *

class Tracker:
    def __init__(self,oni):
        self.mouse_p1   = None
        self.mouse_p2   = None
        self.mouse_drag = False
        self.kinect     = Kinect(oni)
        self.kinect.get_depth_stream()
        self.kinect.get_color_stream()
        self.color_array= None
        self.depth_array= None
        self.bb         = None
        self.meanshift  = Meanshift()
        self.start()

    def start(self):
        self.color_array = self.kinect.color_to_cv()
        self.color_array = cv2.cvtColor(self.color_array,cv2.COLOR_BGR2RGB)
        cv2.imshow('Img',self.color_array)
        cv2.setMouseCallback('Img',self.__mouseHandler)
        # if not self.bb:
        #     self.color_array = self.kinect.color_to_cv()
        #     self.color_array = cv2.cvtColor(self.color_array,cv2.COLOR_BGR2RGB)
        #     cv2.imshow('Img',self.color_array)
        #     cv2.waitKey(30)
        cv2.waitKey(0)

    def __mouseHandler(self,event,x,y,flags,params):
        if event == cv.CV_EVENT_LBUTTONDOWN and not self.mouse_drag:
            self.mouse_p1 = (x,y)
            self.mouse_drag = True
        elif event == cv.CV_EVENT_MOUSEMOVE and self.mouse_drag:
            self.mouse_p2 = (x,y)
            # cv2.rectangle(self.color_array,self.mouse_p1,(x,y),(0,0,255),2)
        elif event == cv.CV_EVENT_LBUTTONUP and self.mouse_drag:
            self.mouse_p2 = (x,y)
            self.mouse_drag = False
            cv2.rectangle(self.color_array,self.mouse_p1,(x,y),(0,0,255),2)
        cv2.imshow('Img',self.color_array)
        cv2.waitKey(30)
        if self.mouse_p1 and self.mouse_p2 and not self.mouse_drag:
            cv2.destroyWindow('Img')
            self.bb = (self.mouse_p1[0],self.mouse_p1[1],self.mouse_p2[0],self.mouse_p2[1])
            self.kalman = cv.CreateKalman(4, 2, 0)
            self.kalman_state = cv.CreateMat(4, 1, cv.CV_32FC1)
            self.kalman_process_noise = cv.CreateMat(4, 1, cv.CV_32FC1)
            self.kalman_measurement = cv.CreateMat(2, 1, cv.CV_32FC1)
            # set previous state for prediction
            self.kalman.state_pre[0,0]  = self.mouse_p1[0]
            self.kalman.state_pre[1,0]  = self.mouse_p2[1]
            self.kalman.state_pre[2,0]  = 0
            self.kalman.state_pre[3,0]  = 0
            # set kalman transition matrix
            self.kalman.transition_matrix[0,0] = 1
            self.kalman.transition_matrix[0,1] = 0
            self.kalman.transition_matrix[0,2] = 0
            self.kalman.transition_matrix[0,3] = 0
            self.kalman.transition_matrix[1,0] = 0
            self.kalman.transition_matrix[1,1] = 1
            self.kalman.transition_matrix[1,2] = 0
            self.kalman.transition_matrix[1,3] = 0
            self.kalman.transition_matrix[2,0] = 0
            self.kalman.transition_matrix[2,1] = 0
            self.kalman.transition_matrix[2,2] = 0
            self.kalman.transition_matrix[2,3] = 1
            self.kalman.transition_matrix[3,0] = 0
            self.kalman.transition_matrix[3,1] = 0
            self.kalman.transition_matrix[3,2] = 0
            self.kalman.transition_matrix[3,3] = 1
            # set Kalman Filter
            cv.SetIdentity(self.kalman.measurement_matrix, cv.RealScalar(1))
            cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(1e-5))
            cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(1e-1))
            cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(1))
            self.track()


    def track(self):
        while True:
            self.depth_array = self.kinect.depth_to_cv()
            self.color_array = self.kinect.color_to_cv()
            self.color_array = cv2.cvtColor(self.color_array,cv2.COLOR_BGR2RGB)

            if self.bb is not None:
                x,y,w,h = self.bb
                print "%s,%s,%s,%s" % (x,y,w,h)
                # cv2.rectangle(self.color_array,(x,y),(x+w,y+h),(0,0,255),2)

                cv2.circle(self.color_array,(x,y),63,(0,0,255),2)
                self.meanshift.update_window(self.color_array,self.bb)
                self.kalman_prediction = cv.KalmanPredict(self.kalman)
                self.kalman_estimated = cv.KalmanCorrect(self.kalman, self.kalman_measurement)
                state_pt = (self.kalman_estimated[0,0], self.kalman_estimated[1,0])
                print "%s,%s,%s,%s" % (int(math.floor(state_pt[0])),int(math.floor(state_pt[1])),w,h)
                self.bb = (int(math.floor(state_pt[0])),int(math.floor(state_pt[1])),w,h)
                self.kalman_measurement[0, 0] = state_pt[0]
                self.kalman_measurement[1, 0] = state_pt[1]

            print "Nuevo [%s]:" % ','.join(map(str,self.bb))
            cv2.imshow('Color',self.color_array)
            # cv2.imshow('Depth',self.depth_array)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                  break
        self.kinect.unload()
        cv2.destroyAllWindows()
oni = sys.argv[1] if len(sys.argv) == 2 else None
track = Tracker(oni)
# track.start()
# track.track()





