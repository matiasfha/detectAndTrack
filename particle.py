import numpy as np
import math
from polya import bhattacharyya
from kinect import Kinect
import cv2


class Image:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.image=self.capture.read()
        self.size = (self.image.shape[1],self.image.shape[0])
        self.nbins=16
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
        self.found=[]

    def get(self):
        self.image = self.kn.color_to_cv()
        self.found, w = self.hog.detectMultiScale(self.image, winStride=(8,8), padding=(32,32), scale=1.05)
    

    def getColorHistogram(self,state):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        x,y,h,w=state[0:4]
        if((x<0 or x>self.size[0]) or (y<0 or y>self.size[1])):
            return(np.zeros(self.nbins))
        else:
            hsv_roi = hsv[y:y+w, x:x+h]
            return(cv2.calcHist( [hsv_roi], [0], None, [self.nbins], [0, 180] ))

    def show(self):
        if len(self.found)>0:
            self.draw_detections(1)
        cv2.imshow('Tracker',self.image)
    
    def __del__(self):
        self.kn.unload()
        cv2.destroyAllWindows()

    def show_hist(self,hist):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def draw_detections(self,thickness = 1):
        for x, y, w, h in self.found:
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(self.image, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


class ParticleFilter:

    def __init__(self, x0, P0, hist, sigma2, num):
        """ x_{k+1} = A*x_{k}+B*u_k + v_k
        y_k = KDE(x_k) + e_k 
        v_k ~ N(0,Q)
        e_k ~ exp(-D/sigma2)
        x0 = x_0, P0 = P_0
        D= Bhattacharyya distance between current color histogram and reference histogram
        P is the variance of the estimate
        """
        self.A = np.matrix([[1,0,0,0,1,0,0,0],
                            [0,1,0,0,0,1,0,0],
                            [0,0,1,0,0,0,1,0],
                            [0,0,0,1,0,0,0,1],
                            [0,0,0,0,1,0,0,1],
                            [0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,1]])
        self.dim=8
        self.num=num
        self.hist_ref=hist_ref        
        self.Q = np.eye(self.dim)      # Measurement noise covariance
        self.sigma2 = sigma2      # Process noise covariance
        self.x0=x0
        self.P0 = P0     # Estimated covariance
        self.states = np.random.multivariate_normal(self.x0,self.P0,self.num) # Estimated state
        self.weights=np.ones(num)/np.float(num)
        self.threshold=.5

        
    def predict(self):
        for i in range(self.num):
            self.states[i]=np.dot(self.A,self.states[i])+np.random.multivariate_normal(np.zeros(self.dim),self.Q)

    def _calculate_weights(self,img):
        for i in range(self.num):
            observed_hists=img.getColorHistogram(self.states[i])
            D=bhattacharyya(observed_hists,self.hist_ref)
            self.weights[i]=self.weights[i]*np.exp(-D/self.sigma2)
        self.weights=self.weights[i]/np.float(self.weights.sum())

    def _resample(self):
        indices=np.random.choice(np.arange(0,self.num),size=self.num,replace=True,p=self.weights)
        self.weights=np.ones(num)/np.float(num)
        self.states=self.states[indices]

    def update(self,img):
        self._calculate_weights(img)
        if (1/np.float((self.weights**2).sum()) < self.threshold):
            self._resample()



img = Image()

'''while(len(img.found)==0):
    img.get()
    img.show()
'''
while(True):
    img.get()
    img.show()
    if len(img.found)>0:
        hist=img.getColorHistogram(img.found[0])
        img.show_hist(hist)
    if 0xFF & cv2.waitKey(5) == 27:
        break
