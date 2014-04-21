# encoding: utf-8
import numpy as np
from polya import bhattacharyya
import cv2

class ParticleFilter:
   # sigma 1/20
   #
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
        self.hist_ref=hist
        self.Q = np.eye(self.dim)      # Measurement noise covariance
        self.sigma2 = sigma2      # Process noise covariance
        self.x0=x0
        self.P0 = P0     # Estimated covariance
        self.states = np.zeros((self.num,self.dim))
        for i in range(self.num): 
            self.states[i]=self.x0+np.concatenate([np.random.normal(0,10,2),np.random.normal(0,1,2),np.random.normal(0,1,4)])
        
        self.weights=np.ones(num)/np.float(num)
        self.threshold=.7


    def predict(self,max_width,max_height,control):
        for i in range(self.num):
            self.states[i]=np.concatenate([self.states[i][:4],control[i,:]])
            noise=np.concatenate([np.random.normal(0,5,2),np.random.normal(0,.1,2),np.random.normal(0,.1,4)])
            val = np.dot(self.A,self.states[i])+noise
            self.states[i]= val
            
    def _calculate_weights(self,img):
        for i in range(self.num):
            observed_hists=img.getColorHistogram(self.states[i]).ravel()
            #D=bhattacharyya(observed_hists,self.hist_ref) # Cambia por Polya
            D=cv2.compareHist(observed_hists,self.hist_ref,cv2.cv.CV_COMP_BHATTACHARYYA)
            self.weights[i]=(1/np.sqrt(2*self.sigma2))*np.exp(-D**2/(2*self.sigma2))
        self.weights=self.weights/np.float(self.weights.sum()) #Normalizacion
        # print self.weights



    def _resample(self):
        indices=np.random.choice(np.arange(0,self.num),size=self.num,replace=True,p=self.weights)
        self.weights=np.ones(self.num)/np.float(self.num)
        self.states=self.states[indices]


    def update(self,img):
        self._calculate_weights(img)
        ess=1/np.float((self.weights**2).sum())
        print("ESS : {0}" .format(ess) )
        if (ess < self.threshold*self.num):
            print("Resample")
            self._resample()
