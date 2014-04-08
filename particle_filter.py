# encoding: utf-8
import numpy as np
from polya import bhattacharyya
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
        self.hist_ref=hist
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
            try:
                observed_hists=img.getColorHistogram(self.states[i])
                D=bhattacharyya(observed_hists,self.hist_ref)
                self.weights[i]=self.weights[i]*np.exp(-D/self.sigma2)
                self.weights=self.weights[i]/np.float(self.weights.sum())
            except:
                pass


    def _resample(self):
        indices=np.random.choice(np.arange(0,self.num),size=self.num,replace=True,p=self.weights)
        self.weights=np.ones(num)/np.float(num)
        self.states=self.states[indices]

    def update(self,img):
        self._calculate_weights(img)
        if (1/np.float((self.weights**2).sum()) < self.threshold):
            self._resample()
