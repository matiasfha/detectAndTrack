import numpy as np
import math
from polya import bhattacharyya,log_like_polya,logP,fit_betabinom_minka,fit_betabinom_minka_alternating,fit_fixedpoint
import cv2


class Image:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        ret,self.image=self.capture.read()
        self.size = (self.image.shape[0],self.image.shape[1])
        self.nbins=8
        self.edges=[np.linspace(0,180,self.nbins+1),np.linspace(0,256,self.nbins+1),np.linspace(0,256,2+1)]
        self.found=[]
        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

    def get(self):
        ret,self.image = self.capture.read()
        if ret:
            self.found = self.cascade.detectMultiScale(self.image, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)


    def getColorHistogram(self,state):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        x,y,h,w=state[0:4]
        if((x<0 or x>self.size[1]) or (y<0 or y>self.size[0])):
            return(np.zeros(self.nbins+1,self.nbins+1,3))
        else:
            hsv_roi = hsv[y:y+w, x:x+h]
            #hist,edges=np.histogramdd(hsv_roi.reshape(-1,3),bins=self.edges)
            #return hist
            return(cv2.calcHist( [hsv_roi], [0,1], None, [self.nbins,self.nbins], [0, 180, 0, 256] ))

    def show(self):
        if len(self.found)>0:
            self.draw_detections(1)
        cv2.imshow('Tracker',self.image)
    
    def __del__(self):
        cv2.destroyAllWindows()

    def show_hist(self,hist,title='hist'):
        hist = hist.reshape(-1)
        #hist=hist/np.float(hist.sum())
        bin_count = hist.shape[0]
        bin_w = 8
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow(title, img)

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
detections=[]
while(True):
    img.get()
    #img.show()
    if(len(img.found)>0):
        hist_ref=img.getColorHistogram(img.found[0]).ravel()
        detections.append(hist_ref)
        img.show_hist(hist_ref)
        hist_ref=hist_ref/float(hist_ref.sum())
        #img.draw_detections(thickness=3)      
    if 0xFF & cv2.waitKey(5) == 27:
        break
print np.asmatrix(detections).shape
alpha_hat,it=fit_betabinom_minka_alternating(detections)

while(True):
    img.get()
    img.show()
    if len(img.found)>0:
        hist=img.getColorHistogram(img.found[0]).ravel()
        img.show_hist(hist)
        hist_norm=hist/float(hist.sum())
        D=bhattacharyya(hist_ref,hist_norm)
        G=np.exp(-20.*D**2)
        P=np.exp(log_like_polya(alpha_hat,np.array([hist])))
        print 'Bhattacharyya={0},Gaussian={1},Polya={2}'.format(D,G,P)
    if 0xFF & cv2.waitKey(5) == 27:
        break
