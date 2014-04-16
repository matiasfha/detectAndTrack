# encoding: utf-8
import cv2
import cv2.cv as cv
import numpy as np
class Image:
    def __init__(self):
        self.device = cv2.VideoCapture(0)
        ret,self.image = self.device.read()
        self.size = (self.device.get(cv.CV_CAP_PROP_FRAME_WIDTH),self.device.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        self.nbins=8
        self.edges=[np.linspace(0,180,self.nbins+1),np.linspace(0,256,self.nbins+1),np.linspace(0,256,2+1)]

    def get(self):
        ret,self.image = self.device.read()

    # def size():
    #     return self.size

    def getColorHistogram(self,roi=None):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        if roi is not None:
            if len(roi) > 4:
                roi = roi[:4]
            x = int(roi[0])
            y = int(roi[1])
            w = int(roi[2])
            h = int(roi[3])
        else:
            #Histogram to all frame
            x,y,h,w= (0,0,self.size[0],self.size[1])
        if((x<0 or x>self.size[1]) or (y<0 or y>self.size[0])):
            # return(np.zeros(self.nbins+1,self.nbins+1,3))
            return np.zeros((64,)).reshape((64,))
        else:
            hsv_roi = hsv[y:y+h, x:x+w]
            # hist,edges=np.histogramdd(hsv_roi.reshape(-1,3),bins=self.edges)
            # return hist
            return(cv2.calcHist( [hsv_roi], [0,1], None, [self.nbins,self.nbins], [0, 180, 0, 256] ))


    def show(self,window_name = 'Image'):
        cv2.imshow(window_name,self.image)

    def show_hist(self,hist,window_name='hist'):
        hist = hist.reshape(-1)
        bin_count = hist.shape[0]
        bin_w = 8
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow(window_name, img)

    def draw_roi(self,roi):
        for x,y,w,h in roi:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(self.image, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), 2)

    def __del__(self):
        cv2.destroyAllWindows()
        self.device.release


