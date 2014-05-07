# encoding: utf-8
import cv2
import cv2.cv as cv
import numpy as np
import os
class Image:
    def __init__(self,video=None):

        if video is not None:
            if os.path.isdir(video): #Es un conjunto de imagenes en un directorio
                self.device = None
                self.images = []
                for dir_entry in os.listdir(video):
                    dir_entry_path = os.path.join(video,dir_entry)
                    if os.path.isfile(dir_entry_path):
                        self.images.append(dir_entry_path)
                self.index = 0
            else:
                self.device = cv2.VideoCapture(video)
        else:
            self.device = cv2.VideoCapture(0)
        self.image = self._readImage()
        if self.device is not None:
            self.size = (self.device.get(cv.CV_CAP_PROP_FRAME_WIDTH),self.device.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        else:
            height, width, depth = self.image.shape
            self.size = (width,height)
        self.nbins=8
        self.edges=[np.linspace(0,180,self.nbins+1),np.linspace(0,256,self.nbins+1),np.linspace(0,256,2+1)]

    def get(self):
        self.image = self._readImage()
        return self.image

    def _readImage(self):
        if self.device is not None:
            ret,image = self.device.read()
        else:
            image = cv2.imread(self.images[self.index])
            self.index = self.index + 1
            #Leer la siguiente imagen desde el directorio
        return image

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
            channels = cv2.split(hsv_roi)
            cv2.equalizeHist(channels[0],channels[0])
            cv2.equalizeHist(channels[1],channels[1])
            cv2.equalizeHist(channels[2],channels[2])
            cv2.merge(channels,hsv_roi)
            # hist,edges=np.histogramdd(hsv_roi.reshape(-1,3),bins=self.edges)
            # return hist
            #cv2.calcHist([hsv_roi],[2],None,[10],[0,180,0,256])
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
        if self.device is not None:
            self.device.release


