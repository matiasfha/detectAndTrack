# encoding: utf-8
from image import Image
from detection import Detection
from particle_filter import ParticleFilter
from polya import bhattacharyya,log_like_polya,logP,fit_betabinom_minka,fit_betabinom_minka_alternating,fit_fixedpoint
import cv2
import numpy as np
import random
if __name__=='__main__':
    img     = Image()
    detect  = Detection()
    # pf      = ParticleFilter()
    histograms = []
    detections = []
    # while(True):
    #     img.get()
    #     #Detect faces
    #     found = detect.faces(img.image)
    #     if len(found) > 0:
    #         # Set reference histogram from roi
    #         # roi is the first face detected assuming one face in the portview
    #         hist_ref = img.getColorHistogram(found[0]).ravel()
    #         histograms.append(hist_ref)
    #         detections.append(found)
    #         img.show_hist(hist_ref)
    #         hist_ref=hist_ref/float(hist_ref.sum())
    #     if 0xFF & cv2.waitKey(5) == 27:
    #         break
    # np.asmatrix(histograms).shape
    # alpha_hat,it= fit_betabinom_minka_alternating(histograms)
    found = None
    initialState = None
    pf = None
    while(True):
        img.get()
        img.show()
        if found is None:
            found = detect.faces(img.image)
        else:
            #Use the found to create initialState if initialState is None
            if initialState is None:
                x,y,w,h = found[0]
                hist = img.getColorHistogram(found[0]).ravel()
                dx = [random.uniform(0,img.size[0]) for i in range(100)]
                dy = [random.uniform(0,img.size[1]) for i in range(100)]
                dw = [random.uniform(0,5) for i in range(100)]
                dh = [random.uniform(0,5) for i in range(100)]
                initialState = [list(s) for s in zip([x],[y],[w],[h],random.sample(dx,1),random.sample(dy,1),random.sample(dw,1),random.sample(dh,1))][0]
                secondState = [list(s) for s in zip([x],[y],[w],[h],random.sample(dx,1),random.sample(dy,1),random.sample(dw,1),random.sample(dh,1))][0]
                x = np.array([initialState,secondState]).T
                cov = np.cov(x)
                pf = ParticleFilter(initialState,cov,hist,0.9 ** 2,10)
                pf.predict()
                pf.update(img)
            else:
                pf.predict()
                pf.update(img)




        # if len(found) > 0:
        #     hist=img.getColorHistogram(found[0]).ravel()
        #     img.show_hist(hist)
        #     hist_norm=hist/float(hist.sum())
        #     D=bhattacharyya(hist_ref,hist_norm)
        #     G=np.exp(-20.*D**2)
        #     P=np.exp(log_like_polya(alpha_hat,np.array([hist])))
        #     print 'Bhattacharyya={0},Gaussian={1},Polya={2}'.format(D,G,P)

        if 0xFF & cv2.waitKey(5) == 27:
            break


