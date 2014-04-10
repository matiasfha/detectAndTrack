# encoding: utf-8
from image import Image
from detection import Detection
from particle_filter import ParticleFilter
from polya import bhattacharyya,log_like_polya,logP,fit_betabinom_minka,fit_betabinom_minka_alternating,fit_fixedpoint
import cv2
import numpy as np
import random


def go_particle():
    img     = Image()
    detect  = Detection()
    found = []
    initialState = None
    pf = None
    histograms = []
    detections = []
    hist_ref = None
    #Recoleccion
    while(True):
        img.get()

        #Face de recoleccion de información/ Aprendizaje
        if len(histograms) < 10:
            print len(histograms)
            #Detect faces
            found = detect.faces(img.image)
            if len(found) > 0:
                hist_ref = img.getColorHistogram(found[0]).ravel()
                histograms.append(hist_ref)
                detections.append(found)
                img.show_hist(hist_ref)
                hist_ref=hist_ref/float(hist_ref.sum())
        else: #Aprendizaje finalizado
            #Iniciar filtro, utilizar el ultimo ROI/found como initialState
            if initialState is None:
                x,y,w,h = found[0]
                hist = img.getColorHistogram(found[0]).ravel()
                dx = [random.uniform(0,img.size[0]) for i in range(8)]
                dy = [random.uniform(0,img.size[1]) for i in range(8)]
                dw = [random.uniform(0,w) for i in range(8)]
                dh = [random.uniform(0,h) for i in range(8)]
                print np.mean(np.asarray(histograms))
                initialState = [list(s) for s in zip([x],[y],[w],[h],dx,dy,dw,dh)][0]
                ceroState = np.random.uniform(0,img.size[1],8)
                cov = np.cov(np.array([initialState,ceroState]).T)
                pf  = ParticleFilter(initialState,cov,hist_ref,1./-20,10)
            else: #ya se ha inicializado el filtro ahora se busca actualizar y predecir
                pf.predict()
                for rect in pf.states:
                    rect = rect[:4]
                    img.draw_roi([rect])
                    print rect
                print '#################'
                pf.update(img)
        img.show()

        if 0xFF & cv2.waitKey(5) == 27:
            break



    #             pf = ParticleFilter(initialState,cov,hist,0.9 ** 2,100) #num depende de la maquina
    #             # pf.predict()
    #             # pf.update(img)
    #             #pf.states -> dibujar
    #         else:
    #             pf.predict()

    #             for rect in pf.states.tolist():
    #                 img.draw_roi( [rect[:4]] )

    #             pf.update(img)
    #         img.show()
    #         if 0xFF & cv2.waitKey(5) == 27:
    #             break
    #
def compare_distributions():
    img     = Image()
    detect  = Detection()
    histograms = []
    detections = []
    while(True):
        img.get()
        #Detect faces
        found = detect.faces(img.image)
        if len(found) > 0:
            # Set reference histogram from roi
            # roi is the first face detected assuming one face in the portview
            hist_ref = img.getColorHistogram(found[0]).ravel()
            histograms.append(hist_ref)
            detections.append(found)
            img.show_hist(hist_ref)
            hist_ref=hist_ref/float(hist_ref.sum())
        if 0xFF & cv2.waitKey(5) == 27:
            break
    print np.asmatrix(histograms).shape
    alpha_hat,it= fit_betabinom_minka_alternating(histograms)

    while(True):
        img.get()
        found = detect.faces(img.image)
        if len(found) > 0:
            img.draw_roi([found[0]])
            hist=img.getColorHistogram(found[0]).ravel()
            img.show_hist(hist)
            hist_norm=hist/float(hist.sum())
            #For face
            D=bhattacharyya(hist_ref,hist_norm)
            G=np.exp(-20.*D**2)
            P=np.exp(log_like_polya(alpha_hat,np.array([hist])))

            #FOr not face

            x = 0#random.sample([random.uniform(0,img.size[0]) for i in range(100)],1)[0]
            y = 0#random.sample([random.uniform(0,img.size[1]) for i in range(100)],1)[0]
            w = found[0][2]#random.sample([random.uniform(0,200) for i in range(100)],1)[0]
            h = found[0][3]#random.sample([random.uniform(0,200) for i in range(100)],1)[0]
            rect = (x,y,w,h)

            hist = img.getColorHistogram(rect).ravel()
            img.show_hist(hist,'NO FACE')
            hist_norm=hist/float(hist.sum())
            D2=bhattacharyya(hist_ref,hist_norm)
            G2=np.exp(-20.*D2**2)
            P2=np.exp(log_like_polya(alpha_hat,np.array([hist])))
            print 'B={0}, B={1} | G={2}, G={3} | P={4}, P={5}'.format(D,D2,G,G2,P,P2)

        img.show()
        if 0xFF & cv2.waitKey(5) == 27:
            break

if __name__=='__main__':
    go_particle()


