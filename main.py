# encoding: utf-8
from image import Image
from detection import Detection
from particle_filter import ParticleFilter
from polya import bhattacharyya,log_like_polya,logP,fit_betabinom_minka,fit_betabinom_minka_alternating,fit_fixedpoint
import cv2
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
#import pylab as pl


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
                img.show_hist(hist_ref)
                hist_ref=hist_ref/float(hist_ref.sum())
        else: #Aprendizaje finalizado
            #Iniciar filtro, utilizar el ultimo ROI/found como initialState
            # print len(hist_ref)

            if initialState is None:
                x,y,w,h = found[0]
                hist = img.getColorHistogram(found[0]).ravel()
                dx = random.uniform(0,5)
                dy = random.uniform(0,5)
                dw = random.uniform(0,.5)
                dh = random.uniform(0,.5)
                initialState = np.array([x,y,w,h,dx,dy,dw,dh])
                ceroState = np.random.uniform(0,img.size[1],8)
                cov = np.cov(np.asarray([initialState,ceroState]).T)
                pf  = ParticleFilter(initialState,cov,hist_ref,10.,100)
                u=np.zeros((100,4))
                for rect in pf.states:
                    rect = rect[:4]
                    img.draw_roi([rect])
                # pf.update(img)

            else: #ya se ha inicializado el filtro ahora se busca actualizar y predecir
                old=pf.states
                pf.predict(img.size[0],img.size[1],u)
                pf.update(img)
                for rect in pf.states:
                    rect = rect[:4]
                    img.draw_roi([rect])
                u=(old-pf.states)[:,-4:]
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

    gauss_data_f = []
    gauss_data_n = []
    polya_data_f = []
    polya_data_n = []
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
            gauss_data_n.append(G2)
            gauss_data_f.append(G)
            polya_data_n.append(P2)
            polya_data_f.append(P)
        img.show()
        if 0xFF & cv2.waitKey(5) == 27:
            cv2.destroyAllWindows()
            break
    gauss_data_f = np.asarray(gauss_data_f)        
    gauss_data_f = gauss_data_f[~np.isnan(gauss_data_f)]
    gauss_data_n = np.asarray(gauss_data_n)
    gauss_data_n = gauss_data_n[~np.isnan(gauss_data_n)]
    d            =  np.concatenate((gauss_data_f,gauss_data_n))
    v = np.concatenate(( np.ones(len(gauss_data_f)), np.zeros(len(gauss_data_n))))
    
    fpr, tpr, thresholds = roc_curve(v, d)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve : %f" % roc_auc)
    
    polya_data_f = np.asarray(polya_data_f)        
    polya_data_f = polya_data_f[~np.isnan(polya_data_f)] #Remove Nan elements
    polya_data_n = np.asarray(polya_data_n)
    polya_data_n = polya_data_n[~np.isnan(polya_data_n)] #Remove Nan elements
    d            =  np.concatenate((polya_data_f,polya_data_n))
    v = np.concatenate(( np.ones(len(polya_data_f)), np.zeros(len(polya_data_n))))
    fpr2, tpr2, thresholds = roc_curve(v, d)
    roc_auc2 = auc(fpr2, tpr2)
    print("Area under the ROC curve : %f" % roc_auc2)
    
    pl.subplot(212)
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC for Gaussian')
    pl.subplot(211)
    pl.plot(fpr2, tpr2, label='ROC curve (area = %0.2f)' % roc_auc2)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC for Polya')
    pl.show()

    
    

if __name__=='__main__':
    go_particle()
    #compare_distributions()


