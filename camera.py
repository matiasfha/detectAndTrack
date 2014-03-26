# encoding: utf-8
import cv2
import cv2.cv as cv
import numpy as np

import random
import numpy as np
from scipy import stats

class Image:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        # cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_WIDTH,512)
        # cv.SetCaptureProperty(self.capture,cv.CV_CAP_PROP_FRAME_HEIGHT,512)
        ret,self.image = self.capture.read()
        cv2.imshow("Capture",self.image)
        self.size = (self.image.shape[0],self.image.shape[1])

    def create(self):
        self.image = self.capture.read()

    def getCol(self,sv):
        x = sv[0]
        y = sv[1]
        if((x<0 or x>self.size[0]) or (y<0 or y>self.size[1])):
            return((0,0,0,0))
        else:
            return(cv.Get2D(self.image,int(sv[1]),int(sv[0])))

class SystemModel:
    def __init__(self,model):
        self.model = model
    def generate(self,sv,w):
        return(self.model(sv,w))

class Likelihood:
    def __init__(self,model):
        self.model = model

    def generate(self,sv,mv):
        return(self.model(sv,mv))

    def normalization(self,svs,mv):
        return(sum([self.generate(sv,mv) for sv in svs]))

def model_s(sv,w):
    F = np.matrix([[1,0,1,0],
                   [0,1,0,1],
                   [0,0,1,0],
                   [0,0,0,1]])
    return(np.array(np.dot(F,sv))[0]+w)

def model_l(sv,mv):
    mv_col = img.getCol(sv)
    mv_col = mv_col[0:3]
    target_col = (150,90,40)

    delta = np.array(mv_col)-np.array(target_col)
    dist_sqr = sum(delta*delta)
    sigma = 10000.0
    gauss = np.exp(-0.5*dist_sqr/(sigma*sigma)) / (np.sqrt(2*np.pi)*sigma)
    return(gauss)

def resampling(svs,weights):
    N = len(svs)
    sorted_particle = sorted([list(x) for x in zip(svs,weights)],key=lambda x:x[1],reverse=True)
    resampled_particle = []
    while(len(resampled_particle)<N):
        for sp in sorted_particle:
            resampled_particle += [sp[0]]*(sp[1]*N)
    resampled_particle = resampled_particle[0:N]

    return(resampled_particle)

def filtration(svs,mv,systemModel,likelihood):
    dim = len(svs[1])
    N = len(svs)
    sigma = 2.0
    rnorm = stats.norm.rvs(0,sigma,size=N*dim)
    ranges = zip([N*i for i in range(dim)],[N*i for i in (range(dim+1)[1:])])
    ws = np.array([rnorm[p:q] for p,q in ranges])
    ws = ws.transpose()

    svs_predict = [systemModel.generate(sv,w) for sv,w in zip(svs,ws)]

    normalization_factor = likelihood.normalization(svs_predict,mv)
    likelihood_weights = [likelihood.generate(sv,mv)/normalization_factor for sv in svs_predict]
    svs_resampled = resampling(svs_predict,likelihood_weights)
    return(svs_resampled)

def initStateVectors(imageSize,sampleSize):
    xs = [random.uniform(0,imageSize[0]) for i in range(sampleSize)]
    ys = [random.uniform(0,imageSize[1]) for i in range(sampleSize)]
    vxs = [random.uniform(0,5) for i in range(sampleSize)]
    vys = [random.uniform(0,5) for i in range(sampleSize)]

    return([list(s) for s in zip(xs,ys,vxs,vys)])

def showImage(svs,img):
    for sv in svs:
        # cv.Circle(dst,(int(sv[0]),int(sv[1])),3,cv.CV_RGB(0,0,255))
        cv2.circle(img.image,(int(sv[0]),int(sv[1])),3,cv.CV_RGB(0,0,255))
    cv2.flip(img.image,1)
    cv2.imshow('Capture',img.image)
    #cv2.WriteFrame(vw,dst)


if(__name__=="__main__"):
    img = Image()
    #vw = cv2.VideoWriter('cap.avi', cv.CV_FOURCC(*'DIB '), 30.0, (512, 512), False)
    sampleSize = 100
    systemModel = SystemModel(model_s)
    likelihood = Likelihood(model_l)
    svs = initStateVectors(img.size,sampleSize)

    while(True):
        showImage(svs,img)
        img.create()
        svs = filtration(svs,img,systemModel,likelihood)



