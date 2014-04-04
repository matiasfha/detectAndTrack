# encoding: utf-8
from image import Image
from detection import Detection
from particle_filter import ParticleFilter

if __name__=='__main__':
    img     = Image()
    detect  = Detection()
    pf      = ParticleFilter()
    hist_ref= None
    found   = None
    while(True):
        img.get()
        #Detect faces
        found = detect.faces(img.image)
        if len(found) > 0:
            # Set reference histogram from roi
            # roi is the first face detected assuming one face in the portview
            hist_ref = img.getColorHistogram(found[0])