import numpy as np
import cv2 
from kinect import *

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


def depth_roi(depth_frame,persons):
    roi=[]
    for ri,(x, y, w, h) in enumerate(persons):
        depth=depth_frame[y:y+h, x:x+w]
        print depth
        roi.append(depth)
    return roi

def show_hist(hist):
    bin_count = hist.shape[0]
    bin_w = 24
    img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
    for i in xrange(bin_count):
        h = int(hist[i])
        cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    cv2.imshow('hist', img)

kn=Kinect()
kn.get_depth_stream()
kn.get_color_stream()

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

while True:
    depth_frame=kn.depth_to_cv()
    color_frame=kn.color_to_cv()
    found, w = hog.detectMultiScale(color_frame, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    roi=depth_roi(depth_frame,found_filtered)
    draw_detections(color_frame, found_filtered,3)
    cv2.imshow('persondetect', color_frame)
    if 0xFF & cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()
kn.unload()