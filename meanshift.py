import cv2
import cv2.cv as cv
import numpy as np

class Meanshift:

	def __init__(self):
		self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

	def update_window(self,image,bb):
		try:
			x,y,h,w = bb
			r,h,c,w = bb
			# roi = image[y:y+63, x:h+63]
			roi = image[r:r+h, c:c+w]
			# cv2.imshow("ROI",roi)
			hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
			roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
			cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
			hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
			dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
			ret,bb = cv2.meanShift(dst, bb, self.term_crit)

		except:
			pass
		return bb
