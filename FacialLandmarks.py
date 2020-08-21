from ROI_creation import ROI_Creator_FL
import cv2
import numpy as np
vc = cv2.VideoCapture(0)
while cv2.waitKey(1)==-1:
	_,frame = vc.read()
	frame1,frame2,frame3,frame4,frame5 = ROI_Creator_FL(frame)
	frame6=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	display=np.vstack((np.hstack((frame1,frame2,frame3)),np.hstack((frame4,frame5,frame6))))
	display=cv2.resize(display,(1280,720))
	cv2.imshow("Frame",display)
vc.release()
cv2.destroyAllWindows()
