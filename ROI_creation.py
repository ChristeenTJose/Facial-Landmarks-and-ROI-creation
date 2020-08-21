import dlib
import cv2
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def ROI_Creator_FL(frame):
	frame1=cv2.flip(frame,1)
	frame2=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	frame3=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	frame4=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	frame5=np.full(shape=(480,640,3),fill_value=(0,0,0),dtype=np.uint8)
	gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
	for face in faces:
		x1,y1,x2,y2 = face.left(),face.top(),face.right(),face.bottom()
		cv2.rectangle(frame1,(x1,y1),(x2,y2),(0,0,255),1)
		landmarks = predictor(gray, face)
		pts=[]#Face
		for i in range(17):#1-17
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		for i in range(26,21,-1):#23-27
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		for i in range(21,17,-1):#18-22
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts=np.array(pts)
		cv2.fillPoly(frame3, [pts], (0,255,0))
		cv2.fillPoly(frame5, [pts], (255,255,255))
		pts=[]#Left Eye
		for i in range(36,42):#37-42
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts=np.array(pts)
		cv2.fillPoly(frame4, [pts], (0,0,255))
		cv2.fillPoly(frame5, [pts], (0,0,0))
		pts=[]#Right Eye
		for i in range(42,48):#43-48
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts=np.array(pts)
		cv2.fillPoly(frame4, [pts], (0,0,255))
		cv2.fillPoly(frame5, [pts], (0,0,0))
		pts=[]#Nose
		for i in range(27,31):#28-31
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		#Left Half of Nose
		pts_left=pts.copy()
		for i in range(33,30,-1):#34-32
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts_left.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts_left=np.array(pts_left)
		cv2.fillPoly(frame4, [pts_left], (0,255,0))
		cv2.fillPoly(frame5, [pts_left], (0,0,0))
		#Right Half of Nose
		pts_right=pts.copy()
		for i in range(33,36):#34-36
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts_right.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts_right=np.array(pts_right)
		cv2.fillPoly(frame4, [pts_right], (0,255,0))
		cv2.fillPoly(frame5, [pts_right], (0,0,0))
		pts=[]#Lips
		for i in range(48,60):#49-60
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts=np.array(pts)
		cv2.fillPoly(frame4, [pts], (255,0,0))
		cv2.polylines(frame5,[pts],True,(0,0,0),1)
		#cv2.fillPoly(frame5, [pts], (0,0,0))
		pts=[]#Inbetween Lips
		for i in range(60,68):#61-68
			x,y = landmarks.part(i).x,landmarks.part(i).y
			pts.append([x,y])
			cv2.circle(frame1,(x,y),2,(0, 0,255),-1)
			cv2.circle(frame2,(x,y),3,(0,0,255),-1)
		pts=np.array(pts)
		cv2.fillPoly(frame4, [pts], (0,255,255))
		cv2.fillPoly(frame5, [pts], (0,0,0))
	return frame1,frame2,frame3,frame4,frame5
