import cv2 as cv
import numpy as np
from torch_recog import init_classifier, predict

skip_fact = 25
vid = cv.VideoCapture(0)
init_classifier()

r1_off, r2_off = 200, 100

c1_off, c2_off = 100, 100

first_iter = True
avg = np.zeros((500, 500), np.uint8)

backSub = cv.createBackgroundSubtractorKNN()		
frameCount = 0
dc = 0
while(True):
	dc+=1
	ret, frame = vid.read()

	if first_iter:
		avg = frame.copy()
		# avg = cv.cvtColor(avg, cv.COLOR_BGR2GRAY)
		first_iter = False
	# cv.imshow('Frame', frame)

	fgMask = backSub.apply(frame)

	# cv.imshow('FG Mask', fgMask)
	# print(fgMask.shape)
	img = np.zeros_like(frame)

	img = (fgMask)

	# cv.imshow('onlu', img)
	# img = cv.bitwise_not(fgMask)
	# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

	# img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	img = img[r1_off:-r2_off, c1_off:-c2_off]
	
	
	kernel = (25, 25)
	# blurred = img.copy()
	blurred = cv.GaussianBlur(img, kernel, 0)

	# # _, thresh1 = cv.threshold(blurred, 190, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
	# cv.imshow('bl',blurred)

	# cv.imshow('threhsolded', thresh1)


	# thresh2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
 #                                          cv.THRESH_BINARY, 199, 5)
  
	thresh3 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV, 199, 5)	


	# cv.accumulateWeighted(cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32), avg.astype(np.float32), 0.005 )
	# cv.accumulateWeighted(frame.astype(np.float32), avg.astype(np.float32), 0.005 )

	# avg_result = cv.convertScaleAbs(avg)

	# cv.imshow('adaptive thresh mean',thresh2)
	# cv.imshow('adaptive thresh gaussian', thresh3)

	erosion_kernel = np.ones((5, 5), np.uint8)
	img_erosion = cv.erode(thresh3, erosion_kernel, iterations=1)
	img_ero_dila = cv.dilate(img_erosion, erosion_kernel, iterations=1)

	# cv.imshow('eroded and dilated', img_ero_dila)

	contours, heirarchy = cv.findContours(img_ero_dila.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

	if len(contours) > 0:# and dc%skip_fact == 0:
		copy_frame = frame.copy()
		cnt = max(contours, key=lambda x: cv.contourArea(x))

		x, y, w, h = cv.boundingRect(cnt)
		
		X = x + c1_off #- c2_off
		Y = y + r1_off #- r2_off
		# hull = cv.convexHull(cnt, returnPoints=False)
		# drawing = np.zeros(frame.shape, np.uint8)

		# defects = cv.convexityDefects(cnt, hull)
		count_defects = 0		
		cv.rectangle(frame, (X, Y), (X+w, Y+h), (0, 0, 255), 0)
		print( (X, Y),'->', (X+w, Y+h))
		
		buff = 20
		roi_frame = copy_frame[Y-buff:Y+h+buff, X-buff:X+h+buff]
		cv.imwrite('/home/ayush/Desktop/Stuff/DVCON/handframes/fr'+str(frameCount)+'.jpg', roi_frame)
		pred = predict(roi_frame)
		cv.putText(frame, pred, (X, Y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
		
		if frameCount % skip_fact == 0:
			cv.imwrite('/home/ayush/Desktop/Stuff/DVCON/predictions/fr_'+str(frameCount)+'_'+pred+'.jpg', roi_frame)


		frameCount+=1
	
	cv.imshow('frame', frame)

	# cv.drawContours(img_ero_dila, contours, -1, (0, 255, 0), 3)

	if cv.waitKey(1) & 0xFF == ord('q'):
		break





vid.release()
cv.destroyAllWindows()
