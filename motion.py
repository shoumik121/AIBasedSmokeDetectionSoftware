import cv2
import numpy as np
import time
import datetime
import imutils

def detect_motion():
  video_capture = cv2.VideoCapture(0)
  #video_capture = cv2.VideoCapture("demo.mp4")
  time.sleep(2)

  frame1st = None

  while True:
    ret,frame = video_capture.read()
    if not ret:
      continue
    text = 'movement not detected'

    greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_frame = cv2.GaussianBlur(greyscale_frame, (21,21),0)
    
    blur_frame = cv2.blur(gaussian_frame, (5,5))

    if frame1st is None:
      frame1st = greyscale_frame
    else:
      pass
    
    frame = imutils.resize (frame, width = 500)

    frame_d = cv2.absdiff(frame1st, greyscale_frame)
    thresh = cv2.threshold(frame_d, 100, 230, cv2.THRESH_BINARY)[1]

    dilate_image = cv2.dilate(thresh, None, iterations = 2)

    c,hierarchy = cv2.findContours(dilate_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i in c:    	
    	if cv2.contourArea(1) > 800:
    		(x,y,w,h) = cv2.boundingRect(i)    		
    		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    		text = 'motion detected'
    	else:
    		pass

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, f'Video frame status: {text}', (10,20), font, 1, (0,0,255), 2)

    cv2.putText(frame, datetime.datetime.now().strftime (' %A %d %B %Y %I:%M:%S%p'), (10, frame.shape[0] - 10), font, 1, (0,0,255), 3)

    cv2.imshow('normal frame feed:',frame)
    cv2.imshow('image after thresholding & foreground masking:',dilate_image)
    cv2.imshow('Delta framing:',frame_d)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      cv2.destroyAllWindows()
      break

if __name__ == '__main__':
  detect_motion()
  