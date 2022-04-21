import cv2
import numpy as np
import time
import math
import dlib
from tracker import *



AboveLinePosition = 490
BelowLinePosition = 680
offset = 6

minRectHeight = 70
minRectWidth = 70

maxRectHeight = 190
maxRectWidth = 190

def centroidDetector(x,y,w,h):
    x_ = x + (w//2)
    y_ = y + (h//2)
    return x_, y_

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(abs(math.pow(location2[0] - location1[0], 2)) + abs(math.pow(location2[1] - location1[1], 2)))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed


if __name__ == '__main__':
    #importing cctv video
    capture = cv2.VideoCapture('video.mp4')

    #initialize Substractor   
    algorithm = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=150)
    # algorithm = cv2.createBackgroundSubtractorKNN()

    vehicle = {}
    vehicleLocation1 = {}
    vehicleLocation2 = {}
    initialTime = {}
    finalTime = {}
    currVehId = 0
    speed = [None] * 500
    frameCounter = 0

    while True:
        # initital_time = time.time()


        ret, frame = capture.read()
        frameCounter += 1
        vehicleIDtoDelete = []
        for vehID in vehicle.keys():
            trackingQuality = vehicle[vehID].update(frame)
            print('tracking quality')
            if trackingQuality < 7:
                vehicleIDtoDelete.append(vehID)
        
        for vehID in vehicleIDtoDelete:
            print('Removing vehicle ID ' + str(vehID) + ' from list of trackers.')
            vehicle.pop(vehID,None)
            vehicleLocation1.pop(vehID,None)
            vehicleLocation2.pop(vehID,None)

        if frameCounter % 10 != 0:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grey, (3,3),5)

            image = algorithm.apply(blur)
            dilat = cv2.dilate(image, np.ones((6,6)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            dilateData = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
            dilateData = cv2.morphologyEx(dilateData,cv2.MORPH_CLOSE,kernel)
            counter, h = cv2.findContours(dilateData,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame,(200,AboveLinePosition), (1000, AboveLinePosition),(0,255,255),3)

            cv2.line(frame,(25,BelowLinePosition), (1200, BelowLinePosition),(255,0,255),3)

            matchVehicleID = None
            for cnt in counter:
                # Calculate area and remove small elements
                (x, y, w, h) = cv2.boundingRect(cnt)
                if not ((maxRectHeight >= h >= minRectHeight) and (maxRectHeight >= w >= minRectWidth)):
                    continue
                    #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                center = centroidDetector(x,y,w,h)
                # vehicle.append(center)   # yanhan problem hai
                cv2.circle(frame,center,4,(0,0,255),-2)

                for vehID in vehicle.keys():
                    currentPosition = vehicle[vehID].get_position()

                    _x = int(currentPosition.left())
                    _y = int(currentPosition.top())
                    _w = int(currentPosition.width())
                    _h = int(currentPosition.height())

                    x_corr = _x + (_w // 2)
                    y_corr = _y + (_h // 2)
                    if ((_x <= center[0] <= _x + _w) and (_y <= center[1] <= _y + _h) and (x <= x_corr <= (x + w)) and (y <= y_corr <= (y + h))):
                        matchVehicleID = vehID

                
                if matchVehicleID is None:
                    print('Creating new Tracker ' + str(currVehId))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image,dlib.rectangle(x,y,x+w,y+h))
                    vehicle[currVehId] = tracker

                    currVehId += 1
        
        for vehID in vehicle.keys():
            currentPosition = vehicle[vehID].get_position()

            _x = int(currentPosition.left())
            _y = int(currentPosition.top())
            _w = int(currentPosition.width())
            _h = int(currentPosition.height())

            if _y < AboveLinePosition + offset and _y > AboveLinePosition - offset:
                vehicleLocation1[vehID] = [_x,_y,_w,_h]
                initialTime[vehID] = time.time

            if _y < BelowLinePosition + offset and _y > BelowLinePosition - offset:
                vehicleLocation2[vehID] = [_x,_y,_w,_h]
                finalTime[vehID] = time.time
        
        for i in vehicleLocation1.keys():	
            if frameCounter % 14 == 0:
                [x1, y1, w1, h1] = vehicleLocation1[i]
                [x2, y2, w2, h2] = vehicleLocation2[i]

                # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])
                vehicleLocation1[i] = [x2, y2, w2, h2]
                # fps = 1
                # if initialTime[i] != finalTime[i]:
                #     fps = 1.0 / abs(initialTime[i] - finalTime[i])

                # print 'new previous location: ' + str(carLocation1[i])
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    #if y1 > 275 and y1 < 285:
                    if speed[i] != None:
                        cv2.putText(frame, str(int(speed[i])) + " km/hr", (int(x1 + w1/2), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                    
                    #print ('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')

                    #else:
                    #	cv2.putText(resultImage, "Far Object", (int(x1 + w1/2), int(y1)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # cv2.imshow('Detector', dilateData)
        cv2.imshow('Orignal Video', frame)
        if cv2.waitKey(33) == 13:
            break
    cv2.destroyAllWindows()
    capture.release()
