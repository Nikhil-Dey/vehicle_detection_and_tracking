import cv2
from tracker import *

above_line_position = 520
below_line_position = 600


def centroid(x,y,w,h):
    x1 = w//2
    y1 = h//2
    cx = x + x1
    cy = y + y1
    return cx, cy

def estimateSpeed(location1, location2):
	d_pixels = math.sqrt(abs(math.pow(location2[0] - location1[0], 2)) + abs(math.pow(location2[1] - location1[1], 2)))
	# ppm = location2[2] / carWidht
	ppm = 8.8
	d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	fps = 18
	speed = d_meters * fps * 3.6
	return speed

detect = []
offset = 6
counter = 0

location1 = {}
location2 = {}



# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("video.mp4")

# Object detection from Stable camera
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80)
object_detector = cv2.createBackgroundSubtractorKNN()

speed = [None] * 500
while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape

    # Extract Region of interest

    # 1. Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []


    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if area > 6000:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h])

    cv2.line(frame, (180, above_line_position), (1100, above_line_position), (255,0,255),3)
    cv2.line(frame, (25, below_line_position), (1200, below_line_position), (255,255,0),3)
    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        center = centroid(x,y,w,h)
        detect.append(center)
        cv2.circle(frame, center,4,(0,0,255), -1)

        # for (i,j) in detect:
        #     # print(i,j,sep=" ")
        #     if j<(count_line_position + offset) and j>(count_line_position - offset):
        #         counter += 1
        #         cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255,0,255),3)
        #         detect.remove((i,j))
        #         print("vehicle counter:" + str(counter))
        for (i,j) in detect:
            if j<(above_line_position + offset) and j>(above_line_position - offset):
                location1[id] = (i,j)
                cv2.line(frame, (180, above_line_position), (1100, above_line_position), (127,0,255),3)


            if j<(below_line_position + offset) and j>(below_line_position - offset):
                location2[id] = (i,j)
                counter += 1
                cv2.line(frame, (25, below_line_position), (1200, below_line_position), (255,0,127),3)
                print("vehicle counter:" + str(counter))
                detect.remove((i,j))


    for key in set(location1) & set(location2):
        [x1, y1] = location1[key]
        [x2, y2] = location2[key]

        if [x1, y1] != [x2, y2]:
            if (speed[key] == None or speed[key] == 0) and y1 >= 275 and y1 <= 285:
                speed[key] = estimateSpeed([x1, y1], [x2, y2])

            #if y1 > 275 and y1 < 285:
            if speed[key] != None:
                cv2.putText(frame, str(int(speed[key])) + " km/hr", (int(x1 + 10), int(y1-5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
    cv2.putText(frame, "VEHICLE COUNTER :" + str(counter), (550,70), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    # cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 13:
        break

cap.release()
cv2.destroyAllWindows()