import cv2
import os
import HandTracker as ht
import numpy as np
from utils import only, isInRange

header_folder_path = "Assets/Headers"
header_files = os.listdir(header_folder_path)

headers = [cv2.resize(cv2.imread(f'{header_folder_path}/{header_file}'), (1920, 125))
           for header_file in header_files if header_file.startswith("Header") and header_file.endswith(".png")]

current_header = headers[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

# Variables for drawing -> This variables can be changed by the user during the program execution
current_color = None
current_thickness = 15
brush_thickness = 15
eraser_thickness = 50

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tracker = ht.HandDetector(detectionCon=0.6)
canvas = np.zeros((1080, 1920, 3), np.uint8)

# previous positions saved for drawing lines
xp, yp = None, None

frame_count = 0
process_interval = 2 # Process every 2nd frame

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    if frame_count % process_interval == 0:
        frame = tracker.findHands(frame, draw=False)
        lmList = tracker.findPosition(frame, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = tracker.fingersUp()

            if only(fingers, [1, 2]): # Selection mode
                xp, yp = None, None

                if y1 < 125: 
                    if isInRange(x1, 188.75, 303.45):
                        current_header = headers[1]
                        current_color = (0, 0, 255)
                        current_thickness = brush_thickness
                    elif isInRange(x1, 406.25, 520.95):
                        current_header = headers[2]
                        current_color = (255, 0, 0)
                        current_thickness = brush_thickness
                    elif isInRange(x1, 623.75, 738.45):
                        current_header = headers[3]
                        current_color = (0, 255, 0)
                        current_thickness = brush_thickness
                    elif isInRange(x1, 828.95, 966.25):
                        current_header = headers[4]
                        current_color = (0, 0, 0)
                        current_thickness = eraser_thickness

                if current_color:
                    cv2.rectangle(frame, (x1, y1-30), (x2, y2+30), current_color, cv2.FILLED)
                
            elif only(fingers, [1]): # Draw mode
                if current_color:
                    cv2.circle(frame, (x1, y1), 15, current_color, cv2.FILLED)

                    if xp is not None and yp is not None:
                        distance = int(np.hypot(x1 - xp, y1 - yp))

                        step = max(1, distance // 10)
                        prev_x, prev_y = xp, yp

                        if distance > 0:
                            for i in range(0, distance + 1, step):
                                xi = int(xp + (x1 - xp) * (i / distance))
                                yi = int(yp + (y1 - yp) * (i / distance))

                                cv2.line(canvas, (prev_x, prev_y), (xi, yi), current_color, current_thickness)
                                prev_x, prev_y = xi, yi
                        
                    xp, yp = x1, y1
            else:
                xp, yp = None, None
    
    if xp is not None and yp is not None:
        cv2.circle(frame, (xp, yp), 15, current_color, cv2.FILLED)
    
    grayImg = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(grayImg, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)
            

    frame[0:125, 0:1920] = current_header
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()