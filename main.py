import cv2
import os
import HandTracker as ht
from utils import only, isInRange

header_folder_path = "Assets/Headers"
header_files = os.listdir(header_folder_path)

headers = []
for header_file in header_files:
    if header_file.startswith("Header") and header_file.endswith(".png"):
        image = cv2.imread(f'{header_folder_path}/{header_file}')
        headers.append(image)

current_header = headers[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

current_color = None

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tracker = ht.HandDetector(detectionCon=0.8)

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)

    frame = tracker.findHands(frame)
    lmList = tracker.findPosition(frame, draw=False)

    if len(lmList) == 0:
        continue

    x1, y1 = lmList[8][1:]
    x2, y2 = lmList[12][1:]

    fingers = tracker.fingersUp()

    if only(fingers, [1, 2]): # Selection mode
        if y1 < 125: 
            if isInRange(x1, 188.75, 303.45):
                current_header = headers[1]
                current_color = (0, 0, 255)
            elif isInRange(x1, 406.25, 520.95):
                current_header = headers[2]
                current_color = (255, 0, 0)
            elif isInRange(x1, 623.75, 738.45):
                current_header = headers[3]
                current_color = (0, 0, 0)
            elif isInRange(x1, 828.95, 966.25):
                current_header = headers[4]
                current_color = None

        if current_color:
            cv2.rectangle(frame, (x1, y1-30), (x2, y2+30), current_color, cv2.FILLED)
        
    elif only(fingers, [1]): # Draw mode
        if current_color:
            cv2.circle(frame, (x1, y1), 15, current_color, cv2.FILLED)

    frame[0:125, 0:1280] = current_header
    cv2.imshow("Image", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()