import cv2
import os
import HandTracker as ht
import numpy as np
from utils import only, isInRange

assets_folder_path = "Assets"

cursor_filename = "cursor.png"

picker_folder_path = f"{assets_folder_path}/Pickers"
picker_files = os.listdir(picker_folder_path)

header_folder_path = f"{assets_folder_path}/Headers"
header_files = os.listdir(header_folder_path)

headers = [cv2.resize(cv2.imread(f'{header_folder_path}/{header_file}'), (1280, 125))
           for header_file in header_files if header_file.startswith("Header") and header_file.endswith(".png")]

cursor = cv2.resize(cv2.imread(f'{assets_folder_path}/{cursor_filename}'), (50, 50))

current_header = headers[0]

def draw_cursor(frame, x1, x2, y1, y2):
    if cursor is None:
        print("Error: Cursor image not loaded.")
        return
    
    if frame is None:
        print("Error: Frame is empty.")
        return

    cursor_x = (x1 + x2) // 2
    cursor_y = (y1 + y2) // 2

    cursor_h, cursor_w, _ = cursor.shape

    # Calculate the top-left and bottom-right coordinates
    top_left_x = max(0, cursor_x - cursor_w // 2)
    top_left_y = max(0, cursor_y - cursor_h // 2)

    if top_left_y < 125:
        return

    bottom_right_x = min(frame.shape[1], cursor_x + cursor_w // 2)
    bottom_right_y = min(frame.shape[0], cursor_y + cursor_h // 2)

    # Ensure the dimensions are valid for resizing
    resized_width = max(1, bottom_right_x - top_left_x)
    resized_height = max(1, bottom_right_y - top_left_y)

    # Resize the cursor image
    cursor_resized = cv2.resize(cursor, (resized_width, resized_height))

    # Overlay the cursor image onto the frame
    frame[top_left_y:top_left_y + resized_height, top_left_x:top_left_x + resized_width] = cursor_resized

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Variables for drawing -> This variables can be changed by the user during the program execution
current_color = None
current_mode = None
current_thickness = 15
brush_thickness = 15
eraser_thickness = 50

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tracker = ht.HandDetector(detectionCon=0.6)
canvas = np.zeros((720, 1280, 3), np.uint8)

# previous positions saved for drawing lines
xp, yp = None, None
coords = None

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
                current_mode = "Selection"

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

                coords = [x1, x2, y1, y2]
                if current_color:
                    draw_cursor(frame, x1, x2, y1, y2)
                
            elif only(fingers, [1]): # Draw mode
                current_mode = "Draw"
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
                    coords = None
            else:
                xp, yp = None, None
                coords = None
                current_mode = None
    
    if current_mode == "Draw":
        if xp is not None and yp is not None:
            cv2.circle(frame, (xp, yp), 15, current_color, cv2.FILLED)
    elif current_mode == "Selection":
        if coords is not None:
            draw_cursor(frame, *coords)
    
    if current_mode:
        cv2.putText(frame, f"{current_mode} mode", (800, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    grayImg = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(grayImg, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)
            

    frame[0:125, 0:1280] = current_header
    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()