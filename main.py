import cv2
import os

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

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)



    frame[0:125, 0:1280] = current_header
    cv2.imshow("Image", frame)
    cv2.waitKey(1)