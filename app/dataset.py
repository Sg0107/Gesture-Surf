import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

offset = 20
imgSize = 224

folder = folder = "data/five"
counter = 0

custom_connections = [
    (8, 7), (7, 6), (6, 5),
    (12, 11), (11, 10), (10, 9),
    (16, 15), (15, 14), (14, 13),
    (20, 19), (19, 18), (18, 17),
    (4, 3), (3, 2), (2, 1), (1, 0),
    (5, 2), (5, 9), (9, 13), (13, 17), (17, 0)
]   

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        success, img = cap.read()

        # Flip the image horizontally
        img = cv2.flip(img, 1)

        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks on the image with custom connections and set alpha value
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                connections=custom_connections,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255, 10), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255, 10), thickness=1)
            )

            # Get the bounding box of the hand
            x_min, x_max, y_min, y_max = 1.0, 0.0, 1.0, 0.0
            for landmark in hand_landmarks.landmark:
                x, y = landmark.x, landmark.y
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            x_min, x_max = int(x_min * img.shape[1]), int(x_max * img.shape[1])
            y_min, y_max = int(y_min * img.shape[0]), int(y_max * img.shape[0])
            w, h = x_max - x_min, y_max - y_min

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)

        # Quit the program if 'q' key is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        elif key == ord("s"):
            counter += 1
            timestamp = time.strftime("%Y%m%d%H%M%S")
            filename = f'Image_{timestamp}_{counter}.png'
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, imgWhite)
            print(f"Image saved: {filename}")
            print(counter)
        

cv2.destroyAllWindows()
