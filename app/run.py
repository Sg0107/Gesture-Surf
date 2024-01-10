import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import numpy as np
import math
import os

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
offset = 20
imgsize = 224  

model_path = 'keras_model.h5'
absolute_path = os.path.abspath(model_path)

model = load_model(absolute_path)

labels = ['one' , 'two', 'three', 'four', 'five']


while True:
    success, img = cap.read() # Read the image from the webcam
    img = cv2.flip(img, 1)  # Flip the image horizontally

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    results = hands.process(img_rgb)                                

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        x, y, w, h = x_min - offset, y_min - offset, x_max - x_min + 2 * offset, y_max - y_min + 2 * offset
        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgcrop = img[y : y + h, x : x + w]

        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = imgsize / h
            w_calc = math.ceil(k * w)
            imgresize = cv2.resize(imgcrop, (w_calc, imgsize))
            w_gap = math.ceil((imgsize - w_calc) / 2)
            imgwhite[:, w_gap:w_gap + w_calc] = imgresize
        else:
            k = imgsize / w
            h_calc = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, h_calc))
            h_gap = math.ceil((imgsize - h_calc) / 2)
            imgwhite[h_gap:h_gap + h_calc, :] = imgresize

        img_for_prediction = imgwhite / 255.0
        img_for_prediction = np.expand_dims(img_for_prediction, axis=0)

        prediction = model.predict(img_for_prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = labels[predicted_class_index]

        cv2.putText(img, predicted_class_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", img)
    key = cv2.waitKey(2)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
