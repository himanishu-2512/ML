import pickle
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=1, static_image_mode=True, min_detection_confidence=0.3)

plt.figure(figsize=(8, 6))  # Create figure outside the loop

while True:
    data_aux = []
    x_ = []
    y_ = []
    # ret, img = cap.read()
    # H, W, _ = img.shape
    # frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imread("./Data/B/1.jpg")
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append([x - min(x_),y-min(y_)])
                # data_aux.append(y - min(y_))
            
        
        

        for i in data_aux:
            print(i)
    cv2.imshow('frame', img)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
