import cv2
import numpy as np
import math
import time
import os
import string

counter={char: 0 for char in string.ascii_uppercase}
cap = cv2.VideoCapture(1)
offset = 20
imgSize = 300


while True:
    success, img = cap.read()   
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key != -1:  # Check if any key is pressed
        if key == ord('q'):  # Check if the pressed key is 'q' to quit
            break
        else:
            key_pressed = chr(key)  # Convert the key value to its corresponding character
            folder_path = f'Data/{key_pressed}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)  # Create the folder if it doesn't exist
            counter[key_pressed] += 1
            cv2.imwrite(f'Data/{key_pressed}/{counter[key_pressed]}.jpg', img)
            print(counter[key_pressed])

cap.release()
cv2.destroyAllWindows()
