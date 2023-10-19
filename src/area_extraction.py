import cv2 
import numpy as np
import os

drawing = False
ix, iy = -1, -1
extracted_areas = []

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, extracted_areas
    mode = True 

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            extracted_area = img[iy:y, ix:x]  # Extract the marked area
            extracted_areas.append(extracted_area)
            drawing = False


# Load the image
image_path = os.path.dirname(os.path.dirname(__file__)) + "/data/Bilde1.png"
img = cv2.imread(image_path)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()

# Convert the extracted areas to numpy arrays
print(extracted_areas)
      

