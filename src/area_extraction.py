import cv2 
import numpy as np
import os
import copy

drawing = False
ix, iy = -1, -1
extracted_areas = []
image_path = os.path.dirname(os.path.dirname(__file__)) + "/data/Bilde1.png"
img = cv2.imread(image_path)
original_img = copy.deepcopy(img)

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, extracted_areas, img, original_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:

            
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
            if ix > x:
                ix, x = x, ix
            if iy > y:
                iy, y = y, iy
            extracted_area = img[iy+1:y-1, ix+1:x-1]  # Extract the marked area
            extracted_areas.append(extracted_area)
            drawing = False
            original_img = copy.deepcopy(img)

    elif event == cv2.EVENT_MOUSEMOVE: 
        if drawing: 
            temp_img = original_img.copy()
            cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 1)
            img = temp_img.copy()
    

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
for i, area in enumerate(extracted_areas):
    print(area.shape)
    cv2.imshow(f'Area {i}', area)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


