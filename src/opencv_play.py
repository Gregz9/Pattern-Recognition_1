import cv2 
import numpy as np
import os

drawing = False
ix, iy = -1, -1
extracted_areas = []
# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, extracted_areas #, img, original_img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
            extracted_area = img[iy:y, ix:x]  # Extract the marked area
            print(iy, ix)
            print(y, x)
            extracted_areas.append(extracted_area)
            drawing = False

    # elif event == cv2.EVENT_MOUSEMOVE: 
        # if drawing: 
            # temp_img = original_img.copy()
            # cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
            # img = temp_img.copy()
    

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
print(extracted_areas[0].shape)
print(extracted_areas[1].shape)
# print(extracted_areas[0])
# print(extracted_areas[1])
# print(extracted_areas[2])


