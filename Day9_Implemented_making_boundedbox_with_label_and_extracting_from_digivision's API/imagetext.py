import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR model (use lang='en' for English)
ocr = PaddleOCR(use_angle_cls=True, lang='en') 

# Read image
img_path = "images_multi/adharsamp.png"
img = cv2.imread(img_path)

# Run OCR - get results with boxes and texts
result = ocr.ocr(img_path)

# Loop over each detected text box
rec_texts = result[0]["rec_texts"]
rec_boxes = result[0]["rec_boxes"]
rectangle=[]
for text, box in zip(rec_texts, rec_boxes):
    pt1=[int(box[0]),int(box[1])]
    pt2=[int(box[2]),int(box[3])]
    x_min = min(pt1[0], pt2[0])
    x_max = max(pt1[0], pt2[0])
    y_min = min(pt1[1], pt2[1])
    y_max = max(pt1[1], pt2[1])

    rectangle=[(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    print(rectangle)
    pts = np.array(rectangle, dtype=np.int32)

    # Reshape for polylines: needs shape (number_of_points, 1, 2)
    pts = pts.reshape((-1, 1, 2))

    # Draw the polygon (rectangle) on the image
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        

cv2.imwrite("bimg.png",img)
print("img created")