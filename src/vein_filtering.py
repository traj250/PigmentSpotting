import sys

import cv2
import numpy as np
from skimage import io

def vein_enhance(img):
    img = cv2.blur(img, (5,5))
    #Find veins by using an adaptive threshold
    #img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,7,-10)
    #Find veins by using a thresholded Laplacian operator
    img = cv2.Laplacian(img, cv2.CV_16S, ksize = 5)
    io.imshow(img)
    io.show()
    (trash, img) = cv2.threshold(img, -300, 255, cv2.THRESH_BINARY_INV)
    #Close approximation of the "Remove Outliers..." tool of ImageJ
    img = img.astype(np.uint8)
    img = np.maximum(img, cv2.medianBlur(img, 3))
    #Try to close the gaps between vein segments
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, element, iterations = 2)

    io.imshow(img)
    io.show()
    #Removal of small spots
    label_count, labels = cv2.connectedComponents(img, connectivity=8)
    sizes, bins = np.histogram(labels, label_count+1)

    included = [i for i in range(1,label_count+1) if sizes[i] > img.shape[0]/50]
    result = cv2.inRange(labels, included[0], included[0])
    for component in included[1:]:
        result = cv2.bitwise_or(result, cv2.inRange(labels, component, component))
    return result
    
if __name__ == "__main__":
    vein_image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    result = vein_enhance(vein_image)
    io.imshow(result)
    io.show()
