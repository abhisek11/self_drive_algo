import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150) #recommended to use low:high (1:3) intensity treshhold
    return canny

def region_of_intrest(image):
    height = image.shape[0]
    polygon = np.array([
        [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
corped_image = region_of_intrest(canny)

# plt.imshow(canny)
# plt.show()
cv2.imshow("results",corped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()