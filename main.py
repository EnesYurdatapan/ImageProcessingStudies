import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

# Converting to gray
img_original = cv2.imread('Pictures/picture1.jpg',)
gray_scale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)


plt.subplot(2,1,1),plt.imshow(img_original)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(gray_scale)
plt.title('Gray'), plt.xticks([]), plt.yticks([])
plt.show()



##########


# Logarithmic transformation

img = cv2.imread('Pictures/picture1.jpg')

# Apply log transformation method
c = 255 / np.log(1 + np.max(img))
log_image = c * (np.log(img + 1))
log_image = np.array(log_image, dtype=np.uint8)


plt.subplot(2,1,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(log_image)
plt.title('Log Image'), plt.xticks([]), plt.yticks([])
plt.show()



#######

# Prewitt Algorithm
img = cv2.imread('Pictures/picture1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)


kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)



plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img_prewittx + img_prewitty)
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(img_prewittx)
plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(img_prewitty)
plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
plt.show()



#######

#Sobel
img_original = cv2.imread('Pictures/picture1.jpg',)
gray_scale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

img = cv2.GaussianBlur(gray_scale,(3,3),0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)  # y

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

######


#Making image negative

img = cv2.imread('Pictures/picture1.jpg')

img_neg = cv2.bitwise_not(img)

plt.subplot(2,1,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(img_neg)
plt.title('Negative'), plt.xticks([]), plt.yticks([])
plt.show()



#######



# Robert Cross Algorithm
roberts_cross_v = np.array( [[1, 0 ],
							[0,-1 ]] )

roberts_cross_h = np.array( [[ 0, 1 ],
							[ -1, 0 ]] )

img = cv2.imread('Pictures/picture1.jpg', 0).astype('float64')
img/=255.0
vertical = ndimage.convolve( img, roberts_cross_v )
horizontal = ndimage.convolve( img, roberts_cross_h )

edged_img = np.sqrt( np.square(horizontal) + np.square(vertical))
edged_img*=255

plt.subplot(2,1,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(edged_img)
plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
plt.show()
