A = rgb2gray(imread('images/minion.jpg'))
imshow(A)
plt.show()

def LgTrans(x):
    C = 1//log(1+1)
    return C*log(1+x)

imshow(A)
plt.colorbar()
plt.show()

im = np.abs(np.fft.fft2(A))

h,w = im.shape
im = np.roll(im,h//2,0)
im = np.roll(im,h//2,1)

for i in range(h):
    for j in range(w):
        im[i][j] = LgTrans(im[i][j])

imshow(im,vmin=0)
plt.colorbar()
plt.show()

print("max:",im.max())
print("min:",im.min())




















import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('images/minion.jpg',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

rows, cols = img.shape
crow,ccol = rows//2 , cols//2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()




















