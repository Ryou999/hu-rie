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
