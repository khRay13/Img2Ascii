import random
import cv2, matplotlib.pyplot as plt, numpy as np

im_file = 'Samples/simpson.jpg'
im = cv2.imread(im_file)

h, w = im.shape[:2]
nh, nw = [128, 128]

# reshape
print("Reshape")
if h > w:
	rate = w/h
	nh = int(nw/rate)
	# print(nh,nw)
	im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC)

elif w > h:
	rate = h/w
	nw = int(nh/rate)
	# print(nh,nw)
	im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC)

else:
	im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_CUBIC)

chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
# plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB));plt.show()


print("Fetch Contour")
mosaicrate = 2
prepos_h = 0
prepos_w = 0
cnt = 0
# contour enhancement
im2 = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
for i in range(mosaicrate, nh + mosaicrate, mosaicrate):
	prepos_w=0

	for j in range(mosaicrate,nw + mosaicrate,mosaicrate):
		im2[prepos_h:i, prepos_w:j] = im2[prepos_h:i, prepos_w:j].max() - im2[prepos_h:i, prepos_w:j].min()
		prepos_w += mosaicrate
		cnt+=1

	prepos_h += mosaicrate
# plt.imshow(im2, cmap='gray');plt.show()


print("Convert image to Ascii")
im2_test = im2/(255/(chars.size-1))
with open('Samples/simpson.txt', 'w') as f:
	f.writelines("\n".join(("".join(r) for r in chars[im2_test.astype(int)])))
