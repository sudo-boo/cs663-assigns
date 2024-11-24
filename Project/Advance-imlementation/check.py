import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



def bgr2ycbcr(img):
    """Converts a BGR image to YCbCr color space.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        np.ndarray: YCbCr image.
    """
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    return ycbcr

def ycbcr2bgr(img):
    """Converts a YCbCr image to BGR color space.

    Args:
        img (np.ndarray): YCbCr image.

    Returns:
        np.ndarray: BGR image.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
    return bgr


def bgr2gray(img):
    """Converts a BGR image to grayscale.

    Args:
        img (np.ndarray): BGR image.

    Returns:
        np.ndarray: Grayscale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def downsample(img):
    """
    Downsample the Cb and Cr channels by a factor of 2.
    """
    y, cb, cr = cv2.split(img)
    cb = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2))
    cr = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2))
    cb = cv2.resize(cb, (y.shape[1], y.shape[0]))
    cr = cv2.resize(cr, (y.shape[1], y.shape[0]))
    
    return cv2.merge((y, cb, cr))


img = cv2.imread("output/original_imgs/img1.jpg", cv2.IMREAD_COLOR)
# cv2.imshow("Original", img)
# cv2.waitKey(0)
# plt.imshow(img)
# plt.show()
img = bgr2ycbcr(img)
img = downsample(img)
# img.shape
y = img[:,:,0]
cb = img[:,:,1]
cr = img[:,:,2]

# y.shape, cb.shape, cr.shape
# img.shape
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# plt.imshow(img)

# mpimg.imsave("test.jpg", img) # save the image as a file
# y = cv2.resize(y, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
# cb = cv2.resize(cb, (y.shape[1],y.shape[0]))
# cr = cv2.resize(cr, (y.shape[1], y.shape[0]))
# y.shape, cb.shape, cr.shape
final_img = cv2.merge((y, cb, cr))
final_img = ycbcr2bgr(final_img)
# cv2.imshow("Final", final_img)
# cv2.waitKey(0)
plt.imshow(final_img)
plt.show()


print(y.shape, cb.shape, cr.shape)
cv2.imwrite("test_y.jpg", y)
cv2.imwrite("test_cb.jpg", cb)
cv2.imwrite("test_cr.jpg", cr)

# mpimg.imsave("test_y.jpg", y) # save the image as a file
# mpimg.imsave("test_cb.jpg", cb) # save the image as a file
# mpimg.imsave("test_cr.jpg", cr) # save the image as a file

y_l = cv2.imread("test_y.jpg", cv2.IMREAD_GRAYSCALE)
cb_l = cv2.imread("test_cb.jpg", cv2.IMREAD_GRAYSCALE)
cr_l = cv2.imread("test_cr.jpg", cv2.IMREAD_GRAYSCALE)

# y_l = plt.imread("test_y.jpg")
# cb_l = plt.imread("test_cb.jpg")
# cr_l = plt.imread("test_cr.jpg")


#change y_l, cb_l, cr_l from 256x256x3 to 256x256
# y_l = cv2.cvtColor(y_l, cv2.COLOR_BGR2GRAY)
# cb_l = cv2.cvtColor(cb_l, cv2.COLOR_BGR2GRAY)
# cr_l = cv2.cvtColor(cr_l, cv2.COLOR_BGR2GRAY)

y_l = cv2.resize(y_l, (final_img.shape[1], final_img.shape[0]))
cb_l = cv2.resize(cb_l, (final_img.shape[1], final_img.shape[0]))
cr_l = cv2.resize(cr_l, (final_img.shape[1], final_img.shape[0]))

print(y_l.shape, cb_l.shape, cr_l.shape)
# print(y_l[0][0][:])

display_img = cv2.merge((y_l, cb_l, cr_l))
display_img = ycbcr2bgr(display_img)
# plt.imshow(display_img)
# plt.show()
# cv2.imshow("Compressed", display_img)
# cv2.waitKey(0)
mpimg.imsave("test2.jpg", display_img) # save the image as a file

# display_img = cv2.merge((y, cb, cr))
# # display_img = 
# cv2.imshow("Compressed", display_img)
# cv2.waitKey(0)

# final_img.shape
# final_img = cv2.cvtColor(final_img, cv2.COLOR_YCR_CB2BGR)
# cv2.imwrite("test2.jpg", final_img)
# # final_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2RGB)
# plt.imshow(final_img)
# # final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2YCrCb)
# # final_imd = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)
# plt.imshow(final_img)
# final_img2 = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2RGB)
# plt.imshow(final_img2)
