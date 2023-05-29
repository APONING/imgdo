import cv2
import numpy as np

img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

distances = [m.distance for m in matches]
threshold = np.mean(distances) + np.std(distances)
good_matches = [m for m in matches if m.distance < threshold]

similarity = len(good_matches) / max(len(des1), len(des2)) * 100
print('图片相似度：', similarity, '%')