import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.stereo_processing import StereoProcessor
from scripts.stereoconfig import stereoCamera
import cv2
import numpy as np
cap = cv2.VideoCapture('data/dual_camera_20250505_163126.avi')
ret, frame = cap.read()
height, width = frame.shape[:2]
left_img = frame[:, :width//2]
right_img = frame[:, width//2:]
left_img = cv2.resize(left_img, (128, 96))
right_img = cv2.resize(right_img, (128, 96))
config = stereoCamera()
processor = StereoProcessor(config, 128, 96)
left_gray, right_gray = processor.preprocess(left_img, right_img)
left_rect, right_rect = processor.rectifyImage(left_gray, right_gray)
disparity = processor.stereoMatchSGBM(left_rect, right_rect)
points_3d = processor.compute_3d_points(disparity)
depth = np.median(points_3d[:, :, 2][np.isfinite(points_3d[:, :, 2]) & (points_3d[:, :, 2] > 0) & (points_3d[:, :, 2] < 10000)])
print(f"Median depth: {depth:.2f} mm")
cv2.imwrite('disparity_test.jpg', disparity / disparity.max() * 255)
cap.release()