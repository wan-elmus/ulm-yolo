""" 
Expected: rectified_test.jpg shows left and right images side-by-side with horizontal green lines aligned across corresponding features (e.g., branches). If misaligned, R, T, or camera matrices need recalibration.
"""

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
left_rect, right_rect = processor.rectifyImage(left_img, right_img)
output = np.hstack((left_rect, right_rect))
for y in range(0, 96, 10):
    cv2.line(output, (0, y), (256, y), (0, 255, 0), 1)
cv2.imwrite('rectified_test.jpg', output)
cap.release()