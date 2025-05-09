# -*- coding: utf-8 -*-
import cv2
import numpy as np
import logging
from scripts.stereoconfig import stereoCamera

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StereoProcessor:
    def __init__(self, config, width, height):
        self.config = config
        self.width = width
        self.height = height
        # rectification maps and Q matrix
        self.map1x, self.map1y, self.map2x, self.map2y, self.Q = self.getRectifyTransform()
        # SGBM matcher
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=9, 
            P1=8 * 3 * 9 ** 2,
            P2=32 * 3 * 9 ** 2,
            disp12MaxDiff=1,
            preFilterCap=31,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_HH
        )
        # WLS filter
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
        self.wls_filter.setLambda(8000.0)
        self.wls_filter.setSigmaColor(1.5)

    def preprocess(self, img1, img2):
        """Convert to grayscale and apply CLAHE for better contrast."""
        if img1.ndim == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.ndim == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img1 = clahe.apply(img1)
        img2 = clahe.apply(img2)
        return img1, img2

    def getRectifyTransform(self):
        """Compute rectification transforms and reprojection matrix."""
        left_K = self.config.cam_matrix_left
        right_K = self.config.cam_matrix_right
        left_dist = self.config.distortion_l
        right_dist = self.config.distortion_r
        R = self.config.R
        T = self.config.T
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            left_K, left_dist, right_K, right_dist, (self.width, self.height), R, T, alpha=0
        )
        map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_dist, R1, P1, (self.width, self.height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_dist, R2, P2, (self.width, self.height), cv2.CV_32FC1)
        return map1x, map1y, map2x, map2y, Q

    def rectifyImage(self, img1, img2):
        """Apply rectification to images."""
        rectified_img1 = cv2.remap(img1, self.map1x, self.map1y, cv2.INTER_LINEAR)
        rectified_img2 = cv2.remap(img2, self.map2x, self.map2y, cv2.INTER_LINEAR)
        return rectified_img1, rectified_img2

    def stereoMatchSGBM(self, left_image, right_image):
        """Compute disparity map using SGBM with WLS filtering."""
        left_disparity = self.left_matcher.compute(left_image, right_image).astype(np.float32) / 16.0
        right_disparity = self.right_matcher.compute(right_image, left_image).astype(np.float32) / 16.0
        disparity = self.wls_filter.filter(left_disparity, left_image, None, right_disparity)
        disparity = cv2.medianBlur(disparity, 5)
        return disparity

    def compute_3d_points(self, disparity):
        """Compute 3D points from disparity map."""
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        return points_3d

    def get_depth_for_bbox(self, points_3d, bbox):
        """Compute median depth for a bounding box with strict validation."""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, self.width - 1))
        x2 = max(0, min(x2, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        y2 = max(0, min(y2, self.height - 1))
        if x2 <= x1 or y2 <= y1:
            logger.debug(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
            return None
        bbox_points = points_3d[y1:y2, x1:x2, 2]
        valid_points = bbox_points[np.isfinite(bbox_points) & (bbox_points > 200) & (bbox_points < 4000)]
        if valid_points.size == 0:
            logger.debug(f"No valid depth points in bbox: ({x1}, {y1}, {x2}, {y2}). Points: {bbox_points.flatten()[:10]}")
            return None
        return float(np.median(valid_points))