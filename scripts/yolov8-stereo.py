# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import sys
import warnings
import logging
import argparse
from ultralytics import YOLO
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["NNPACK_ENABLED"] = "0"
os.environ["PYTORCH_NO_NNPACK"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.stereo_processing import StereoProcessor
from scripts.stereoconfig import stereoCamera

class KalmanFilter:
    """Simple Kalman filter for tracking (x, y, depth)."""
    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 3)  
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        return self.kf.correct(measurement)

    @property
    def statePost(self):
        return self.kf.statePost

def enhance_image(image):
    """Enhance contrast and edges using CLAHE in LAB color space, preserving color."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened

def refine_bbox_center(image, bbox):
    """Refine bounding box center using corner detection for sub-pixel accuracy."""
    x1, y1, x2, y2 = map(int, bbox)
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    roi = image[max(0, y1-10):min(image.shape[0], y2+10), max(0, x1-10):min(image.shape[1], x2+10)]
    if roi.size == 0:
        return center_x, center_y
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1, qualityLevel=0.01, minDistance=10)
    if corners is not None:
        x, y = corners[0].ravel()
        return x + max(0, x1-10), y + max(0, y1-10)
    return center_x, center_y

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def resolve_overlaps(detections, height, width):
    """Adjust class name and depth label positions to avoid overlaps."""
    adjusted_detections = []
    occupied_rects = []

    for det in detections:
        if det['depth'] is None:
            continue

        bbox = det['bbox']
        class_name = det['class_name']
        depth = det['depth']
        x1, y1, x2, y2 = map(int, bbox)
        font_scale = 0.3
        font_thickness = 1

        # Class name rectangle
        class_text = class_name
        text_size, baseline = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
        text_y = y1 - 5 - text_size[1]
        if text_y < text_size[1] + 5:
            text_y = y1 + text_size[1] + 5
        class_rect = [text_x, text_y - text_size[1], text_x + text_size[0], text_y + baseline]

        # Depth extension rectangle
        depth_text = f"{int(depth)} mm"
        text_size, baseline = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        ext_height = text_size[1] + 4
        ext_y1 = y2
        ext_y2 = y2 + ext_height
        depth_rect = [x1, ext_y1, x2, ext_y2]

        # Check for overlaps
        overlap = False
        for occ_rect in occupied_rects:
            if compute_iou(class_rect, occ_rect) > 0.1 or compute_iou(depth_rect, occ_rect) > 0.1:
                overlap = True
                break

        # Try repositioning class name
        if overlap:
            attempts = [
                (y1 - 15 - text_size[1], -15),  
                (y1 + text_size[1] + 15, 15),   
                (y1 - 25 - text_size[1], -25),  
            ]
            for new_y, offset in attempts:
                new_class_rect = [text_x, new_y - text_size[1], text_x + text_size[0], new_y + baseline]
                new_overlap = any(compute_iou(new_class_rect, occ_rect) > 0.1 for occ_rect in occupied_rects)
                if not new_overlap and 0 <= new_y - text_size[1] <= height and 0 <= new_y + baseline <= height:
                    class_rect = new_class_rect
                    text_y = new_y
                    overlap = False
                    break

        if overlap:
            logger.debug(f"Skipping drawing for {class_name} at bbox {bbox} due to unresolvable overlap")
            continue

        # Update detection with adjusted positions
        det['text_x'] = text_x
        det['text_y'] = text_y
        det['class_rect'] = class_rect
        det['depth_rect'] = depth_rect
        adjusted_detections.append(det)
        occupied_rects.append(class_rect)
        occupied_rects.append(depth_rect)

    return adjusted_detections

def process_frame(frame, stereo_processor, model, width, height, frame_width, frame_height, use_tracking=False, depth_history=None, detection_history=None, kalman_filters=None, object_id=0):
    """Process a single frame or image for detection and depth estimation."""
    frame_height, frame_width = frame.shape[:2]
    left_img = frame[:, :frame_width//2]
    right_img = frame[:, frame_width//2:]

    # Resize images
    left_img = cv2.resize(left_img, (width, height))
    right_img = cv2.resize(right_img, (width, height))
    left_img_enhanced = enhance_image(left_img)
    right_img_enhanced = enhance_image(right_img)

    # YOLOv8 detection on enhanced image
    try:
        results = model.predict(left_img_enhanced, conf=0.2, classes=[0, 1], iou=0.7)
        result = results[0]
    except Exception as e:
        logger.error(f"YOLOv8 prediction error: {e}")
        return left_img, [], object_id

    # Preprocess and compute disparity
    try:
        left_gray, right_gray = stereo_processor.preprocess(left_img, right_img)
        left_rect, right_rect = stereo_processor.rectifyImage(left_gray, right_gray)
        disparity = stereo_processor.stereoMatchSGBM(left_rect, right_rect)
    except Exception as e:
        logger.error(f"Stereo processing error: {e}")
        return left_img, [], object_id

    # Compute 3D points
    points_3d = stereo_processor.compute_3d_points(disparity)

    # Process detections
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls)
        class_name = result.names[cls_id]
        bbox = box.xyxy[0].cpu().numpy()
        conf = box.conf.cpu().numpy()[0]

        center_x, center_y = refine_bbox_center(left_img, bbox)

        # Compute and validate depth
        depth = stereo_processor.get_depth_for_bbox(points_3d, bbox)
        obj_key = f"{class_name}_{object_id}"
        smoothed_depth = depth

        if depth is not None and 200 < depth < 4000:
            if use_tracking:
                if depth_history is None or detection_history is None or kalman_filters is None:
                    logger.error("Tracking parameters not provided for video processing")
                    return left_img, [], object_id

                # Temporal smoothing and Kalman filter for video
                if obj_key not in depth_history:
                    depth_history[obj_key] = deque(maxlen=10)
                    kalman_filters[obj_key] = KalmanFilter()
                    object_id += 1
                depth_history[obj_key].append(depth)
                depths = list(depth_history[obj_key])
                smoothed_depth = np.median(depths) if depths else depth

                # Kalman filter update
                kf = kalman_filters[obj_key]
                kf.predict()
                measurement = np.array([[center_x], [center_y], [smoothed_depth]], dtype=np.float32)
                kf.correct(measurement)
                state = kf.statePost
                center_x, center_y, smoothed_depth = state[0, 0], state[1, 0], state[2, 0]
            else:
                # For images, no tracking, just increment object_id
                object_id += 1

            detections.append({
                'class_name': class_name,
                'confidence': conf,
                'depth': smoothed_depth,
                'bbox': bbox,
                'center_x': center_x,
                'center_y': center_y,
                'obj_id': obj_key
            })
            logger.info(f"Object: {class_name} (ID: {obj_key}), Confidence: {conf:.2f}, Depth: {smoothed_depth:.2f} mm, Center: ({center_x:.2f}, {center_y:.2f})")
        else:
            logger.warning(f"No valid depth for {class_name} at bbox {bbox}")
            object_id += 1

    # Temporal consistency for video
    if use_tracking and detection_history is not None:
        detection_history.append(detections)
        if len(detection_history) > 1:
            prev_detections = detection_history[-2]
            for det in detections:
                for prev_det in prev_detections:
                    if det['class_name'] == prev_det['class_name']:
                        iou = compute_iou(det['bbox'], prev_det['bbox'])
                        if iou > 0.7:
                            det['confidence'] = max(det['confidence'], prev_det['confidence'] * 0.95)

    # Sort detections by confidence for display
    max_display_objects = None
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    if max_display_objects is not None:
        detections = detections[:max_display_objects]
        
    detections = resolve_overlaps(detections, height, width)

    # Draw on original image
    for det in detections:
        bbox = det['bbox']
        class_name = det['class_name']
        depth = det['depth']
        text_x = det['text_x']
        text_y = det['text_y']
        obj_id = det['obj_id']
        color = (255, 0, 0) if class_name == 'branch' else (0, 0, 255) 
        x1, y1, x2, y2 = map(int, bbox)

        # Draw bounding box
        cv2.rectangle(left_img, (x1, y1), (x2, y2), color, 2)

        # Draw class name with marker line
        class_text = class_name
        font_scale = 0.3
        font_thickness = 1
        text_size, baseline = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        line_y_start = text_y + baseline - 2
        line_y_end = y1 if text_y < y1 else y1 + text_size[1]
        line_x = text_x + text_size[0] // 2
        cv2.line(left_img, (line_x, line_y_start), (line_x, line_y_end), color, 2)
        cv2.putText(left_img, class_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Draw depth label as box extension
        depth_text = f"{int(depth)} mm"
        text_size, baseline = cv2.getTextSize(depth_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        ext_height = text_size[1] + 4
        ext_y1 = y2
        ext_y2 = y2 + ext_height
        cv2.rectangle(left_img, (x1, ext_y1), (x2, ext_y2), color, -1)
        text_x = int(x1 + (x2 - x1 - text_size[0]) / 2)
        text_y = ext_y1 + text_size[1] + 2
        cv2.putText(left_img, depth_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    # Draw resolution on the image
    resolution_text = f"Resolution: {frame_width}x{frame_height}"
    font_scale = 0.5
    font_thickness = 1
    text_size, baseline = cv2.getTextSize(resolution_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x = 10
    text_y = 20
    cv2.rectangle(left_img, (text_x - 2, text_y - text_size[1] - 2), 
                  (text_x + text_size[0] + 2, text_y + baseline + 2), (0, 0, 0), -1) 
    cv2.putText(left_img, resolution_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    return left_img, detections, object_id

def main():
    # Parse cli arguments
    parser = argparse.ArgumentParser(description="Process stereo video or image for object detection and depth estimation.")
    parser.add_argument('--input', type=str, required=True, help='Path to input video or image (e.g., data/dual_camera.avi or data/30.jpg)')
    args = parser.parse_args()

    # Determine input type based on file extension
    input_path = args.input
    is_video = input_path.lower().endswith(('.avi', '.mp4', '.mov'))
    is_image = input_path.lower().endswith(('.jpg', '.jpeg', '.png'))

    if not (is_video or is_image):
        logger.error("Input must be a video (.avi, .mp4, .mov) or image (.jpg, .jpeg, .png)")
        return

    # Load YOLOv8 model
    model_path = os.path.join(os.path.dirname(__file__), '../runs/detect/train/weights/best.pt')
    fallback_model = os.path.join(os.path.dirname(__file__), '../models/yolov8n.pt')
    if not os.path.exists(model_path):
        logger.warning(f"Model file {model_path} not found. Falling back to {fallback_model}.")
        model_path = fallback_model
    try:
        model = YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return

    # Initialize stereo processor
    config = stereoCamera()
    width, height = 320, 256
    try:
        stereo_processor = StereoProcessor(config, width, height)
    except Exception as e:
        logger.error(f"Error initializing StereoProcessor: {e}")
        return

    if is_video:
        # Process video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {input_path}.")
            return

        # Get resolution
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Video output setup
        output_path = os.path.join(os.path.dirname(__file__), '../output/processed_stereo.avi')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        # Initialize tracking variables
        depth_history = {}
        detection_history = deque(maxlen=3)
        kalman_filters = {}
        object_id = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video or error reading frame.")
                break

            # Process frame
            left_img, detections, object_id = process_frame(
                frame, stereo_processor, model, width, height, frame_width, frame_height,
                use_tracking=True, depth_history=depth_history,
                detection_history=detection_history, kalman_filters=kalman_filters,
                object_id=object_id
            )

            # Log detection summary
            valid_detections = [d for d in detections if d['depth'] is not None]
            logger.info(f"Frame {frame_count}: Detected {len(valid_detections)} objects: {[d['class_name'] for d in valid_detections]}")
            if valid_detections:
                depths = [d['depth'] for d in valid_detections]
                logger.info(f"Depth stats: Min={min(depths):.2f}mm, Max={max(depths):.2f}mm, Mean={np.mean(depths):.2f}mm")

            # output video
            out.write(left_img)
            logger.debug("Rendered frame with bounding boxes, class names, and depth labels")

            cv2.imshow('YOLOv8 + Stereo', left_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
            if frame_count % 10 == 0:
                logger.info(f"Processed {frame_count} frames")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info("Video processing complete.")

    else:
        # Process image
        frame = cv2.imread(input_path)
        if frame is None:
            logger.error(f"Could not read image {input_path}.")
            return

        # Get resolution
        frame_height, frame_width = frame.shape[:2]

        # Process image
        left_img, detections, _ = process_frame(
            frame, stereo_processor, model, width, height, frame_width, frame_height,
            use_tracking=False
        )

        # Log detection summary
        valid_detections = [d for d in detections if d['depth'] is not None]
        logger.info(f"Image processed: Detected {len(valid_detections)} objects: {[d['class_name'] for d in valid_detections]}")
        if valid_detections:
            depths = [d['depth'] for d in valid_detections]
            logger.info(f"Depth stats: Min={min(depths):.2f}mm, Max={max(depths):.2f}mm, Mean={np.mean(depths):.2f}mm")

        # Save output image
        output_dir = os.path.join(os.path.dirname(__file__), '../output')
        os.makedirs(output_dir, exist_ok=True)
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"processed_{input_filename}.jpg")
        cv2.imwrite(output_path, left_img)
        logger.info(f"Saved processed image to {output_path}")

        cv2.imshow('YOLOv8 + Stereo', left_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.info("Image processing complete.")

if __name__ == "__main__":
    main()