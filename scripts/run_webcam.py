#!/usr/bin/env python
"""
Standalone webcam demo for real-time driver alertness detection.

Captures video from webcam, detects facial landmarks, computes EAR/MAR metrics,
and displays results with alert status directly on frame using OpenCV.

No Flask required - pure OpenCV visualization.

Usage:
    python scripts/run_webcam.py --config configs/default.yaml
    python scripts/run_webcam.py --checkpoint outputs/models/best_model.pt

Controls:
    Q: Quit
    S: Take screenshot
    P: Pause/Resume
"""
import sys
from pathlib import Path
from collections import deque
from datetime import datetime

import click
import cv2
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.landmark_extractor import FrameProcessor, compute_ear, compute_mar
from src.preprocessing.feature_engineering import perclos, count_blinks, detect_yawn
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


class WebcamDemo:
    """Real-time webcam demo for alertness detection."""
    
    def __init__(self, config_path, checkpoint_path=None, device='cpu'):
        self.config = load_config(config_path)
        self.cfg = self.config['default']
        self.device = device
        
        # Thresholds
        self.ear_threshold = self.cfg['thresholds']['ear']
        self.mar_threshold = self.cfg['thresholds']['mar']
        self.perclos_threshold = self.cfg['thresholds']['perclos']
        self.ear_window_size = self.cfg['thresholds']['ear_window']
        self.perclos_window_size = self.cfg['thresholds']['perclos_window']
        
        # Metrics history
        self.ear_history = deque(maxlen=self.perclos_window_size)
        self.mar_history = deque(maxlen=self.perclos_window_size)
        self.blink_count = 0
        self.frame_count = 0
        
        # State
        self.paused = False
        self.drowsy = False
        self.display_settings = self.cfg['inference']
        
        # Initialize frame processor
        try:
            self.frame_processor = FrameProcessor()
            logger.info("FrameProcessor initialized")
        except Exception as e:
            logger.warning(f"FrameProcessor failed: {e}. Running without landmark detection.")
            self.frame_processor = None
    
    def process_frame(self, frame):
        """Process frame and compute metrics."""
        if self.frame_processor is None:
            return None
        
        try:
            features = self.frame_processor.process(frame, timestamp=None)
            return features
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            return None
    
    def compute_alertness(self):
        """Compute overall alertness state."""
        if len(self.ear_history) < self.perclos_window_size:
            return False, 0.0
        
        # Compute PERCLOS
        ear_array = np.array(list(self.ear_history))
        perclos_value = perclos(ear_array, threshold=self.ear_threshold)
        
        # Detect drowsiness
        drowsy = perclos_value > self.perclos_threshold
        
        return drowsy, perclos_value
    
    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks on frame."""
        if landmarks is None or len(landmarks) == 0:
            return
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Draw landmarks as circles
        for i, (x, y, z) in enumerate(landmarks):
            # Denormalize coordinates
            px = int(x * w)
            py = int(y * h)
            
            # Draw point
            cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
        
        logger.debug(f"Drew {len(landmarks)} landmarks")
    
    def draw_metrics(self, frame, features):
        """Draw EAR/MAR and other metrics on frame."""
        if features is None:
            return
        
        h, w = frame.shape[:2]
        
        # Update history
        self.ear_history.append((features.ear_left + features.ear_right) / 2)
        self.mar_history.append(features.mar)
        
        # Compute alertness
        drowsy, perclos_value = self.compute_alertness()
        self.drowsy = drowsy
        
        # Blink detection
        if len(self.ear_history) >= self.ear_window_size:
            ear_window = np.array(list(self.ear_history))[-self.ear_window_size:]
            blink_count = count_blinks(ear_window, threshold=self.ear_threshold, min_frames=3)
        else:
            blink_count = 0
        
        # Yawn detection
        if len(self.mar_history) >= 15:
            mar_window = np.array(list(self.mar_history))[-15:]
            yawn_detected = detect_yawn(mar_window, threshold=self.mar_threshold, min_frames=10)
        else:
            yawn_detected = False
        
        # Prepare text
        metrics_text = [
            f"EAR: {(features.ear_left + features.ear_right) / 2:.3f}",
            f"MAR: {features.mar:.3f}",
            f"Head Pose: P={features.pitch:.1f}° Y={features.yaw:.1f}° R={features.roll:.1f}°",
            f"PERCLOS: {perclos_value:.1%}",
            f"Blinks: {blink_count}",
            f"Yawn: {'YES' if yawn_detected else 'NO'}"
        ]
        
        # Draw metrics
        y_offset = 30
        for i, text in enumerate(metrics_text):
            if self.display_settings['display_metrics']:
                cv2.putText(frame, text, (10, y_offset + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw alert status
        alert_color = (0, 0, 255) if drowsy else (0, 255, 0)
        alert_text = "DROWSY ALERT!" if drowsy else "ALERT"
        alert_thickness = 3 if drowsy else 2
        
        cv2.putText(frame, alert_text, (w - 300, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, alert_thickness)
        
        # Draw FPS
        if self.display_settings['display_fps']:
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(frame, fps_text, (w - 150, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def draw_info(self, frame):
        """Draw info text on frame."""
        h, w = frame.shape[:2]
        
        info_lines = [
            "Press Q to quit, S to screenshot, P to pause/resume",
        ]
        
        for i, text in enumerate(info_lines):
            cv2.putText(frame, text, (10, h - 10 - i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def run(self):
        """Run webcam demo."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return
        
        # Set video properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Webcam opened. Press Q to quit.")
        
        frame_count = 0
        prev_time = cv2.getTickCount()
        self.fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Compute FPS
                current_time = cv2.getTickCount()
                self.fps = cv2.getTickFrequency() / (current_time - prev_time)
                prev_time = current_time
                
                frame_count += 1
                self.frame_count = frame_count
                
                # Process frame
                if not self.paused:
                    # Skip frames if configured
                    skip_frames = self.display_settings.get('frame_skip', 0)
                    if frame_count % (skip_frames + 1) == 0:
                        features = self.process_frame(frame)
                        
                        # Draw landmarks
                        if self.display_settings['display_landmarks'] and features:
                            self.draw_landmarks(frame, features.landmarks)
                        
                        # Draw metrics
                        if features:
                            self.draw_metrics(frame, features)
                else:
                    cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 100, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                
                # Draw info
                self.draw_info(frame)
                
                # Display frame
                cv2.imshow('Driver Alertness Detection - Press Q to quit', frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quitting...")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"Saved screenshot to {screenshot_path}")
                elif key == ord('p'):
                    self.paused = not self.paused
                    logger.info(f"Paused: {self.paused}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {frame_count} frames. Exiting.")


@click.command()
@click.option('--config', default='configs/default.yaml', help='Config YAML path')
@click.option('--checkpoint', default=None, help='Model checkpoint path (optional)')
@click.option('--device', default='auto', help='Device: auto, cpu, or cuda')
def main(config, checkpoint, device):
    """Run real-time webcam demo for driver alertness detection."""
    
    # Select device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create demo
    demo = WebcamDemo(config, checkpoint_path=checkpoint, device=device)
    
    print("\n" + "="*60)
    print("DRIVER ALERTNESS DETECTION - WEBCAM DEMO")
    print("="*60)
    print(f"Config: {config}")
    print(f"Device: {device}")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Screenshot")
    print("  P - Pause/Resume")
    print("\nThresholds:")
    print(f"  EAR threshold: {demo.ear_threshold}")
    print(f"  MAR threshold: {demo.mar_threshold}")
    print(f"  PERCLOS threshold: {demo.perclos_threshold}")
    print("="*60 + "\n")
    
    # Run demo
    demo.run()


if __name__ == '__main__':
    main()

