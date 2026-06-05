import os
import time
from flask import Flask, request, render_template, jsonify
from src.api.inference import InferenceEngine
from src.utils.logger import get_logger
import numpy as np
import base64
import io

logger = get_logger(__name__)

# Try to import heavy dependencies
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from src.preprocessing.landmark_extractor import FrameProcessor
    HAS_LANDMARK_EXTRACTOR = True
except Exception:
    HAS_LANDMARK_EXTRACTOR = False


def _load_env_defaults(example_path: str = None) -> None:
    """Load environment defaults from a .env.example file if env vars missing.

    This is a lightweight parser that sets os.environ only for missing keys.
    """
    if example_path is None:
        example_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env.example'))
    if not os.path.exists(example_path):
        return
    with open(example_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            if k not in os.environ:
                os.environ[k] = v


def decode_image_from_base64(image_data_url: str) -> np.ndarray | None:
    """Decode a base64 image data URL to numpy array (BGR)."""
    if not HAS_CV2:
        return None
    try:
        # Remove data URL prefix if present
        if ',' in image_data_url:
            image_data_url = image_data_url.split(',')[1]
        # Decode base64
        image_bytes = base64.b64decode(image_data_url)
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f'Error decoding image: {e}')
        return None


def create_app():
    _load_env_defaults()
    templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static'))
    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    model_path = os.environ.get('MODEL_PATH', 'model_weights.pth')
    app.config['MODEL_PATH'] = model_path
    app.inference = InferenceEngine(model_path=model_path)

    # Initialize landmark processor if available
    frame_processor = None
    if HAS_LANDMARK_EXTRACTOR:
        try:
            frame_processor = FrameProcessor(static_image_mode=False, refine_landmarks=True)
            app.frame_processor = frame_processor
            logger.info('Landmark extractor initialized')
        except Exception as e:
            logger.warning(f'Failed to initialize FrameProcessor: {e}')
            frame_processor = None

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/health')
    def health():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'mediapipe_available': HAS_LANDMARK_EXTRACTOR
        })

    @app.route('/predict', methods=['POST'])
    def predict():
        """Process image and return facial metrics."""
        try:
            payload = request.get_json(force=True)
            image_data = payload.get('data', '')

            if not image_data:
                return jsonify({'error': 'No image data provided'}), 400

            # Decode image
            frame = decode_image_from_base64(image_data)
            if frame is None:
                return jsonify({'error': 'Failed to decode image'}), 400

            # Extract landmarks and metrics
            metrics = {
                'ear_left': 0.0,
                'ear_right': 0.0,
                'mar': 0.0,
                'pitch': 0.0,
                'yaw': 0.0,
                'roll': 0.0,
                'perclos': 0.0,
                'confidence': 0.0,
            }

            if hasattr(app, 'frame_processor') and app.frame_processor:
                try:
                    features = app.frame_processor.process(frame)
                    metrics['ear_left'] = float(features.ear_left)
                    metrics['ear_right'] = float(features.ear_right)
                    metrics['mar'] = float(features.mar)
                    metrics['pitch'] = float(features.pitch)
                    metrics['yaw'] = float(features.yaw)
                    metrics['roll'] = float(features.roll)
                    # Confidence is based on average EAR
                    avg_ear = (features.ear_left + features.ear_right) / 2.0 if features.ear_left > 0 else 0.5
                    metrics['confidence'] = min(1.0, max(0.0, avg_ear))
                    # Simple PERCLOS estimate
                    metrics['perclos'] = max(0.0, 1.0 - avg_ear) if avg_ear > 0 else 0.5
                except Exception as e:
                    logger.error(f'Error processing landmarks: {e}')
                    metrics['confidence'] = 0.5
            else:
                # Fallback: return dummy metrics (when MediaPipe not available)
                metrics['confidence'] = np.random.uniform(0.4, 0.8)
                metrics['ear_left'] = np.random.uniform(0.25, 0.35)
                metrics['ear_right'] = np.random.uniform(0.25, 0.35)
                metrics['mar'] = np.random.uniform(0.4, 0.6)

            return jsonify(metrics)
        except Exception as e:
            logger.error(f'Prediction error: {e}')
            return jsonify({'error': str(e)}), 500

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
