from flask import Flask, render_template, request, flash, redirect, url_for
import os
import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import dlib
from face_morph import segment_image_spectral  # import from your script
import logging
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or "static/uploads"
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


app.config.from_object(Config)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Download dlib model if not exists (for production)
def download_dlib_model():
    model_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        import urllib.request
        import bz2
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        logger.info("Downloading dlib model...")
        try:
            urllib.request.urlretrieve(url, "model.dat.bz2")
            with bz2.BZ2File("model.dat.bz2", 'rb') as f:
                with open(model_path, 'wb') as out:
                    out.write(f.read())
            os.remove("model.dat.bz2")
            logger.info("Dlib model downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download dlib model: {e}")
            return None
    return model_path


# Initialize dlib models with error handling
try:
    model_path = download_dlib_model()
    if model_path and os.path.exists(model_path):
        predictor = dlib.shape_predictor(model_path)
        detector = dlib.get_frontal_face_detector()
        logger.info("Dlib models loaded successfully")
    else:
        predictor = None
        detector = None
        logger.warning("Dlib models not available")
except Exception as e:
    logger.error(f"Error loading dlib models: {e}")
    predictor = None
    detector = None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def generate_unique_filename(original_filename):
    """Generate unique filename to avoid conflicts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(secure_filename(original_filename))
    return f"{name}_{timestamp}_{unique_id}{ext}"


def cleanup_old_files():
    """Clean up old uploaded files to save space"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        current_time = datetime.now().timestamp()

        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getctime(file_path)
                # Delete files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 16MB.', 'error')
    return redirect(request.url), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('error.html',
                           error_message="Internal server error occurred. Please try again."), 500


@app.errorhandler(404)
def not_found(error):
    return render_template('error.html',
                           error_message="Page not found."), 404


# Health check endpoint for monitoring
@app.route('/health')
def health_check():
    return {
        'status': 'healthy',
        'dlib_available': predictor is not None,
        'upload_dir_exists': os.path.exists(app.config['UPLOAD_FOLDER'])
    }, 200


# ---------------- Face Morph ----------------
@app.route("/face_morph", methods=["GET", "POST"])
def face_morph():
    if request.method == "POST":
        try:
            # Clean up old files
            cleanup_old_files()

            if "image" not in request.files:
                flash("Please upload an image!", 'error')
                return render_template("face_morph.html", result=None, graph=None)

            img_file = request.files["image"]

            if img_file.filename == '':
                flash("No file selected!", 'error')
                return render_template("face_morph.html", result=None, graph=None)

            if not allowed_file(img_file.filename):
                flash("Invalid file type. Please upload an image file.", 'error')
                return render_template("face_morph.html", result=None, graph=None)

            # Generate unique filename
            unique_filename = generate_unique_filename(img_file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            img_file.save(img_path)

            # Process image
            img = cv2.imread(img_path)
            if img is None:
                flash("Invalid image file!", 'error')
                return render_template("face_morph.html", result=None, graph=None)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Segment image
            segmented_labels, segmented_colored = segment_image_spectral(img_rgb, num_clusters=6)

            # Save segmented image with unique name
            result_filename = f"morph_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(segmented_colored.astype(np.uint8), cv2.COLOR_RGB2BGR))

            # Generate matplotlib figure
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(img_rgb)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(segmented_labels, cmap="viridis")
            axs[1].set_title("Label Map")
            axs[1].axis("off")

            axs[2].imshow(segmented_colored.astype(np.uint8))
            axs[2].set_title("Segmented Image")
            axs[2].axis("off")

            graph_filename = f"segmentation_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
            graph_path = os.path.join(app.config['UPLOAD_FOLDER'], graph_filename)
            plt.tight_layout()
            plt.savefig(graph_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            # Clean up input file
            try:
                os.remove(img_path)
            except:
                pass

            return render_template("face_morph.html",
                                   result=f"uploads/{result_filename}",
                                   graph=f"uploads/{graph_filename}")

        except Exception as e:
            logger.error(f"Error in face_morph: {e}")
            flash("An error occurred while processing the image. Please try again.", 'error')
            return render_template("face_morph.html", result=None, graph=None)

    return render_template("face_morph.html", result=None, graph=None)


# ---------------- Face Swap ----------------
def get_landmarks(img):
    if detector is None or predictor is None:
        return None
    try:
        faces = detector(img)
        if len(faces) == 0:
            return None
        shape = predictor(img, faces[0])
        return np.array([[p.x, p.y] for p in shape.parts()])
    except Exception as e:
        logger.error(f"Error in get_landmarks: {e}")
        return None


def warp_triangle(img1, img2, t1, t2):
    try:
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))
        t1_rect = np.array([[p[0] - r1[0], p[1] - r1[1]] for p in t1])
        t2_rect = np.array([[p[0] - r2[0], p[1] - r2[1]] for p in t2])

        img1_crop = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        if img1_crop.size == 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
        warped = cv2.warpAffine(img1_crop, M, (r2[2], r2[3]), None,
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))

        img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = \
            img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (1 - mask) + warped * mask
    except Exception as e:
        logger.error(f"Error in warp_triangle: {e}")


@app.route("/face_swap", methods=["GET", "POST"])
def face_swap():
    if request.method == "POST":
        try:
            # Check if dlib models are available
            if detector is None or predictor is None:
                flash("Face detection models are not available. Please try again later.", 'error')
                return render_template("face_swap.html", result=None)

            # Clean up old files
            cleanup_old_files()

            if "source" not in request.files or "target" not in request.files:
                flash("Please upload both source and target images!", 'error')
                return render_template("face_swap.html", result=None)

            source_file = request.files["source"]
            target_file = request.files["target"]

            if source_file.filename == '' or target_file.filename == '':
                flash("Please select both files!", 'error')
                return render_template("face_swap.html", result=None)

            if not (allowed_file(source_file.filename) and allowed_file(target_file.filename)):
                flash("Invalid file types. Please upload image files.", 'error')
                return render_template("face_swap.html", result=None)

            # Save uploaded files with unique names
            source_filename = generate_unique_filename(source_file.filename)
            target_filename = generate_unique_filename(target_file.filename)

            source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_filename)

            source_file.save(source_path)
            target_file.save(target_path)

            # Load and process images
            img1 = cv2.imread(source_path)
            img2 = cv2.imread(target_path)

            if img1 is None or img2 is None:
                flash("Invalid image files!", 'error')
                return render_template("face_swap.html", result=None)

            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # Get facial landmarks
            landmarks1 = get_landmarks(img1_rgb)
            landmarks2 = get_landmarks(img2_rgb)

            if landmarks1 is None or landmarks2 is None:
                flash("Could not detect faces in one or both images. Please ensure faces are clearly visible.", 'error')
                # Clean up uploaded files
                try:
                    os.remove(source_path)
                    os.remove(target_path)
                except:
                    pass
                return render_template("face_swap.html", result=None)

            # Perform face swap
            tri = Delaunay(landmarks2)
            img2_face_swapped = img2.copy()

            for tri_indices in tri.simplices:
                t1 = [landmarks1[i] for i in tri_indices]
                t2 = [landmarks2[i] for i in tri_indices]
                warp_triangle(img1, img2_face_swapped, t1, t2)

            # Save result with unique name
            result_filename = f"swap_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, img2_face_swapped)

            # Clean up input files
            try:
                os.remove(source_path)
                os.remove(target_path)
            except:
                pass

            return render_template("face_swap.html", result=f"uploads/{result_filename}")

        except Exception as e:
            logger.error(f"Error in face_swap: {e}")
            flash("An error occurred while processing the images. Please try again.", 'error')
            return render_template("face_swap.html", result=None)

    return render_template("face_swap.html", result=None)


# ---------------- Home ----------------
@app.route("/")
def index():
    return render_template("index.html")


# Production configuration
if __name__ == "__main__":
    # Development mode
    app.run(debug=True, host='0.0.0.0', port=5000)
else:
    # Production mode
    app.config['DEBUG'] = False
    # Configure for production logging
    if not app.debug:
        import logging
        from logging.handlers import RotatingFileHandler

        if not os.path.exists('logs'):
            os.mkdir('logs')

        file_handler = RotatingFileHandler('logs/face_app.log', maxBytes=10240, backupCount=10)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Face App startup')