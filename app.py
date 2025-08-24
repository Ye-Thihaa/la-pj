from flask import Flask, render_template, request, flash, redirect
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import dlib
from face_morph import segment_image_spectral  # your segmentation function
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Upload settings
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", "static/uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif", "bmp"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- dlib Model ----------------
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(MODEL_PATH):
    import urllib.request, bz2
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, "model.dat.bz2")
    with bz2.BZ2File("model.dat.bz2", "rb") as f, open(MODEL_PATH, "wb") as out:
        out.write(f.read())
    os.remove("model.dat.bz2")

predictor = dlib.shape_predictor(MODEL_PATH)
detector = dlib.get_frontal_face_detector()

# ---------------- Helpers ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def generate_unique_filename(original_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    name, ext = os.path.splitext(secure_filename(original_filename))
    return f"{name}_{timestamp}_{unique_id}{ext}"

# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

# ---- Face Morph ----
@app.route("/face_morph", methods=["GET", "POST"])
def face_morph():
    if request.method == "POST":
        if "image" not in request.files:
            flash("Please upload an image!", "error")
            return render_template("face_morph.html", result=None, graph=None)

        img_file = request.files["image"]
        if img_file.filename == "" or not allowed_file(img_file.filename):
            flash("Invalid file type or no file selected.", "error")
            return render_template("face_morph.html", result=None, graph=None)

        filename = generate_unique_filename(img_file.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        img_file.save(img_path)

        img = cv2.imread(img_path)
        if img is None:
            flash("Invalid image file!", "error")
            return render_template("face_morph.html", result=None, graph=None)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Segment image
        segmented_labels, segmented_colored = segment_image_spectral(img_rgb, num_clusters=6)

        # Save segmented image
        result_filename = f"morph_{uuid.uuid4().hex[:8]}.png"
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(segmented_colored.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # Generate matplotlib figure
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_rgb); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(segmented_labels, cmap="viridis"); axs[1].set_title("Label Map"); axs[1].axis("off")
        axs[2].imshow(segmented_colored.astype(np.uint8)); axs[2].set_title("Segmented"); axs[2].axis("off")
        graph_filename = f"graph_{uuid.uuid4().hex[:8]}.png"
        graph_path = os.path.join(app.config["UPLOAD_FOLDER"], graph_filename)
        plt.tight_layout(); plt.savefig(graph_path); plt.close(fig)

        return render_template("face_morph.html",
                               result=f"uploads/{result_filename}",
                               graph=f"uploads/{graph_filename}")

    return render_template("face_morph.html", result=None, graph=None)

# ---- Face Swap ----
def get_landmarks(img):
    faces = detector(img)
    if not faces:
        return None
    shape = predictor(img, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()])

def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1_rect = np.array([[p[0]-r1[0], p[1]-r1[1]] for p in t1])
    t2_rect = np.array([[p[0]-r2[0], p[1]-r2[1]] for p in t2])
    img1_crop = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if img1_crop.size == 0: return
    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(img1_crop, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0,1.0,1.0))
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = \
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]*(1-mask)+warped*mask

@app.route("/face_swap", methods=["GET", "POST"])
def face_swap():
    if request.method == "POST":
        if "source" not in request.files or "target" not in request.files:
            flash("Please upload both images!", "error")
            return render_template("face_swap.html", result=None)

        source_file, target_file = request.files["source"], request.files["target"]
        if source_file.filename == "" or target_file.filename == "":
            flash("Missing file(s).", "error")
            return render_template("face_swap.html", result=None)

        src_name, tgt_name = generate_unique_filename(source_file.filename), generate_unique_filename(target_file.filename)
        src_path, tgt_path = os.path.join(app.config["UPLOAD_FOLDER"], src_name), os.path.join(app.config["UPLOAD_FOLDER"], tgt_name)
        source_file.save(src_path); target_file.save(tgt_path)

        img1, img2 = cv2.imread(src_path), cv2.imread(tgt_path)
        if img1 is None or img2 is None:
            flash("Invalid image files!", "error")
            return render_template("face_swap.html", result=None)

        img1_rgb, img2_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        landmarks1, landmarks2 = get_landmarks(img1_rgb), get_landmarks(img2_rgb)
        if landmarks1 is None or landmarks2 is None:
            flash("Face not detected in one or both images.", "error")
            return render_template("face_swap.html", result=None)

        tri = Delaunay(landmarks2)
        swapped = img2.copy()
        for tri_indices in tri.simplices:
            warp_triangle(img1, swapped, [landmarks1[i] for i in tri_indices], [landmarks2[i] for i in tri_indices])

        result_filename = f"swap_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], result_filename)
        cv2.imwrite(result_path, swapped)

        return render_template("face_swap.html", result=f"uploads/{result_filename}")

    return render_template("face_swap.html", result=None)

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
