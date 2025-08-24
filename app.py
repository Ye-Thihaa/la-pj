from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
import dlib
from face_morph import segment_image_spectral  # import from your script

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static/uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load dlib models for face swap
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()


# ---------------- Face Morph ----------------
@app.route("/face_morph", methods=["GET", "POST"])
def face_morph():
    if request.method == "POST":
        if "image" not in request.files:
            return "Please upload an image!"
        img_file = request.files["image"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(img_path)

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        segmented = segment_image_spectral(img_rgb, num_clusters=6)

        # Save result
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], "morph_result.png")
        cv2.imwrite(result_path, (segmented * 40).astype(np.uint8))  # scale colors for visibility
        return render_template("face_morph.html", result="uploads/morph_result.png")
    return render_template("face_morph.html", result=None)


# ---------------- Face Swap ----------------
def get_landmarks(img):
    faces = detector(img)
    if len(faces) == 0:
        return None
    shape = predictor(img, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()])


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    t1_rect = np.array([[p[0] - r1[0], p[1] - r1[1]] for p in t1])
    t2_rect = np.array([[p[0] - r2[0], p[1] - r2[1]] for p in t2])
    img1_crop = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    M = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(img1_crop, M, (r2[2], r2[3]), None,
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                1 - mask) + warped * mask


@app.route("/face_swap", methods=["GET", "POST"])
def face_swap():
    if request.method == "POST":
        if "source" not in request.files or "target" not in request.files:
            return "Please upload both images!"
        source_file = request.files["source"]
        target_file = request.files["target"]

        source_path = os.path.join(app.config['UPLOAD_FOLDER'], source_file.filename)
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_file.filename)
        source_file.save(source_path)
        target_file.save(target_path)

        img1 = cv2.imread(source_path)
        img2 = cv2.imread(target_path)
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        landmarks1 = get_landmarks(img1_rgb)
        landmarks2 = get_landmarks(img2_rgb)
        if landmarks1 is None or landmarks2 is None:
            return "Could not detect face in one of the images."

        tri = Delaunay(landmarks2)
        img2_face_swapped = img2.copy()
        for tri_indices in tri.simplices:
            t1 = [landmarks1[i] for i in tri_indices]
            t2 = [landmarks2[i] for i in tri_indices]
            warp_triangle(img1, img2_face_swapped, t1, t2)

        result_path = os.path.join(app.config['UPLOAD_FOLDER'], "swap_result.jpg")
        cv2.imwrite(result_path, img2_face_swapped)
        return render_template("face_swap.html", result="uploads/swap_result.jpg")

    return render_template("face_swap.html", result=None)


# ---------------- Home ----------------
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
