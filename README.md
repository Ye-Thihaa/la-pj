<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Morph App — Python + Linear Algebra</title>
  <meta name="description" content="A Python-based face morphing app using linear algebra (affine transforms, Delaunay triangulation) and OpenCV." />
  <style>
    :root{
      --bg: #0b0f17;
      --bg-soft:#101826;
      --card:#121b2b;
      --text:#e6eefc;
      --muted:#a6b3c8;
      --accent:#6ea8fe;
      --accent-2:#7cf5ff;
      --ring: 0 0 0 2px var(--accent) inset;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
      --radius: 18px;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;font-family:Inter,ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji";background:radial-gradient(1200px 800px at 10% -10%,rgba(126, 170, 255,.18),transparent),linear-gradient(180deg, #0b0f17 0%, #0c1220 100%);color:var(--text);}
    a{color:var(--accent)}
    .container{max-width:1100px;margin:0 auto;padding:24px}
    .nav{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:12px 0}
    .badge{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(90deg,var(--accent) 0%, var(--accent-2) 100%);color:#0a0f18;border-radius:999px;padding:8px 14px;font-weight:700;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
    .hero{display:grid;grid-template-columns:1.1fr .9fr;gap:28px;align-items:center;padding:56px 0}
    .card{background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));backdrop-filter: blur(6px);border:1px solid rgba(255,255,255,.08);border-radius:var(--radius);box-shadow:var(--shadow)}
    .hero h1{font-size:48px;line-height:1.08;margin:0 0 10px}
    .hero p{color:var(--muted);font-size:18px;margin:0 0 24px}
    .hero-cta{display:flex;gap:12px;flex-wrap:wrap}
    .btn{appearance:none;border:0;border-radius:14px;padding:12px 18px;font-weight:700;cursor:pointer}
    .btn-primary{background:linear-gradient(90deg,var(--accent),var(--accent-2));color:#07111f}
    .btn-ghost{background:transparent;border:1px solid rgba(255,255,255,.15);color:var(--text)}
    .code{position:relative;background: #0b1322;border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:16px;font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;font-size:13px;overflow:auto}
    .copy{position:absolute;top:10px;right:10px;font-size:12px;padding:6px 10px;border-radius:10px;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.06);color:var(--text);cursor:pointer}
    .grid{display:grid;gap:18px}
    .grid-3{grid-template-columns:repeat(3,1fr)}
    .grid-2{grid-template-columns:repeat(2,1fr)}
    .feature{padding:18px}
    .feature h3{margin:8px 0 6px}
    .muted{color:var(--muted)}
    section{padding:36px 0}
    h2{font-size:28px;margin:0 0 14px}
    .matrix{display:inline-grid;grid-template-columns:repeat(3,auto);gap:6px;padding:10px;border-radius:10px;background:#0b1322;border:1px solid rgba(255,255,255,.08)}
    .matrix span{padding:2px 6px;background:#0e1730;border-radius:6px}
    .footer{padding:30px 0;border-top:1px solid rgba(255,255,255,.08);color:var(--muted);font-size:14px}

    @media (max-width: 940px){
      .hero{grid-template-columns:1fr}
    }
    @media (max-width: 700px){
      .grid-3{grid-template-columns:1fr}
      .grid-2{grid-template-columns:1fr}
      .hero h1{font-size:38px}
    }
  </style>
</head>
<body>
  <header class="container nav">
    <div style="display:flex;align-items:center;gap:10px">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 3l3.5 6h-7L12 3zm0 18l-3.5-6h7L12 21z" fill="url(#g)"/><defs><linearGradient id="g" x1="0" y1="0" x2="24" y2="24" gradientUnits="userSpaceOnUse"><stop stop-color="#6ea8fe"/><stop offset="1" stop-color="#7cf5ff"/></linearGradient></defs></svg>
      <strong>Face Morph App</strong>
    </div>
    <nav style="display:flex;gap:14px">
      <a href="#features">Features</a>
      <a href="#how">How it works</a>
      <a href="#install">Install</a>
      <a href="#usage">Usage</a>
      <a href="#faq">FAQ</a>
    </nav>
  </header>

  <main class="container">
    <section class="hero">
      <div>
        <span class="badge">Python · OpenCV · Linear Algebra</span>
        <h1>Smooth Face Morphing powered by Affine Transforms</h1>
        <p>Blend one face into another with Delaunay triangulation, barycentric warping, and alpha blending. Built with Python, NumPy, and OpenCV.</p>
        <div class="hero-cta">
          <a class="btn btn-primary" href="#install">Get Started</a>
          <a class="btn btn-ghost" href="#demo">View Demo</a>
        </div>
      </div>
      <div class="card" style="padding:18px">
        <div class="code" id="code-install">
<pre>$ git clone https://github.com/your-username/face-morph-app
$ cd face-morph-app
$ pip install -r requirements.txt</pre>
          <button class="copy" data-copy="#code-install">Copy</button>
        </div>
        <div style="height:12px"></div>
        <div class="code" id="code-run">
<pre>$ python morph.py images/faceA.jpg images/faceB.jpg --alpha 0.5 --save out/morph_050.png</pre>
          <button class="copy" data-copy="#code-run">Copy</button>
        </div>
      </div>
    </section>

    <section id="features">
      <h2>Features</h2>
      <div class="grid grid-3">
        <div class="card feature">
          <h3>Triangulated Warping</h3>
          <p class="muted">Delaunay triangulation partitions faces into stable regions for distortion-free morphing.</p>
        </div>
        <div class="card feature">
          <h3>Affine Transforms</h3>
          <p class="muted">Each triangle is mapped by <em>2×3</em> affine matrices derived from point correspondences.</p>
        </div>
        <div class="card feature">
          <h3>Alpha Blending</h3>
          <p class="muted">Interpolate structure and color with a tunable <code>α</code> for smooth transitions.</p>
        </div>
      </div>
    </section>

    <section id="how">
      <h2>How it works (Linear Algebra)</h2>
      <div class="grid grid-2">
        <div class="card" style="padding:18px">
          <h3>Intermediate Shape</h3>
          <p class="muted">Given corresponding keypoints <code>P</code> (source) and <code>Q</code> (target), the intermediate landmarks are</p>
          <div class="code"><pre>S(α) = (1 − α) · P + α · Q</pre></div>
          <p class="muted">Triangulate <code>S(α)</code> once, then warp triangles from both images into this shape.</p>
        </div>
        <div class="card" style="padding:18px">
          <h3>Per‑Triangle Affine Map</h3>
          <p class="muted">For triangle vertices <code>p₁,p₂,p₃</code> → <code>s₁,s₂,s₃</code>, solve for the affine matrix <code>A</code> and translation <code>t</code>:</p>
          <div class="code"><pre>[ s ] = A [ p ] + t,  where  A ∈ ℝ^{2×2}, t ∈ ℝ^{2}

A = (P · B) · (S · B)^{-1}
</pre></div>
          <p class="muted">(Using a barycentric basis <code>B</code>; in practice you can compute with <code>cv2.getAffineTransform</code>.)</p>
        </div>
      </div>
      <div class="card" style="padding:18px;margin-top:18px">
        <h3>Blending</h3>
        <p class="muted">After warping both images into the intermediate shape, blend pixel values:</p>
        <div class="code"><pre>I_morph = (1 − α) · I_source→S + α · I_target→S</pre></div>
      </div>
    </section>

    <section id="install">
      <h2>Installation</h2>
      <div class="card" style="padding:18px">
        <ol class="muted">
          <li>Install Python 3.9+.</li>
          <li>Clone the repository and install dependencies.</li>
          <li>Prepare two face images with roughly frontal pose.</li>
        </ol>
        <div class="code" id="reqs">
<pre># requirements.txt
numpy
opencv-python
matplotlib
scipy
</pre>
          <button class="copy" data-copy="#reqs">Copy</button>
        </div>
      </div>
    </section>

    <section id="usage">
      <h2>Usage</h2>
      <div class="grid grid-2">
        <div class="card" style="padding:18px">
          <h3>Command Line</h3>
          <div class="code" id="cli">
<pre>python morph.py path/to/faceA.jpg path/to/faceB.jpg \
  --alpha 0.5 \
  --pointsA data/pointsA.txt \
  --pointsB data/pointsB.txt \
  --save out/morph_050.png</pre>
            <button class="copy" data-copy="#cli">Copy</button>
          </div>
        </div>
        <div class="card" style="padding:18px">
          <h3>Programmatic API</h3>
          <div class="code" id="api">
<pre>from morph import morph_faces
img = morph_faces("faceA.jpg", "faceB.jpg", alpha=0.35)
# returns a NumPy array; save with cv2.imwrite or show with matplotlib</pre>
            <button class="copy" data-copy="#api">Copy</button>
          </div>
        </div>
      </div>
    </section>

    <section id="demo">
      <h2>Demo</h2>
      <div class="grid grid-3">
        <div class="card" style="padding:12px;text-align:center">
          <img alt="Input face A" src="https://placehold.co/480x320/png" style="width:100%;border-radius:12px" />
          <p class="muted">Face A</p>
        </div>
        <div class="card" style="padding:12px;text-align:center">
          <img alt="Morphed output" src="https://placehold.co/480x320/png" style="width:100%;border-radius:12px" />
          <p class="muted">Morphed (α = 0.5)</p>
        </div>
        <div class="card" style="padding:12px;text-align:center">
          <img alt="Input face B" src="https://placehold.co/480x320/png" style="width:100%;border-radius:12px" />
          <p class="muted">Face B</p>
        </div>
      </div>
    </section>

    <section id="faq">
      <h2>FAQ</h2>
      <div class="grid grid-2">
        <div class="card" style="padding:18px">
          <h3>Do I need landmarks?</h3>
          <p class="muted">Yes. Provide corresponding facial landmarks (e.g., from dlib or MediaPipe Face Mesh). The quality of morph depends on these points.</p>
        </div>
        <div class="card" style="padding:18px">
          <h3>What if images are different sizes?</h3>
          <p class="muted">Images are resized to a common canvas, and triangles are computed in the shared coordinate frame before warping.</p>
        </div>
        <div class="card" style="padding:18px">
          <h3>Is this real‑time?</h3>
          <p class="muted">CPU morphing is typically offline. Lightweight models or GPU acceleration can achieve near real‑time for small images.</p>
        </div>
        <div class="card" style="padding:18px">
          <h3>What about color artifacts?</h3>
          <p class="muted">Use feathered masks per triangle and perform color correction (mean/variance matching) before blending.</p>
        </div>
      </div>
    </section>

    <section>
      <div class="card" style="padding:18px">
        <h2>License</h2>
        <p class="muted">MIT License — free to use, modify, and distribute.</p>
      </div>
    </section>

    <footer class="container footer">
      <div>© <span id="year"></span> Face Morph App. Built with ❤ using Python & Linear Algebra.</div>
    </footer>
  </main>

  <script>
    // Copy buttons
    document.querySelectorAll('.copy').forEach(btn => {
      btn.addEventListener('click', () => {
        const sel = btn.getAttribute('data-copy');
        const el = document.querySelector(sel);
        if(!el) return;
        const text = el.querySelector('pre').innerText;
        navigator.clipboard.writeText(text).then(()=>{
          const prev = btn.textContent;
          btn.textContent = 'Copied!';
          setTimeout(()=> btn.textContent = prev, 1200);
        });
      });
    });
    // Year
    document.getElementById('year').textContent = new Date().getFullYear();
  </script>
</body>
</html>
