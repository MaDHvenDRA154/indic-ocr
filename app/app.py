"""
app.py - Flask web UI for Indic OCR
Uses trained CRNN model (checkpoints/crnn_best.pt) for inference.
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import cv2
import pytesseract
import base64
import io
import json
from pathlib import Path
from flask import Flask, request, jsonify, Response

from src.preprocess import preprocess_from_array

app = Flask(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRNN_CHECKPOINT = "./checkpoints/crnn_best.pt"
VOCAB_PATH = "./checkpoints/vocab.json"

crnn_model = None
idx2char = {}


def configure_tesseract() -> bool:
  if shutil.which("tesseract"):
    return True
  candidates = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
  ]
  for path in candidates:
    if Path(path).exists():
      pytesseract.pytesseract.tesseract_cmd = path
      return True
  return False


TESSERACT_READY = configure_tesseract()

def load_crnn_model():
    global crnn_model, idx2char
    if not Path(CRNN_CHECKPOINT).exists():
        print("CRNN checkpoint not found.")
        return
    try:
        from src.model import get_crnn
        if Path(VOCAB_PATH).exists():
          with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            char2idx = json.load(f)
            idx2char = {int(v): k for k, v in char2idx.items()}
            vocab_size = len(char2idx)
        else:
            chars = ["क","ख","ग","घ","ङ","च","छ","ज","झ","ञ","ट","ठ","ड","ढ","ण",
                     "त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व","श",
                     "ष","स","ह","क्ष","त्र","ज्ञ","०","१","२","३","४","५","६","७","८","९"]
            idx2char = {i+1: c for i, c in enumerate(chars)}
            idx2char[0] = "<PAD>"
            vocab_size = len(idx2char) + 1
        crnn_model = get_crnn(vocab_size=vocab_size, device=DEVICE)
        crnn_model.load_state_dict(torch.load(CRNN_CHECKPOINT, map_location=DEVICE))
        crnn_model.eval()
        print(f"CRNN loaded. Vocab: {vocab_size}, Device: {DEVICE}")
    except Exception as e:
        print(f"Could not load CRNN: {e}")

load_crnn_model()

def predict_crnn(pil_image):
    if crnn_model is None:
        return None, "Model not loaded"
    try:
        np_image = np.array(pil_image.convert("L"))
        img = preprocess_from_array(np_image)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            log_probs = crnn_model(tensor)
        probs = torch.exp(log_probs)
        preds = log_probs.argmax(dim=2).squeeze(1)
        decoded = []
        prev = -1
        for p in preds:
            p = p.item()
            if p != 0 and p != prev:
                ch = idx2char.get(p, "")
                if ch and ch != "<PAD>":
                    decoded.append(ch)
            prev = p
        result = "".join(decoded)
        return result if result else "(No text detected)", None
    except Exception as e:
        return None, str(e)


def predict_paragraph(pil_image):
    """Run OCR for multi-line handwritten/printed Hindi text."""
    if not TESSERACT_READY:
        return None, "Tesseract executable not found. Install Tesseract OCR and Hindi language data."
    try:
        rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Upscale small inputs to improve OCR quality on handwriting.
        h, w = gray.shape[:2]
        if max(h, w) < 1400:
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        den = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thr = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(thr, lang="hin", config=config).strip()
        return text if text else "(No text detected)", None
    except Exception as e:
        return None, str(e)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Indic OCR</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
:root {
  --ink: #0a0a0a;
  --paper: #f7f4ef;
  --accent: #c84b2f;
  --accent2: #2f6bc8;
  --muted: #8a8580;
  --border: #d8d3cc;
  --card: #ffffff;
  --success: #2a7a3b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Syne', sans-serif;
  background: var(--paper);
  color: var(--ink);
  min-height: 100vh;
  overflow-x: hidden;
}

/* Background texture */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    radial-gradient(circle at 20% 20%, rgba(200,75,47,0.06) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(47,107,200,0.06) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}

.wrap { position: relative; z-index: 1; max-width: 1000px; margin: 0 auto; padding: 0 2rem; }

/* HEADER */
header {
  padding: 2.5rem 0 2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 3rem;
  animation: fadeDown 0.6s ease both;
}
.header-inner { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.logo { display: flex; align-items: center; gap: 14px; }
.logo-mark {
  width: 44px; height: 44px;
  background: var(--ink);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Noto Sans Devanagari', sans-serif;
  font-size: 20px; color: var(--paper); font-weight: 700;
}
.logo-text h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; }
.logo-text p { font-size: 12px; color: var(--muted); margin-top: 3px; font-weight: 400; }
.header-meta { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
  font-size: 11px; font-weight: 600; padding: 5px 12px;
  border-radius: 100px; letter-spacing: 0.03em;
}
.pill-accent { background: var(--accent); color: white; }
.pill-outline { border: 1.5px solid var(--border); color: var(--muted); }
.pill-success { background: #e8f5eb; color: var(--success); }

/* MAIN GRID */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  margin-bottom: 3rem;
  animation: fadeUp 0.7s ease 0.1s both;
}
@media (max-width: 680px) { .main-grid { grid-template-columns: 1fr; } }

/* CARDS */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
}
.card::after {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  opacity: 0;
  transition: opacity 0.3s;
}
.card:focus-within::after, .card.active::after { opacity: 1; }
.card-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
  margin-bottom: 1rem;
}

/* UPLOAD ZONE */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: 12px;
  min-height: 220px;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 10px; cursor: pointer;
  transition: all 0.25s ease;
  position: relative; overflow: hidden;
  background: #fafaf8;
}
.drop-zone:hover { border-color: var(--accent); background: #fdf6f4; }
.drop-zone.has-image { border-style: solid; border-color: var(--accent2); padding: 0; }
.drop-zone.dragging { border-color: var(--accent); background: #fdf0ed; transform: scale(0.99); }
.drop-zone img { width: 100%; height: 100%; object-fit: contain; border-radius: 10px; max-height: 220px; }
.drop-icon { font-size: 2rem; opacity: 0.3; }
.drop-text { font-size: 13px; color: var(--muted); text-align: center; line-height: 1.5; }
.drop-text strong { color: var(--ink); }
#fileInput { display: none; }

/* EXTRACT BUTTON */
.extract-btn {
  width: 100%; margin-top: 1rem;
  padding: 13px 20px;
  background: var(--ink);
  color: var(--paper);
  border: none; border-radius: 10px;
  font-family: 'Syne', sans-serif;
  font-size: 14px; font-weight: 700;
  letter-spacing: 0.04em;
  cursor: pointer;
  transition: all 0.2s;
  display: flex; align-items: center; justify-content: center; gap: 8px;
  position: relative; overflow: hidden;
}
.extract-btn::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  opacity: 0; transition: opacity 0.3s;
}
.extract-btn:hover::before { opacity: 1; }
.extract-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,0,0,0.15); }
.extract-btn:active { transform: translateY(0); }
.extract-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; box-shadow: none; }
.extract-btn span { position: relative; z-index: 1; }
.spin {
  width: 16px; height: 16px;
  border: 2px solid rgba(255,255,255,0.4);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  display: none; position: relative; z-index: 1;
}

/* OUTPUT */
.output-area {
  min-height: 220px;
  border-radius: 12px;
  background: #fafaf8;
  border: 1.5px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  position: relative; overflow: hidden;
  transition: all 0.3s;
}
.output-area.has-result { background: white; border-color: var(--accent2); align-items: flex-start; }
.output-placeholder { text-align: center; color: var(--muted); }
.output-placeholder .big { font-family: 'Noto Sans Devanagari', sans-serif; font-size: 3rem; opacity: 0.15; }
.output-placeholder p { font-size: 12px; margin-top: 8px; }
.output-text {
  font-family: 'Noto Sans Devanagari', sans-serif;
  font-size: 2.5rem; font-weight: 500;
  color: var(--ink); padding: 1.5rem;
  line-height: 1.5; width: 100%;
  animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) both;
}
.output-error {
  font-size: 12px; color: var(--accent);
  padding: 1rem; font-weight: 500;
  animation: fadeUp 0.3s ease both;
}

/* COPY BUTTON */
.copy-btn {
  position: absolute; bottom: 10px; right: 10px;
  background: var(--ink); color: var(--paper);
  border: none; border-radius: 6px;
  font-size: 11px; font-weight: 600;
  padding: 5px 10px; cursor: pointer;
  font-family: 'Syne', sans-serif;
  opacity: 0; transition: opacity 0.2s;
  letter-spacing: 0.04em;
}
.output-area.has-result:hover .copy-btn { opacity: 1; }

/* STATS ROW */
.stats-row {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 1rem; margin-bottom: 3rem;
  animation: fadeUp 0.7s ease 0.2s both;
}
.stat-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.25rem;
  text-align: center;
}
.stat-num {
  font-size: 2rem; font-weight: 800;
  letter-spacing: -0.03em; line-height: 1;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 11px; color: var(--muted); margin-top: 4px; font-weight: 500; }

/* SAMPLE CHARS */
.samples-section { animation: fadeUp 0.7s ease 0.3s both; margin-bottom: 3rem; }
.section-title { font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin-bottom: 1rem; }
.char-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.char-btn {
  width: 52px; height: 52px;
  border: 1.5px solid var(--border);
  border-radius: 10px; background: var(--card);
  font-family: 'Noto Sans Devanagari', sans-serif;
  font-size: 1.4rem; cursor: pointer;
  transition: all 0.15s; display: flex;
  align-items: center; justify-content: center;
  color: var(--ink);
}
.char-btn:hover { border-color: var(--accent); color: var(--accent); transform: scale(1.08); background: #fdf6f4; }
.char-btn.selected { border-color: var(--accent2); background: #eef4ff; color: var(--accent2); }

/* FOOTER */
footer {
  border-top: 1px solid var(--border);
  padding: 2rem 0; text-align: center;
  font-size: 12px; color: var(--muted);
  animation: fadeUp 0.7s ease 0.4s both;
}
footer strong { color: var(--ink); }

/* ANIMATIONS */
@keyframes fadeDown { from { opacity:0; transform:translateY(-16px); } to { opacity:1; transform:none; } }
@keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:none; } }
@keyframes popIn { from { opacity:0; transform:scale(0.85); } to { opacity:1; transform:scale(1); } }
@keyframes spin { to { transform: rotate(360deg); } }

/* PROGRESS BAR */
.progress-bar {
  height: 3px; background: var(--border); border-radius: 2px;
  margin-top: 10px; overflow: hidden; display: none;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  width: 0%; transition: width 0.3s ease;
  border-radius: 2px;
}
</style>
</head>
<body>

<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="header-inner">
      <div class="logo">
        <div class="logo-mark">&#2325;</div>
        <div class="logo-text">
          <h1>Indic OCR</h1>
          <p>Devanagari Handwriting Recognition</p>
        </div>
      </div>
      <div class="header-meta">
        <span class="pill pill-accent">CRNN + CTC</span>
        <span class="pill pill-success">4.4% CER</span>
        <span class="pill pill-outline">Madhvendra Sood · MIT Bengaluru</span>
      </div>
    </div>
  </header>

  <!-- STATS -->
  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-num">4.4%</div>
      <div class="stat-label">Character Error Rate</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">92K</div>
      <div class="stat-label">Training Samples</div>
    </div>
    <div class="stat-card">
      <div class="stat-num">89%</div>
      <div class="stat-label">Better than Tesseract</div>
    </div>
  </div>

  <!-- MAIN GRID -->
  <div class="main-grid">

    <!-- INPUT CARD -->
    <div class="card" id="inputCard">
      <div class="card-label">Input Image</div>
      <div class="drop-zone" id="dropZone"
           onclick="document.getElementById('fileInput').click()"
           ondragover="handleDragOver(event)"
           ondragleave="handleDragLeave(event)"
           ondrop="handleDrop(event)">
        <div class="drop-icon">⬆</div>
        <div class="drop-text">
          <strong>Click to upload</strong> or drag & drop<br>
          JPG, PNG, BMP · Any handwritten Devanagari
        </div>
      </div>
      <input type="file" id="fileInput" accept="image/*" onchange="handleFile(event)">
      <div class="progress-bar" id="progressBar">
        <div class="progress-fill" id="progressFill"></div>
      </div>
      <button class="extract-btn" id="extractBtn" onclick="runOCR()" disabled>
        <span>Extract Text</span>
        <div class="spin" id="spin"></div>
      </button>
    </div>

    <!-- OUTPUT CARD -->
    <div class="card" id="outputCard">
      <div class="card-label">Recognized Text</div>
      <div class="output-area" id="outputArea">
        <div class="output-placeholder">
          <div class="big">&#2325; &#2326; &#2327;</div>
          <p>Upload an image to begin</p>
        </div>
        <button class="copy-btn" id="copyBtn" onclick="copyText()">Copy</button>
      </div>
    </div>

  </div>

  <!-- SAMPLE CHARACTERS -->
  <div class="samples-section">
    <div class="section-title">Test with a sample character</div>
    <div class="char-grid" id="charGrid"></div>
  </div>

  <!-- FOOTER -->
  <footer>
    <strong>Indic OCR</strong> · CRNN + CTC Loss · Trained on Kaggle Devanagari Dataset ·
    Built by <strong>Madhvendra Sood</strong> · Manipal Institute of Technology, Bengaluru
  </footer>

</div>

<script>
const CHARS = ["\u0915","\u0916","\u0917","\u0918","\u0919","\u091A","\u091B","\u091C","\u091D","\u091E","\u091F","\u0920","\u0921","\u0922","\u0923","\u0924","\u0925","\u0926","\u0927","\u0928","\u092A","\u092B","\u092C","\u092D","\u092E","\u092F","\u0930","\u0932","\u0935","\u0936","\u0937","\u0938","\u0939"];
let imageData = null;
let lastResult = '';

// Build character grid
const grid = document.getElementById('charGrid');
CHARS.forEach(ch => {
  const btn = document.createElement('button');
  btn.className = 'char-btn';
  btn.textContent = ch;
  btn.title = `Test with "${ch}"`;
  btn.onclick = () => generateCharImage(ch, btn);
  grid.appendChild(btn);
});

function generateCharImage(char, btn) {
  // Create a canvas with the character
  const canvas = document.createElement('canvas');
  canvas.width = 64; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 64, 64);
  ctx.fillStyle = '#111';
  ctx.font = '44px "Noto Sans Devanagari", serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(char, 32, 34);
  imageData = canvas.toDataURL('image/png');
  // Show preview
  const dz = document.getElementById('dropZone');
  dz.innerHTML = `<img src="${imageData}" alt="${char}" style="max-height:220px;padding:1rem;">`;
  dz.classList.add('has-image');
  document.getElementById('extractBtn').disabled = false;
  // Mark selected
  document.querySelectorAll('.char-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
}

function handleFile(e) {
  const file = e.target.files[0];
  if (file) loadImageFile(file);
}

function loadImageFile(file) {
  const reader = new FileReader();
  reader.onload = ev => {
    imageData = ev.target.result;
    const dz = document.getElementById('dropZone');
    dz.innerHTML = `<img src="${imageData}" alt="preview">`;
    dz.classList.add('has-image');
    document.getElementById('extractBtn').disabled = false;
    document.querySelectorAll('.char-btn').forEach(b => b.classList.remove('selected'));
  };
  reader.readAsDataURL(file);
}

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.add('dragging');
}
function handleDragLeave(e) {
  document.getElementById('dropZone').classList.remove('dragging');
}
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (file) loadImageFile(file);
}

async function runOCR() {
  if (!imageData) return;
  const btn = document.getElementById('extractBtn');
  const spin = document.getElementById('spin');
  const prog = document.getElementById('progressBar');
  const fill = document.getElementById('progressFill');
  const output = document.getElementById('outputArea');

  btn.disabled = true;
  spin.style.display = 'block';
  prog.style.display = 'block';
  fill.style.width = '0%';

  // Animate progress
  let p = 0;
  const interval = setInterval(() => {
    p = Math.min(p + Math.random() * 15, 85);
    fill.style.width = p + '%';
  }, 120);

  output.innerHTML = '<div style="color:#aaa;font-size:13px;font-family:Syne,sans-serif">Analyzing...</div>';
  output.className = 'output-area';

  try {
    const resp = await fetch('/ocr', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: imageData})
    });
    const data = await resp.json();

    clearInterval(interval);
    fill.style.width = '100%';
    setTimeout(() => { prog.style.display = 'none'; fill.style.width = '0%'; }, 600);

    if (data.error) {
      output.innerHTML = `<div class="output-error">⚠ ${data.error}</div>`;
    } else {
      lastResult = data.text;
      output.className = 'output-area has-result';
      output.innerHTML = `
        <div class="output-text">${data.text}</div>
        <button class="copy-btn" id="copyBtn" onclick="copyText()">Copy</button>
      `;
    }
  } catch(e) {
    clearInterval(interval);
    output.innerHTML = `<div class="output-error">⚠ Network error: ${e.message}</div>`;
    prog.style.display = 'none';
  } finally {
    btn.disabled = false;
    spin.style.display = 'none';
  }
}

function copyText() {
  if (!lastResult) return;
  navigator.clipboard.writeText(lastResult).then(() => {
    const btn = document.getElementById('copyBtn');
    if (btn) { btn.textContent = 'Copied!'; setTimeout(() => { btn.textContent = 'Copy'; }, 1500); }
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
  return Response(HTML, content_type="text/html; charset=utf-8")

@app.route("/ocr", methods=["POST"])
def ocr():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",")[1]
    try:
        img_bytes = base64.b64decode(img_data)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        w, h = pil_image.size
        is_paragraph_like = (w >= 260 and h >= 100 and (w / max(1, h)) >= 1.4) or (w >= 500 and h >= 220)

        if is_paragraph_like or crnn_model is None:
            text, err = predict_paragraph(pil_image)
            if err and crnn_model is not None:
                text, err = predict_crnn(pil_image)
        else:
            text, err = predict_crnn(pil_image)

        if err:
            return jsonify({"error": err})
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n=== Indic OCR Web UI ===")
    print("Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
