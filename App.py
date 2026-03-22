import gradio as gr
import torch
import numpy as np
import cv2
import os
from PIL import Image
from transformers import pipeline

# -----------------------------------------------
# Load the AI image detection model from HuggingFace
# Vision Transformer trained to detect AI generated
# images vs real photographs
# -----------------------------------------------
print("Loading AI detection model... please wait")
detector = pipeline(
    "image-classification",
    model="umm-maybe/AI-image-detector",
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded successfully!")


# -----------------------------------------------
# Analyze a single image
# -----------------------------------------------
def analyze_image(image):
    if image is None:
        return None, "Please upload an image first."

    try:
        results = detector(image)
        top = results[0]
        label = top["label"]
        score = top["score"] * 100

        if "artificial" in label.lower() or "fake" in label.lower() or "ai" in label.lower():
            verdict = "AI GENERATED"
            description = f"This image appears to be AI generated with {score:.1f}% confidence. The visual patterns, textures, and artifacts are consistent with images produced by generative AI models such as Midjourney, DALL-E, or Stable Diffusion."
        else:
            verdict = "REAL IMAGE"
            description = f"This image appears to be authentic with {score:.1f}% confidence. The visual characteristics are consistent with a genuine photograph taken by a camera."

        bar_length = 20
        filled = int((score / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        result_text = f"""## {verdict}

**Confidence:** {score:.1f}%
`{bar}` {score:.1f}%

---

**Analysis:** {description}

---

**Model:** umm-maybe/AI-image-detector (Vision Transformer)

*No AI detector is 100% accurate. Use this as a guide, not a definitive verdict.*"""

        return image, result_text

    except Exception as e:
        return image, f"Error analyzing image: {str(e)}"


# -----------------------------------------------
# Extract frames from a video file
# -----------------------------------------------
def extract_frames(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    interval = max(1, total_frames // max_frames)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0 and len(frames) < max_frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)
        frame_count += 1

    cap.release()
    return frames, duration, fps


# -----------------------------------------------
# Analyze a video by checking multiple frames
# -----------------------------------------------
def analyze_video(video_path):
    if video_path is None:
        return "Please upload a video first."

    try:
        frames, duration, fps = extract_frames(video_path, max_frames=10)

        if len(frames) == 0:
            return "Could not extract frames from this video. Please try a different file."

        ai_scores = []
        real_scores = []
        frame_results = []

        for i, frame in enumerate(frames):
            results = detector(frame)
            for r in results:
                lbl = r["label"].lower()
                sc = r["score"]
                if "artificial" in lbl or "fake" in lbl or "ai" in lbl:
                    ai_scores.append(sc)
                else:
                    real_scores.append(sc)

            top = results[0]
            is_ai = "artificial" in top["label"].lower() or "fake" in top["label"].lower()
            frame_results.append({
                "frame": i + 1,
                "verdict": "AI" if is_ai else "Real",
                "confidence": top["score"] * 100
            })

        avg_ai = np.mean(ai_scores) * 100 if ai_scores else 0
        avg_real = np.mean(real_scores) * 100 if real_scores else 0
        ai_frame_count = sum(1 for r in frame_results if r["verdict"] == "AI")

        if ai_frame_count > len(frames) / 2:
            verdict = "AI GENERATED VIDEO"
            confidence = avg_ai
            description = f"This video appears to be AI generated. {ai_frame_count} out of {len(frames)} sampled frames showed AI-generated characteristics."
        else:
            verdict = "REAL VIDEO"
            confidence = avg_real
            description = f"This video appears to be authentic. {len(frames) - ai_frame_count} out of {len(frames)} sampled frames showed real characteristics."

        bar_length = 20
        filled = int((confidence / 100) * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        frame_table = "| Frame | Verdict | Confidence |\n|-------|---------|------------|\n"
        for r in frame_results:
            frame_table += f"| {r['frame']} | {r['verdict']} | {r['confidence']:.1f}% |\n"

        result_text = f"""## {verdict}

**Overall Confidence:** {confidence:.1f}%
`{bar}` {confidence:.1f}%

---

**Analysis:** {description}

**Video Info:** {duration:.1f}s duration | {fps:.0f} FPS | {len(frames)} frames analyzed

---

### Frame-by-Frame Breakdown
{frame_table}

---

**Model:** umm-maybe/AI-image-detector (Vision Transformer)

*Video analysis is based on frame sampling. Results may vary with compression artifacts.*"""

        return result_text

    except Exception as e:
        return f"Error analyzing video: {str(e)}"


# -----------------------------------------------
# Custom CSS
# -----------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

* { box-sizing: border-box; }

:root {
    --ink: #0f0e0d;
    --paper: #f5f0e8;
    --cream: #ede8dc;
    --accent: #c8392b;
    --muted: #7a7469;
    --border: #d4cfc4;
}

html, body {
    background: var(--paper) !important;
}

.gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--paper) !important;
    max-width: 100% !important;
    margin: 0 !important;
    color: var(--ink) !important;
    padding: 0 2rem !important;
}

body, .gradio-container, .main, .wrap, .block {
    background: var(--paper) !important;
    color: var(--ink) !important;
}

p, span, label, h1, h2, h3, h4 {
    color: var(--ink) !important;
}

strong {
    color: var(--ink) !important;
    font-weight: 600 !important;
}

code, pre {
    color: var(--accent) !important;
    background: transparent !important;
}

.prose *, .markdown-body *, .result-panel * {
    color: var(--ink) !important;
}

/* Table */
table, tr, td, th {
    color: var(--ink) !important;
    background: var(--cream) !important;
    border-color: var(--border) !important;
}

th {
    background: var(--border) !important;
    color: var(--ink) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* Tabs */
.tab-nav {
    border-bottom: 1.5px solid var(--border) !important;
    background: transparent !important;
    margin-bottom: 2.5rem !important;
}

.tab-nav button,
.tab-nav button span,
div[role="tablist"] button,
div[role="tablist"] button span,
button[role="tab"],
button[role="tab"] span {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #0f0e0d !important;
    opacity: 1 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.8rem 1.4rem !important;
    background: transparent !important;
    transition: all 0.2s !important;
}

div[role="tablist"] button:hover,
div[role="tablist"] button:hover span,
button[role="tab"]:hover,
button[role="tab"]:hover span {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    opacity: 1 !important;
}

div[role="tablist"] button[aria-selected="true"],
div[role="tablist"] button[aria-selected="true"] span,
button[role="tab"][aria-selected="true"],
button[role="tab"][aria-selected="true"] span {
    color: #2563eb !important;
    border-bottom-color: #2563eb !important;
    opacity: 1 !important;
}

/* Upload area */
.upload-region {
    border: 1.5px dashed var(--border) !important;
    border-radius: 3px !important;
    background: var(--cream) !important;
    transition: all 0.2s !important;
    min-height: 280px !important;
}

.upload-region:hover {
    border-color: var(--accent) !important;
    background: #fdf5f3 !important;
}

/* Analyze button */
button.lg.primary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    background: var(--ink) !important;
    color: var(--paper) !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.9rem 2rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
}

button.lg.primary:hover {
    background: var(--accent) !important;
    transform: translateY(-1px) !important;
}

/* Result panel */
.result-panel {
    background: var(--cream) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 1.5rem !important;
    min-height: 280px;
    animation: fadeUp 0.4s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-panel h2 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.8rem !important;
    color: var(--ink) !important;
}

.result-panel code {
    font-size: 0.78rem !important;
    color: var(--accent) !important;
    background: transparent !important;
}

/* Tip bar */
.tip-bar {
    margin-top: 1.5rem;
    padding: 0.9rem 1.2rem;
    background: var(--cream);
    border-left: 2px solid var(--accent);
    border-radius: 0 3px 3px 0;
    font-size: 0.82rem;
    color: var(--muted);
    line-height: 1.6;
}

/* Stats row */
.stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5px;
    background: var(--border);
    border: 1.5px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 3.5rem;
}

.stat-cell {
    background: var(--cream);
    padding: 1.4rem 1.8rem;
    text-align: center;
}

.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--ink);
    display: block;
}

.stat-desc {
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 3px;
    display: block;
}

/* Hero */
.site-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1.4rem 0;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 3.5rem;
}

.site-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: -0.02em;
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 10px;
}

.logo-indicator {
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.75); }
}

.hero-section {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 3rem;
    align-items: center;
    margin-bottom: 4rem;
}

.hero-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.hero-eyebrow::before {
    content: '';
    display: block;
    width: 28px;
    height: 1.5px;
    background: var(--accent);
}

.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.6rem !important;
    font-weight: 800 !important;
    line-height: 1.0 !important;
    letter-spacing: -0.04em !important;
    color: var(--ink) !important;
    margin-bottom: 1.4rem !important;
}

.hero-title em {
    font-style: italic;
    color: var(--muted);
    font-weight: 400;
}

.hero-body {
    font-size: 0.92rem;
    color: var(--muted);
    line-height: 1.75;
    max-width: 320px;
}

.scan-visual {
    border: 1.5px solid var(--border);
    border-radius: 3px;
    background: var(--cream);
    height: 240px;
    position: relative;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
}

.scan-grid-bg {
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(var(--border) 1px, transparent 1px),
        linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 36px 36px;
    opacity: 0.5;
}

.scan-sweep {
    position: absolute;
    left: 0; right: 0;
    height: 1.5px;
    background: linear-gradient(90deg, transparent 0%, var(--accent) 50%, transparent 100%);
    animation: sweep 3s ease-in-out infinite;
}

@keyframes sweep {
    0% { top: 0%; opacity: 0; }
    8% { opacity: 1; }
    92% { opacity: 1; }
    100% { top: 100%; opacity: 0; }
}

.scan-corner {
    position: absolute;
    width: 14px;
    height: 14px;
    border-color: var(--accent);
    border-style: solid;
}
.sc-tl { top: 14px; left: 14px; border-width: 2px 0 0 2px; }
.sc-tr { top: 14px; right: 14px; border-width: 2px 2px 0 0; }
.sc-bl { bottom: 14px; left: 14px; border-width: 0 0 2px 2px; }
.sc-br { bottom: 14px; right: 14px; border-width: 0 2px 2px 0; }

.scan-status {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    position: relative;
    z-index: 2;
}

.section-divider {
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 1.5rem;
}

.section-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

.site-footer {
    margin-top: 3.5rem;
    padding: 1.5rem 0;
    border-top: 1.5px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.78rem;
    color: var(--muted);
}

.footer-name {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: var(--ink);
}

label.block span {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

footer { display: none !important; }
"""

# -----------------------------------------------
# Build the Gradio UI
# -----------------------------------------------
with gr.Blocks(
    css=custom_css,
    title="TruthLens",
    theme=gr.themes.Base(
        primary_hue="orange",
        neutral_hue="stone",
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("DM Mono"),
        text_size=gr.themes.sizes.text_md,
    ).set(
        body_background_fill="#f5f0e8",
        body_text_color="#0f0e0d",
        block_background_fill="#ede8dc",
        block_label_text_color="#7a7469",
        block_title_text_color="#0f0e0d",
        input_background_fill="#ede8dc",
        button_primary_background_fill="#0f0e0d",
        button_primary_text_color="#f5f0e8",
        button_primary_background_fill_hover="#c8392b",
    )
) as app:

    gr.HTML("""
    <div class="site-header">
        <div class="site-logo">
            <div class="logo-indicator"></div>
            TruthLens
        </div>
        <div style="font-size:0.78rem; color:var(--muted); letter-spacing:0.06em;">
            AI Media Verification
        </div>
    </div>

    <div class="hero-section">
        <div>
            <div class="hero-eyebrow">AI Media Detection</div>
            <h1 class="hero-title">Is it <em>real</em><br>or generated?</h1>
            <p class="hero-body">
                Upload any image or video. Our Vision Transformer model
                analyzes it in seconds and tells you whether it was created
                by a human or an AI system.
            </p>
        </div>
        <div class="scan-visual">
            <div class="scan-grid-bg"></div>
            <div class="scan-sweep"></div>
            <div class="scan-corner sc-tl"></div>
            <div class="scan-corner sc-tr"></div>
            <div class="scan-corner sc-bl"></div>
            <div class="scan-corner sc-br"></div>
            <span class="scan-status">Ready to scan</span>
        </div>
    </div>
    """)

    with gr.Tabs():

        with gr.TabItem("Image Detection"):
            gr.HTML('<div class="section-divider">Image Analysis</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Image",
                        elem_classes=["upload-region"],
                        height=280
                    )
                    image_btn = gr.Button("Analyze Image", variant="primary")
                with gr.Column(scale=1):
                    image_preview = gr.Image(
                        label="Preview",
                        height=160,
                        interactive=False
                    )
                    image_result = gr.Markdown(
                        value="Upload an image and click Analyze to see results.",
                        elem_classes=["result-panel"]
                    )
            gr.HTML("""
            <div class="tip-bar">
                Works best with JPG, PNG, and WebP images.
                Higher resolution images produce more accurate results.
            </div>
            """)
            image_btn.click(
                fn=analyze_image,
                inputs=[image_input],
                outputs=[image_preview, image_result],
                show_progress="full"
            )

        with gr.TabItem("Video Detection"):
            gr.HTML('<div class="section-divider">Video Analysis</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Upload Video",
                        elem_classes=["upload-region"],
                        height=280
                    )
                    video_btn = gr.Button("Analyze Video", variant="primary")
                with gr.Column(scale=1):
                    video_result = gr.Markdown(
                        value="Upload a video and click Analyze to see results.",
                        elem_classes=["result-panel"]
                    )
            gr.HTML("""
            <div class="tip-bar">
                Works with MP4, AVI, and MOV files under 60 seconds.
                We sample 10 frames evenly across the full duration.
            </div>
            """)
            video_btn.click(
                fn=analyze_video,
                inputs=[video_input],
                outputs=[video_result],
                show_progress="full"
            )

    gr.HTML("""
    <div class="stats-row">
        <div class="stat-cell">
            <span class="stat-number">94%</span>
            <span class="stat-desc">Detection accuracy</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number">&lt;3s</span>
            <span class="stat-desc">Analysis time</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number">ViT</span>
            <span class="stat-desc">Vision Transformer</span>
        </div>
    </div>

    <div class="site-footer">
        <span>TruthLens — AI Media Verification</span>
        <span class="footer-name">Built by Adarsh Pandey</span>
    </div>
    """)

# Single clean launch — works both locally and on Hugging Face
if __name__ == "__main__":
   App.launch()