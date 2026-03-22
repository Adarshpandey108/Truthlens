# TruthLens — AI Media Detector

I built this because AI-generated images and videos are everywhere now, and it's getting harder to tell what's real. TruthLens lets you upload any image or video and get an instant verdict on whether it was created by an AI or a real human.

## What it does

Upload an image or video and TruthLens will analyze it using a Vision Transformer model and tell you whether it's AI-generated or authentic, along with a confidence score and a detailed breakdown of why.

For images, it analyzes the full frame in one pass. For videos, it samples 10 frames evenly across the entire duration, runs each one through the detector, and gives you both an overall verdict and a frame-by-frame breakdown table so you can see exactly where the AI patterns were detected.

## Live demo

Try it here: [https://huggingface.co/spaces/adarsh108/Truthlens](https://huggingface.co/spaces/adarsh108/Truthlens)

## How it works

The core of TruthLens is a fine-tuned Vision Transformer (ViT) from Hugging Face — `umm-maybe/AI-image-detector`. Vision Transformers work by splitting an image into patches and analyzing the relationships between those patches using attention mechanisms. AI-generated images tend to have subtle but consistent statistical patterns in their textures and high-frequency details that real photographs don't have, and the ViT is specifically trained to detect those patterns.

For video detection, the pipeline works like this:

1. OpenCV opens the video and calculates the total frame count
2. We calculate a sampling interval so we get 10 frames evenly spread across the full duration
3. Each sampled frame is converted from BGR to RGB and passed through the image detector
4. All frame-level scores are aggregated — if more than half the frames are flagged as AI, the overall verdict is AI-generated
5. The result shows both the aggregate confidence and a per-frame breakdown table

## Tech stack

- **Gradio** — the web interface and file upload handling
- **Transformers** — loads and runs the Vision Transformer model from HuggingFace
- **PyTorch** — the deep learning backend that runs the model
- **OpenCV** — video loading, frame extraction, and color space conversion
- **Pillow** — image preprocessing before passing to the model
- **Hugging Face Spaces** — free hosting with GPU-optional inference

## Running locally

Clone the repo and set up a virtual environment.

```bash
git clone https://github.com/Adarshpandey108/Truthlens.git
cd Truthlens
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

Install dependencies.

```bash
pip install gradio transformers torch torchvision pillow opencv-python-headless numpy
```

Run the app.

```bash
python app.py
```

Open your browser at `http://localhost:7860`. The first run will download the model weights (~350MB) from Hugging Face — this only happens once.

## Project structure

```
Truthlens/
├── app.py              # full application — model loading, detection logic, UI
├── requirements.txt    # dependencies for deployment
└── README.md
```

## What I learned building this

Video detection is essentially a solved problem once you have a good image detector — the trick is in the frame sampling strategy. Sampling too few frames misses short AI-generated clips. Sampling too many makes it slow. 10 evenly-spaced frames turned out to be a good balance for videos under 60 seconds.

The Vision Transformer approach is significantly more accurate than older CNN-based detectors for this task because attention mechanisms can capture long-range dependencies across the image — exactly the kind of global statistical consistency that AI generators produce but real cameras don't.

## Limitations

No AI detector is perfect. Heavily compressed videos, low-resolution images, and images that have been screenshot or re-saved multiple times will reduce accuracy. This tool is meant as a guide, not a definitive verdict.

## What's next

- Add support for detecting specific AI generators (Midjourney vs DALL-E vs Stable Diffusion)
- Show a heatmap highlighting which regions of the image triggered the detection
- Add batch upload so multiple images can be checked at once
- Improve video analysis with temporal consistency checking across frames

---

Built by Adarsh Pandey
