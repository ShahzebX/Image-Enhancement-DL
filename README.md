# Project Title

AI Image Enhancement Studio: a lightweight web app that colorizes grayscale photos and improves low-light images for practical restoration and readability.

---

## Overview

Image restoration tasks like colorizing legacy black-and-white photos and enhancing underexposed images are common in consumer media, archives, surveillance review, and dataset preparation. This repository packages two complementary inference pipelines behind a simple web UI: automatic colorization for grayscale inputs and brightness/contrast enhancement for low-light scenes.

---

## Problem Statement

Both tasks are visually underconstrained:

- **Colorization:** multiple plausible colors can fit the same luminance structure. Naive approaches (e.g., tinting, histogram tricks) cannot recover realistic chroma and often produce flat or inconsistent colors.
- **Low-light enhancement:** dark regions suffer from low signal-to-noise ratio. Simple global exposure changes typically amplify noise, clip highlights, and wash out color.

The goal is to produce *visually plausible* outputs with minimal user input and predictable runtime.

---

## Approach

This repository ships a unified inference pipeline exposed via a Flask app.

### Preprocessing

- **Colorization:** convert input to CIE LAB; extract the L channel; resize L to `224×224`; apply mean shift (`L -= 50`) as expected by the pre-trained network.
- **Low-light enhancement:** operate primarily in LAB/HSV color spaces to separate luminance from chroma.

### Model inference

- **Colorization model:** OpenCV DNN loads the pre-trained Caffe model from Zhang et al. (ECCV 2016) and predicts `ab` chroma channels conditioned on the `L` channel.
- **Enhancement model:** A **MIRNet-inspired** lightweight enhancer implemented with classical CV operations (`CLAHE + denoise + gamma + mild sharpening`) for efficiency.
	- A TensorFlow-based deep model hook exists as a *placeholder* in `mirnet_model.py`, but the default path in the Flask app uses the classical enhancer (`use_deep_learning=False`) to ensure near real-time performance on CPU.

### Postprocessing

- **Colorization:** resize predicted `ab` to original resolution; concatenate with the original-resolution `L`; convert LAB → BGR; clamp to valid range; export JPEG.
- **Enhancement:** convert back to BGR and return the enhanced image.

---

## Experiments & Model Selection

This project reflects an applied iteration cycle focused on robustness on real images rather than narrowly optimized benchmark results.

- **Initial attempt: U-Net colorization (prototype)**
	- **Issue:** outputs tended toward desaturated/“sepia” tones.
	- **Why:** pixel-wise L2 loss encourages averaging across multiple valid color modes, collapsing chroma.

- **Initial attempt: Zero-DCE for low-light enhancement (prototype)**
	- **Performance ceiling:** improvements plateaued without consistently improving perceptual quality.
	- **Generalization gap:** degraded performance on “in-the-wild” images outside the LOL dataset distribution.
	- **Artifacts:** noticeable noise amplification and instability in extremely dark regions.

- **Final choice: MIRNet-style enhancement strategy**
	- MIRNet’s multi-scale design is a strong fit for low-light enhancement in principle; in this repo the shipped implementation is a **fast, deterministic** approximation designed for reliable inference in a web app.
	- The resulting pipeline is stable on a wide range of inputs and avoids training-time dependencies for deployment.

---

## Dataset

This repository is **inference-focused** and does not ship a training dataset.

- **Colorization:** uses a pre-trained model originally trained on large-scale natural images (per the Zhang et al. colorization project).
- **Low-light enhancement (experiments):** development referenced paired low-light datasets such as **LOL** and additional “in-the-wild” samples to sanity-check generalization.
- **Preprocessing/filters in this repo:**
	- Input validation (JPG/PNG, ≤10MB) in the Flask app.
	- Color-space conversions (BGR↔RGB↔LAB/HSV) depending on task.

**Limitations:** no dataset is included, and there is no standardized evaluation split or reproducible training pipeline in this codebase.

---

## Results

This project does not include an automated metrics report (e.g., PSNR/SSIM/LPIPS) in the repository, so results are described qualitatively.

- **Colorization:** produces plausible colors for common objects and skin tones, with global color consistency on many portraits and everyday scenes.
- **Low-light enhancement:** improves visibility of midtones and local contrast while generally preserving edges and avoiding aggressive overexposure.

Expected behavior in practice:

- Colorization outputs are *reasonable guesses*, not ground truth.
- Enhancement may increase perceived noise in very dark regions (a common tradeoff when boosting luminance).

---

## Challenges & Limitations

- **Color ambiguity:** the same grayscale image can map to multiple valid colorizations; the model may pick implausible hues.
- **Domain shift:** performance drops on unusual content (medical imagery, cartoons, infrared, documents).
- **Failure modes:** small faces, rare objects, and low-texture regions can yield color bleeding or muted chroma.
- **Low-light edge cases:** severe underexposure can still show noise/grain after enhancement; highlights may clip on already-bright images.
- **Model naming:** the enhancement module is MIRNet-*inspired*; the default implementation in this repo is classical (not a trained MIRNet checkpoint).
- **Training not included:** there are no end-to-end training scripts or reproducible experiment logs in the current repository.

---

## Tech Stack

- **Python**
- **OpenCV DNN** (Caffe model inference for colorization)
- **NumPy**
- **Flask** (unified web UI + API)
- **Pillow (PIL)** (image I/O in the Flask app)

Model architectures used:

- **Zhang et al. (ECCV 2016) colorization model** (Caffe)
- **MIRNet-inspired enhancement pipeline** (classical CV implementation; optional TensorFlow hook)

---

## Repository Structure

- `app.py` — Unified Flask server: `/process` endpoint for colorization or enhancement (formerly `enhanced_app.py`).
- `mirnet_model.py` — Low-light enhancer implementation (classical default) + optional deep-model placeholder.
- `test_color.py` — Minimal script that runs colorization and writes `result.jpg`.
- `templates/index.html` — Frontend page for the Flask app.
- `static/styles.css`, `static/script.js` — Frontend assets.
- `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, `pts_in_hull.npy` — Colorization model files.
- `pretrained.ipynb` — Notebook showing how to run colorization in Colab.
- `MIRNet.ipynb` — Notebook stub for MIRNet-oriented enhancement experimentation.

---

## How to Run

### 1) Environment setup

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_enhanced.txt
```

### 2) Ensure model files are present

The following files must be in the project root:

- `colorization_deploy_v2.prototxt`
- `pts_in_hull.npy`
- `colorization_release_v2.caffemodel`

Upstream sources:

- Prototxt: https://github.com/richzhang/colorization/tree/caffe/colorization/models
- `pts_in_hull.npy`: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
- Caffe weights (as referenced by this repo): https://www.dropbox.com/scl/fi/d8zffur3wmd4wet58dp9x/colorization_release_v2.caffemodel?rlkey=iippu6vtsrox3pxkeohcuh4oy&dl=0

### 3) Run inference

Start the Flask web app:

```bash
python app.py
```

Open `http://localhost:5000`, choose a mode, upload an image, and download the result.

Or run the minimal test script (writes `result.jpg`):

```bash
python test_color.py
```

### (Optional) Train the model

Training code is not included in this repository.

---

## Future Improvements

- Replace the classical enhancement path with an actual trained MIRNet (or equivalent) checkpoint and add model/version management.
- Add an evaluation harness (fixed test set + PSNR/SSIM/LPIPS + runtime benchmarks).
- Expand dataset coverage and document curation steps for “in-the-wild” robustness.
- Improve colorization realism with user hints (scribbles/points) or confidence-aware chroma blending.
- Add batching, caching, and GPU acceleration options for deployment.

---

## Author

Muhammad Shahzeb — https://github.com/ShahzebX
Sameer Ali - https://github.com
Sohail Ahmed - https://github.com
