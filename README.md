# 🪪 KYC Document Quality Checker
### eSewa Hackathon 2026

A real-time KYC document quality detection system using Classical Computer Vision.
Detects **blur**, **glare**, and **framing issues** in passport/ID images.

---

## 📁 Project Structure

```
kyc-document-quality/
│
├── uk_password/                  ← Your dataset (place here)
│   └── UK/
│       └── {person_id}/
│           └── L{1-4}/B{1-4}/A{1-3}/
│               └── P_UK_{id}_L{l}_B{b}_A{a}_D{d}.jpg
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb   ← Understand & visualize dataset
│   ├── 02_blur_detection.ipynb        ← Blur detection deep dive
│   ├── 03_glare_detection.ipynb       ← Glare detection deep dive
│   ├── 04_framing_detection.ipynb     ← Framing check deep dive
│   └── 05_full_pipeline.ipynb         ← Run everything + CSV report
│
├── src/
│   ├── config.py                      ← All thresholds & settings
│   ├── blur.py                        ← Blur detection module
│   ├── glare.py                       ← Glare detection module
│   ├── framing.py                     ← Framing detection module
│   ├── parser.py                      ← Filename metadata parser
│   └── pipeline.py                    ← Full pipeline runner
│
├── outputs/
│   └── quality_report.csv             ← Generated after running pipeline
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/pipeline.py

# 3. Or explore notebooks step by step
jupyter notebook notebooks/
```

---

## 📊 Dataset Structure

| Variable | Options | Meaning |
|---|---|---|
| **L** | 1–4 | Lighting conditions |
| **B** | 1–4 | Background types |
| **A** | 1–3 | Camera angles |
| **D** | 1–2 | Distance from camera |

**5 people × 96 images each = 480 total images**

---

## 🔍 Quality Checks

| Check | Method | Threshold |
|---|---|---|
| **Blur** | Laplacian Variance | score < 100 = blurry |
| **Glare** | HSV V-channel | >10% bright pixels = glare |
| **Framing** | Canny + Contours | aspect ratio 1.3–2.0 = OK |

---

## 📅 Hackathon Timeline

- **Day 1** → Classical CV pipeline (this repo) ✅
- **Day 2** → YOLOv8 document detection + Streamlit dashboard
- **Submission** → GitHub + demo video
