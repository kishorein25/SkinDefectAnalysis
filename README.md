# DermVISION (SkinDefectAnalysis)

AI-powered skin disease detection with dual-model verification, region validation, and local LLM chat.

## Overview

DermVISION is a complete skin analysis system combining:
- **Image Diagnosis** - Upload a skin photo, select body region, get AI diagnosis with confidence scores
- **Dual-Model Verification** - DINOv2 + ViT consensus for reliable predictions
- **Specialized 23-Class Model** - MobileNetV2 fine-tuned for common conditions (acne, hyperpigmentation, eczema, etc.)
- **Region Validation** - MediaPipe face/hand/pose detection ensures image matches selected body part
- **Doctor Finder** - Nearby dermatologists via Google Places API + OpenStreetMap fallback
- **Local AI Chat** - Privacy-preserving skin Q&A using Ollama (LLaMA) + RAG over medical knowledge base

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  User Uploads   │────▶│  Region Check    │────▶│  Dual Models       │
│  Image + Region │     │  (MediaPipe)     │     │  (DINOv2 + ViT)    │
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
                                                           │
                                              ┌────────────▼────────────┐
                                              │  Consensus Logic      │
                                              │  + MobileNet Override │
                                              └───────────┬───────────┘
                                                          │
                              ┌───────────────────────────┼───────────────────────────┐
                              ▼                           ▼                           ▼
                       ┌─────────────┐             ┌───────────────┐           ┌───────────────┐
                       │ Disease KB  │             │ Treatment KB  │           │ Doctor Finder │
                       │ (50+ cond.) │             │ (tiered Rx)   │           │ (Places/OSM)  │
                       └─────────────┘             └───────────────┘           └───────────────┘
                              │                           │                           │
                              └───────────────────────────┼───────────────────────────┘
                                                          ▼
                                                 ┌─────────────────┐
                                                 │  JSON Response  │
                                                 │  (UI renders)   │
                                                 └─────────────────┘
```

## Project Structure

```
skin-disease-detector/
├── diagnose.py                 # Main Flask app (port 5000) - serves UI + /diagnose + /chat
├── requirements.txt            # Python dependencies
├── config.json                 # Google Maps API key (set via env var GOOGLE_MAPS_API_KEY)
├── .gitignore
├── core/
│   ├── __init__.py
│   ├── disease_detector.py     # Dual HF models (DINOv2 + ViT) + consensus logic
│   ├── mobilenet_detector.py   # 23-class MobileNetV2 for common conditions
│   ├── region_detector.py      # MediaPipe face/hand/pose region validation
│   ├── doctor_finder.py        # Google Places + OSM Overpass fallback
│   └── chat_engine.py          # Ollama LLM + embeddings RAG
├── data/
│   ├── diseases.json           # 50+ conditions: symptoms, causes, severity, regions
│   ├── treatments.json         # Tiered treatments (mild/moderate/severe) + specialist
│   ├── foods.json              # Eat/avoid lists per condition
│   ├── occurrence.json         # Epidemiology: prevalence, age, gender, risk factors
│   ├── contagious.json         # Contagion status + prevention
│   ├── region_disease_map.json # Region-specific disease lists + image guidance
│   └── medical_kb.json         # RAG knowledge base for chat (Q&A pairs)
└── static/
    └── index.html              # Single-file frontend (HTML/CSS/JS embedded)
```

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed and running (`ollama serve`) with `llama3.2:1b` and `mxbai-embed-large` models
- (Optional) Google Maps Places API key for doctor finder

### Installation

```bash
git clone https://github.com/kishorein25/SkinDefectAnalysis.git
cd SkinDefectAnalysis
pip install -r requirements.txt

# Pull Ollama models (in separate terminal)
ollama pull llama3.2:1b
ollama pull mxbai-embed-large
ollama serve
```

### Run

```bash
# Terminal 1: Start diagnosis server (port 5000)
python diagnose.py

# Open http://localhost:5000
```

Both **Diagnose** and **Chat** tabs work on the same port.

## Usage

### Image Diagnosis
1. Select body region (Face, Hand, Leg, Foot, Scalp, Back, Whole Body)
2. Optionally enter your city for nearby doctor suggestions
3. Upload a clear skin image
4. Click **Analyze** - runs dual-model + MobileNet inference
5. View diagnosis with:
   - Disease name, confidence, severity
   - Dual-model agreement status
   - MobileNet 23-class result (overrides for acne/hyperpigmentation)
   - Description, symptoms, occurrence stats
   - Contagion info + prevention
   - Tiered treatments (mild/moderate/severe)
   - Foods to eat/avoid
   - Nearby dermatologists (if location provided)
   - Medical disclaimer + emergency warning

### AI Chat
1. Switch to **Chat** tab
2. Ask questions like:
   - "What causes acne and how to treat it?"
   - "Difference between eczema and psoriasis?"
   - "Foods for healthy skin?"
   - "When should I see a dermatologist?"
3. Answers generated from local LLM + medical knowledge base (RAG)

## Models

| Model | Classes | Purpose |
|-------|---------|---------|
| `Jayanth2002/dinov2-base-finetuned-SkinDisease` | 31 | Primary classifier (DINOv2) |
| `Jayanth2002/vit_base_patch16_224-finetuned-SkinDisease` | 31 | Secondary classifier (ViT) |
| `models/mobilenet_skin23.pt` | 23 | Common conditions specialist |

**Consensus Logic**: Both models must agree on top-1 label for high confidence. If they disagree, confidence is weighted toward primary. MobileNet overrides for acne/hyperpigmentation when confidence ≥ 0.5 and condition fits selected region.

## Region Validation

MediaPipe detectors verify uploaded image matches selected region:
- **Face**: FaceDetection
- **Hand**: HandLandmarks
- **Leg/Foot/Back/Scalp/Whole Body**: PoseLandmarks + visibility scoring

Rejects mismatched uploads with descriptive error (e.g., "This image contains a HAND, not a face").

## Data Files

All clinical data in `data/*.json`:
- **diseases.json** - 50+ conditions with metadata
- **treatments.json** - Evidence-based tiered treatments
- **foods.json** - Nutritional guidance per condition
- **occurrence.json** - Epidemiology statistics
- **contagious.json** - Transmission + prevention
- **region_disease_map.json** - Region-condition mapping
- **medical_kb.json** - 25 Q&A pairs for RAG chat

## Configuration

```json
// config.json
{
  "google_maps_api_key": ""
}
```

Set via environment variable for production:
```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
python diagnose.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves `static/index.html` |
| `/diagnose` | POST | Image diagnosis (multipart: image, region, location) |
| `/chat` | POST | AI chat (JSON: `{question: string}`) |
| `/normal-method` | GET | Health check info |

## Response Format (Diagnose)

```json
{
  "success": true,
  "normal": false,
  "disease_name": "Acne Vulgaris",
  "confidence": 0.87,
  "primary": "acne_and_rosacea",
  "primary_conf": 0.89,
  "secondary": "acne_and_rosacea",
  "secondary_conf": 0.85,
  "agreed": true,
  "model_basis": "new_model",
  "mobilenet": {"display": "Acne & Rosacea", "std_key": "acne", "confidence": 0.92},
  "description": "...",
  "symptoms": ["...", "..."],
  "severity": "Common - treatable",
  "occurrence": {...},
  "contagious": {...},
  "treatments": {...},
  "foods": {...},
  "doctors": [...],
  "disclaimer": "...",
  "emergency_warning": "..."
}
```

## Privacy

- **No cloud inference** - All models run locally
- **Chat uses local Ollama** - No data leaves your machine
- **Images processed in-memory** - Not persisted (uploads folder only for temp processing)
- **Doctor finder** - Only location string sent to Google/OSM APIs

## Limitations

- Not a medical device - for informational purposes only
- Model accuracy varies by condition and image quality
- Region validation requires visible anatomical landmarks
- Doctor finder depends on external API availability
- Ollama must be running locally for chat

## Disclaimer

> This diagnosis is generated by an AI model and is for informational purposes only. It is NOT a substitute for professional medical advice. Always consult a qualified dermatologist or healthcare professional for proper diagnosis and treatment.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this work in research, please cite:
```bibtex
@misc{dermvision2024,
  title={DermVISION: Dual-Model Skin Disease Detection with Region Validation and Local LLM Chat},
  author={Kishore, ...},
  year={2024},
  url={https://github.com/kishorein25/SkinDefectAnalysis}
}
```