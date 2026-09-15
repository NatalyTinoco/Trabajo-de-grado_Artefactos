# Endoscopic Artifact Detection 🩺

Automatic method for detecting **specular reflections** and **motion blur** in endoscopic images, using **complementary binary classifiers** on HSV color features.

**Undergraduate thesis** · Bioengineering, Universidad El Bosque (2015–2019)
📄 Published at **ICPRS 2021 (IET/IEEE)** · Meritorious thesis

## 👥 Authors

Nataly Tinoco Gómez · Daniela Díaz

## 🎯 The problem

Artifacts in endoscopy images (colonoscopy, gastroscopy, airway endoscopy) degrade quality and hinder computer-aided diagnosis (CADx). Two types are detected:

- **RE** — Specular reflection
- **DM** — Motion blur

## ⚙️ Approach

- Color feature extraction in the **HSV** space (hue, saturation, value)
- **Spectral coherence** analysis to distinguish specular reflections
- **Complementary binary classifiers** for each artifact type
- **96% accuracy** on specular reflection detection
- Graphical interface (`Artefactose.py`) to visualize and process images

## 📁 Contents

| Folder / file | Description |
|---|---|
| `Artefactose.py` | Main graphical interface |
| `CutDM/` | Motion-blur image cropping |
| `Caracteristicas/` | HSV feature extraction (DM / RE / NO) |
| `corrlación.py` | Feature correlation analysis |

## 🛠️ Tech stack

Python · OpenCV · scikit-image · scikit-learn · MATLAB

## 📄 Publication

*"Automatic method for detecting specular reflection and motion blur artifacts on endoscopic images using complementary binary classifiers"* — ICPRS 2021, IET/IEEE.
