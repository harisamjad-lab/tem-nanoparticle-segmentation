# TEM Nanoparticle Segmentation

This project performs **segmentation of nanoparticles in grayscale TEM images** using classical image processing techniques.

The goal is to identify **complete nanoparticles** while ignoring clipped particles at image boundaries.

---

## Method Pipeline

The segmentation pipeline uses:

1. Gaussian smoothing to reduce noise
2. CLAHE for contrast enhancement
3. Global Otsu thresholding
4. Morphological opening and closing
5. Distance transform
6. Watershed segmentation
7. Contour filtering to remove invalid particles

---

## Evaluation Metrics

Segmentation performance is evaluated using:

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall

---
