"""
TEM Nanoparticle Segmentation Project
Image Processing 

Classical segmentation using:
Gaussian + CLAHE + Global Otsu Threshold + Distance Transform + Watershed

Muhammad Haris Amjad
Shahmir Javed
Hamza Khalid

"""

import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt



# SEGMENTATION CLASS

class TEMNanoparticleSegmenter:
    """
    Stable classical nanoparticle segmentation for TEM images
    """

    def __init__(self):
        self.gaussian_kernel = (5, 5)
        self.gaussian_sigma = 1.2

        self.min_particle_area = 120
        self.max_particle_area = 25000
        self.edge_margin = 8

    def preprocess_image(self, image):
        # Smooth noise
        blur = cv2.GaussianBlur(
            image, self.gaussian_kernel, self.gaussian_sigma
        )

        # Stronger contrast enhancement (important for your data)
        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(blur)

        return enhanced

    def segment_particles(self, image):
        pre = self.preprocess_image(image)

        # Global Otsu (particles are dark)
        _, binary = cv2.threshold(
            pre, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )

        # Remove salt noise
        opening = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN, kernel, iterations=2
        )

        # Close gaps inside particles
        closing = cv2.morphologyEx(
            opening, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Distance transform
        dist = cv2.distanceTransform(closing, cv2.DIST_L2, 5)

        # MUCH softer threshold
        _, sure_fg = cv2.threshold(
            dist, 0.25 * dist.max(), 255, 0
        )
        sure_fg = np.uint8(sure_fg)

        # Background
        sure_bg = cv2.dilate(closing, kernel, iterations=2)

        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Markers
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Watershed
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color, markers)

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[markers > 1] = 255

        return mask

    def is_particle_clipped(self, contour, shape):
        h, w = shape
        x, y, bw, bh = cv2.boundingRect(contour)

        return (
            x <= self.edge_margin or
            y <= self.edge_margin or
            x + bw >= w - self.edge_margin or
            y + bh >= h - self.edge_margin
        )

    def filter_particles(self, mask, shape):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered = np.zeros_like(mask)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < self.min_particle_area or area > self.max_particle_area:
                continue

            if self.is_particle_clipped(cnt, shape):
                continue

            cv2.drawContours(filtered, [cnt], -1, 255, -1)

        return filtered

    def segment_image(self, image):
        raw = self.segment_particles(image)
        final = self.filter_particles(raw, image.shape)
        return final



# EVALUATION CLASS


class SegmentationEvaluator:

    @staticmethod
    def dice(pred, gt):
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        inter = np.logical_and(pred, gt).sum()
        total = pred.sum() + gt.sum()

        return 2 * inter / total if total > 0 else 1.0

    @staticmethod
    def iou(pred, gt):
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()

        return inter / union if union > 0 else 1.0

    @staticmethod
    def precision_recall(pred, gt):
        pred = (pred > 0).astype(np.uint8)
        gt = (gt > 0).astype(np.uint8)

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall



# VISUALIZATION


def visualize(image, gt, pred, dice, iou, precision, recall):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Original")
    ax[0, 0].axis("off")

    ax[0, 1].imshow(gt, cmap="gray")
    ax[0, 1].set_title(f"Ground Truth\nDice: {dice:.3f}")
    ax[0, 1].axis("off")

    ax[1, 0].imshow(pred, cmap="gray")
    ax[1, 0].set_title(f"Prediction\nIoU: {iou:.3f}")
    ax[1, 0].axis("off")

    overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
    overlay[:, :, 1] = (gt > 0) * 255     # Green = GT
    overlay[:, :, 0] = (pred > 0) * 255   # Red = Prediction

    ax[1, 1].imshow(overlay)
    ax[1, 1].set_title(
        f"Overlay\nPrecision: {precision:.3f} | Recall: {recall:.3f}"
    )
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.show()



# DATASET PROCESSING

def process_dataset(images_folder, masks_folder):
    segmenter = TEMNanoparticleSegmenter()
    evaluator = SegmentationEvaluator()

    image_files = sorted(
        glob.glob(os.path.join(images_folder, "*.jpg")) +
        glob.glob(os.path.join(images_folder, "*.png"))
    )

    dice_scores, iou_scores = [], []
    precision_scores, recall_scores = [], []

    for idx, img_path in enumerate(image_files):
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        name = os.path.basename(img_path)
        mask_path = os.path.join(masks_folder, name)
        if not os.path.exists(mask_path):
            continue

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(np.uint8) * 255

        pred = segmenter.segment_image(image)

        dice = evaluator.dice(pred, gt)
        iou = evaluator.iou(pred, gt)
        precision, recall = evaluator.precision_recall(pred, gt)

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)

        print(
            f"{idx+1:03d} | Dice: {dice:.3f} | IoU: {iou:.3f} "
            f"| precision: {precision:.3f} | recall: {recall:.3f}"
        )

        if idx < 4:
            visualize(image, gt, pred, dice, iou, precision, recall)

    print("\n================ SUMMARY ================")
    print(f"Images processed: {len(dice_scores)}")
    print(f"Mean Dice:      {np.mean(dice_scores):.4f}")
    print(f"Mean IoU:       {np.mean(iou_scores):.4f}")
    print(f"Mean Precision: {np.mean(precision_scores):.4f}")
    print(f"Mean Recall:    {np.mean(recall_scores):.4f}")
    print("========================================")



# MAIN


def main():
    images_folder = r"C:\Users\LENOVO\Documents\Health AI Study\M2 SIA Image Processing\Image processing Project\Image_Processing_Project\TEST SET\TEST SET\Images"
    masks_folder = r"C:\Users\LENOVO\Documents\Health AI Study\M2 SIA Image Processing\Image processing Project\Image_Processing_Project\TEST SET\TEST SET\Masks"

    process_dataset(images_folder, masks_folder)


if __name__ == "__main__":
    main()
