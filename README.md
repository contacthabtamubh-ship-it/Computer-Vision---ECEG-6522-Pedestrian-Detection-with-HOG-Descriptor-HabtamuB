# Computer-Vision-ECEG-6522-Pedestrian-Detection-with-HOG-Descriptor-HabtamuB
A Python implementation of pedestrian detection using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier, using OpenCV's pre-trained detector.
# Pedestrian Detection with HOG + SVM

## Setup
```bash
pip install -r requirements.txt

# Pedestrian Detection with HOG + SVM

A Python implementation of pedestrian detection using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier, using OpenCV's pre-trained detector.

## Results
- **Dataset:** INRIA Person Dataset (288 test images)
- **Recall:** 79.51%
- **Average time per image:** 0.059 seconds (~17 FPS)

## Quick Start
```bash
git clone <your-repo-url>
cd pedestrian-detection-hog-svm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/detect.py --mode evaluate --image_folder data/INRIA_Person_Dataset/Test/JPEGImages
