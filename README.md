# License Plate Recognition using YOLOv8 and EasyOCR

This project aims to demonstrate license plate recognition utilizing YOLOv8 for object detection and EasyOCR for text recognition.

## Objective

The primary objective is to detect license plates in images and extract the text from those plates using two different deep learning models: YOLOv8 for object detection and EasyOCR for text recognition.

## Setup

### Requirements

- Python 3.x
- PyTorch
- OpenCV
- EasyOCR
- YOLOv8 weights and configuration files

### Installation

1. Install the required Python libraries:

    ```bash
    pip install torch torchvision opencv-python easyocr
    ```

2. Download the YOLOv8 weights and configuration files from [link_to_yolov8_repo].

## Usage

### This a colab link : 
https://colab.research.google.com/drive/1cQsT5GpZj6sw87J0_bVmJbMmELe1xQfN?usp=sharing)https://colab.research.google.com/drive/1cQsT5GpZj6sw87J0_bVmJbMmELe1xQfN?usp=sharing

# License Plate Recognition using YOLOv8 and EasyOCR

## Technical Explanation

This project involves two main components: license plate detection and text recognition.

### 1. License Plate Detection with YOLOv8

#### YOLOv8 Model Overview

- **YOLO (You Only Look Once):** YOLOv8 is an object detection model known for its real-time inference capabilities.
- **Architecture:** YOLOv8 uses a deep neural network architecture to detect various objects within an image.
- **Training:** Pre-trained on a large dataset, YOLOv8 can recognize common objects, including license plates, after fine-tuning or using transfer learning.

#### License Plate Detection Process

1. **Input Image:** An image containing vehicles with license plates serves as the input.
2. **Preprocessing:** The input image undergoes preprocessing, such as resizing or normalization, to fit the model's input requirements.
3. **YOLOv8 Inference:** The YOLOv8 model performs inference on the preprocessed image.
4. **Bounding Box Generation:** YOLOv8 generates bounding boxes around detected objects, including license plates, indicating their locations within the image.
5. **Post-processing:** Extracted bounding box coordinates are obtained to isolate the regions of interest (license plates) for further processing.

### 2. Text Recognition with EasyOCR

#### EasyOCR Overview

- **Optical Character Recognition:** EasyOCR is an OCR library capable of recognizing text from images.
- **Architecture:** It employs deep learning techniques, such as CNNs (Convolutional Neural Networks) and RNNs (Recurrent Neural Networks), to recognize text.
- **Language Support:** EasyOCR supports multiple languages and can accurately recognize text from various fonts and styles.

#### Text Recognition Process

1. **Bounding Box Coordinates:** Bounding box coordinates, obtained from YOLOv8, indicate the regions containing license plates.
2. **Extracted Regions:** Extract the regions of interest (license plates) using the bounding box coordinates.
3. **Input to EasyOCR:** The isolated license plate regions are fed as input to the EasyOCR library.
4. **Text Extraction:** EasyOCR performs text recognition on the input regions, extracting the text from the license plates.
5. **Output:** Recognized text is obtained as output, representing the characters present on the license plates.

## Implementation Details

- **Python Scripts:** Separate Python scripts are written to handle YOLOv8 object detection and EasyOCR text recognition.
- **Data Flow:** Data flows from the YOLOv8 detection step to the EasyOCR recognition step, utilizing extracted bounding box coordinates.
- **Dependency Management:** Dependencies such as PyTorch, OpenCV, EasyOCR, and the YOLOv8 model files are required for the project's execution.

## References and Resources

- YOLOv8 Repository: https://github.com/ultralytics/ultralytics
- EasyOCR Documentation: https://github.com/JaidedAI/EasyOCR


