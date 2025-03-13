# Leaf-Disease-Classification
# Leaf Disease Classification Using ResNet50

A deep learning web application built with PyTorch and FastAPI that allows users to classify plant leaf diseases using a fine-tuned ResNet50 model.

## Overview

This project implements an automated plant disease detection system using a custom-trained ResNet50 model. The model has been trained on the PlantVillage dataset to identify various leaf diseases across different plant species. The application provides an intuitive web interface where users can upload images of plant leaves to receive instant disease classification results.

## Features

- **Advanced Image Preprocessing Pipeline**: Includes grayscale conversion, Gaussian smoothing, Otsu's thresholding, and morphological transformations
- **Fine-tuned ResNet50 Architecture**: Pre-trained on ImageNet and customized for leaf disease classification
- **Regularization Techniques**: Implements dropout, early stopping, and learning rate scheduling to prevent overfitting
- **Interactive Web Interface**: Allows users to upload and analyze leaf images through an intuitive UI
- **Real-time Classification**: Delivers instant disease identification with confidence scores
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## Tech Stack

- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning**: ResNet50, PyTorch
- **Image Processing**: OpenCV, PIL
- **Deployment**: Docker (optional)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for inference speed)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/leaf-disease-classification.git
   cd leaf-disease-classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained model (or train your own following the instructions in the Training section):
   ```bash
   # If you have a script to download the model
   python download_model.py
   
   # Or manually place your model files in the correct directory:
   # - leaf_disease_model_final.pth
   # - class_mapping.pth
   ```

## Usage

### Starting the Server

1. Start the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

### Using the Web Interface

1. Click the "Upload Image" button to select a leaf image from your device
2. The application will process the image and display:
   - The detected disease class
   - Confidence score
   - Brief description of the disease
   - Recommended treatment options

## Model Architecture

The disease classification model is based on ResNet50 with the following customizations:

- First 6 layer groups frozen to leverage pre-trained features
- Custom fully connected layer with dropout (0.4) for regularization
- Adam optimizer with weight decay (1e-4)
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting

## Training Your Own Model

The model training script is provided in `train_model.py`. To train your own model:

1. Prepare your dataset in the following structure:
   ```
   PlantVillage/
   ├── train/
   │   ├── class_1/
   │   │   ├── img1.jpg
   │   │   ├── img2.jpg
   │   │   └── ...
   │   ├── class_2/
   │   └── ...
   └── val/
       ├── class_1/
       ├── class_2/
       └── ...
   ```

2. Run the training script:
   ```bash
   python train_model.py
   ```

3. The script will save:
   - Best model weights (`best_leaf_model.pth`)
   - Final model weights (`leaf_disease_model_final.pth`)
   - Class mapping dictionary (`class_mapping.pth`)

## API Endpoints

### `POST /predict`

Accepts an image file and returns disease classification results.

**Request**:
- Form data with an image file

**Response**:
```json
{
  "class_name": "Tomato_Late_blight",
  "confidence": 0.932,
  "description": "Late blight is a fungal disease that affects tomato plants...",
  "recommendations": [
    "Remove and destroy infected plant parts",
    "Apply fungicide as directed by a professional",
    "Ensure proper spacing between plants for airflow"
  ]
}
```

### `GET /classes`

Returns all available disease classes the model can identify.

**Response**:
```json
{
  "classes": [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_apple_rust",
    "Apple_healthy",
    "..."
  ]
}
```

## Performance

The model achieves the following performance metrics on the validation set:
- **Accuracy**: XX% (fill in with your model's performance)
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%

## Future Improvements

- Implement more advanced data augmentation techniques
- Add support for more plant species and diseases
- Integrate with a mobile application
- Add multi-language support
- Implement explainable AI features to highlight affected areas on the leaf

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The PlantVillage dataset
- PyTorch and FastAPI communities
- ResNet model developers
