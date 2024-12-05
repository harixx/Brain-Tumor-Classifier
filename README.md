# Brain Tumor Classification Project

## 🧠 Project Overview

This project develops an advanced deep learning system for classifying brain tumors from MRI scans, aimed at supporting medical professionals in early detection and diagnosis. The system combines state-of-the-art deep learning architectures with interpretability features to ensure both accuracy and transparency in medical decision-making.

### Core Objectives

-   Achieve highly accurate brain tumor classification from MRI scans
-   Provide transparent, interpretable predictions to support medical decisions
-   Create an accessible, user-friendly interface for medical professionals
-   Maintain ethical considerations in medical AI deployment

## 📊 Dataset Characteristics

The project utilizes a comprehensive dataset of 6,523 MRI scans, carefully curated to represent different tumor types:

| Tumor Type | Training Samples | Testing Samples | Total |
|------------|------------------|-----------------|-------|
| Glioma     | 1,400            | 350             | 1,750 |
| Meningioma | 1,300            | 325             | 1,625 |
| Pituitary  | 1,200            | 300             | 1,500 |
| No Tumor   | 1,812            | 453             | 2,265 |

## 🏗 Project Structure

```
brain-tumor-classification/
│
├── data/
│   ├── training/
│   └── testing/
│
├── models/
│   ├── xception_model.h5
│   ├── mobilenet_model.h5
│   └── cnn_model.h5
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── model_architecture.py
│   └── streamlit_app.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── training_metrics.png
│   └── saliency_maps/
│
└── README.md
```

## 🤖 Model Architecture

### 1. Xception (Primary Model)

-   Architecture: Pre-trained Xception with custom classification layers
-   Parameters: 21 million
-   Performance:
    -   Accuracy: 99%
    -   Precision: 98.5%
    -   Recall: 99%
-   Best suited for research labs and high-performance computing environments

### 2. MobileNet

-   Lightweight architecture optimized for speed
-   Performance:
    -   Accuracy: 99%
    -   Precision: 98.7%
    -   Recall: 98.9%
-   Ideal for resource-constrained environments and edge deployment

### 3. Custom CNN

-   Architecture: 4 Convolutional Blocks
-   Parameters: 4.7 million
-   Performance:
    -   Accuracy: 91%
    -   Precision: 90%
    -   Recall: 91%
-   Specialized for brain tumor classification tasks

## 📝 Methodology

### Data Preprocessing

1.  Image Standardization
    1.  Resize all images to 299x299 pixels
    2.  Normalize pixel values to range [0,1]
    3.  Apply data augmentation techniques
        1.  Random brightness adjustment
        2.  Rotation and flipping
        3.  Contrast modification
2.  Dataset Organization
    1.  Split data into training (70%), validation (15%), and testing (15%) sets
    2.  Implement stratified sampling to maintain class distribution

### Model Training

1.  Transfer Learning Approach
    1.  Fine-tune pre-trained models on our dataset
    2.  Implement progressive learning rates
    3.  Apply early stopping to prevent overfitting
2.  Custom Model Training
    1.  Train from scratch with optimized architecture
    2.  Implement batch normalization
    3.  Use dropout for regularization

## 🔍 Key Technical Innovations

### Interpretability Features

1.  Saliency Map Generation
    1.  Highlights regions of interest in MRI scans
    2.  Provides visual explanation of model decisions
    3.  Helps build trust with medical professionals
2.  AI-Powered Explanations
    1.  Integration with Google Gemini 1.5 Flash
    2.  Generates context-aware medical insights
    3.  Provides natural language explanations of predictions

## 🚀 Deployment

### Streamlit Web Application

-   User-friendly interface for MRI scan upload
-   Real-time prediction with multiple models
-   Interactive visualization of results
-   Built-in LLM chat support for result interpretation

### Installation Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/brain-tumor-classification.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run src/streamlit_app.py
```

## 🔬 Limitations and Future Work

### Current Limitations

1.  Dataset Diversity
    1.  Limited representation of rare tumor types
    2.  Potential demographic biases
    3.  Variation in MRI scan quality
2.  Technical Constraints
    1.  High computational requirements for Xception model
    2.  Real-time processing challenges
    3.  Integration with existing medical systems

### Future Development Plans

1.  Technical Enhancements
    1.  Implement ensemble learning techniques
    2.  Develop real-time diagnostic capabilities
    3.  Expand model interpretability features
2.  Clinical Integration
    1.  Integrate with hospital information systems
    2.  Develop API for third-party medical software
    3.  Implement HIPAA-compliant data handling

## 📚 Ethical Considerations

1.  Medical Decision Support
    1.  System designed to augment, not replace, medical expertise
    2.  Clear communication of model confidence levels
    3.  Regular validation of model performance
2.  Patient Privacy
    1.  HIPAA-compliant data handling
    2.  Secure storage and transmission
    3.  Anonymous processing of patient data

## 📬 Contact Information

-   Project Lead: Muhammad Haris Salman
-   Email: Muhammadharissalman@gmail.com
-   LinkedIn: [Muhammad Haris Salman](https://www.linkedin.com/in/muhammadharissalman)

## 📄 License

```
This project is licensed under the MIT License - see the LICENSE file for details.
```
