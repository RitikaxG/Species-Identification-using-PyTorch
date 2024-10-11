# PetGuard: AI-Powered Pet Species Classification and Allergy Prediction

**PetGuard** is an AI-powered application designed to classify pet species using deep learning and provide insights into potential allergic reactions based on the identified species. The project leverages computer vision models built with **PyTorch** for image classification and integrates the **Gemini API** for allergy-related information.

## Features
- **Multi-Class Classification**: Identifies pet species (37 breeds of cats and dogs) from images.
- **Allergy Prediction**: After identifying the pet species, the application uses the **Gemini API** to provide potential allergic reaction details, enhancing user awareness and safety.
- **Interactive Web Application**: Built with **Streamlit** for an easy-to-use interface, allowing users to upload images and view classification results in real-time.
- **API Integration**: Allergy prediction data fetched in real-time from the Gemini API for enhanced insights.

## Project Workflow

### 1. **Data Collection**
   - The **Oxford IIIT Pet Dataset** was used, containing images of various cat and dog breeds. The dataset includes 37 distinct pet species, labeled for classification tasks.

### 2. **Data Preprocessing**
   - **Image Resizing**: All images are resized to a uniform size (224x224 pixels) to feed into the model.
   - **Normalization**: Pixel values are normalized to accelerate the training process.
   - **Data Augmentation**: Techniques like rotation, zooming, and flipping were used to increase the diversity of the dataset and improve model generalization.

### 3. **Model Architecture**
   - **Tiny VGG Model**: A simplified version of the VGG architecture was chosen due to its balance between performance and computational cost. The model was enhanced with:
     - **Batch Normalization**: For stable training and faster convergence.
     - **Adam Optimizer**: To improve the speed and accuracy of model training.
   - The model achieved an accuracy improvement from 0.78 to 0.83 after hyperparameter tuning.

### 4. **Web Application**
   - **Frontend**: Built with **Streamlit**, allowing users to upload pet images easily and receive real-time species classification and allergy information.
   - **Backend**: The model is served using **Flask**, handling image uploads and communication with the PyTorch model for inference.

### 5. **API Integration**
   - After the species classification, the application calls the **Gemini API**, which provides information regarding potential allergic reactions based on the identified species. This API integration allows for a unique, safety-conscious feature that helps users make informed decisions about pet allergies.

## Technologies Used
- **Python**: Core programming language for the project.
- **PyTorch**: Used for building the deep learning model.
- **Flask**: Backend framework for handling model inference and API calls.
- **Streamlit**: Frontend for creating an interactive and user-friendly web interface.
- **Gemini API**: Provides allergy-related information based on pet species.
- **Pandas and NumPy**: Used for data manipulation and preprocessing.
- **OpenCV**: Used for image processing and augmentation.

## Key Achievements
- Improved classification accuracy from **0.78 to 0.83** by optimizing the Tiny VGG architecture.
- Built a real-time classification system that predicts pet species and integrates allergy information in a user-friendly web interface.
- Successfully integrated a third-party API (Gemini) to provide additional functionality beyond basic classification.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/petguard.git

2. Install the required dependencies
    ```bash
    pip install -r requirements.txt

4. Run application
    ```bash
   streamlit run app.py

