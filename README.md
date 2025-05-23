# AI Chatbot for Mental Health Support
**Overview**
This project is an AI-powered chatbot designed to provide mental health support using machine learning (ML). The chatbot understands user emotions and generates empathetic responses through a pre-trained ML model. Built with Flask for the web interface, the chatbot integrates the transformers library to process user inputs and deliver supportive messages. The project has been successfully deployed on Render and is accessible for users to interact with, offering a seamless experience for mental health support.

**Project Structure**

**app.py** : The main Flask application that powers the web interface and handles user interactions.

**test_model.py**: Loads the ML model and processes user inputs to generate responses.

**train_model.py**: Script used to train the ML model, producing the model weights and configuration files.

**download_syntheticchat.py**: Script to download the synthetic_therapy_conversation.csv dataset from Kaggle.

**templates/chat.html**: The HTML template for the chat interface.

**static/style.css**: CSS styles for the chat interface.

**requirements.txt**: Lists the Python dependencies, including Flask, gunicorn, transformers, torch, and kaggle.

**Procfile**: Configuration file for deploying the app on Render using Gunicorn.

**output/**: Contains ML model configuration files (e.g., config.json, vocab.txt). Large files like model.safetensors and tokenizers.pkl were excluded to optimize deployment.

## Setup and Usage
### Prerequisites

Python 3.8+

Git

A Kaggle account and API key (for downloading the dataset)

Access to the deployed app on Render :[https://ai-chatbot-mental-health-support.onrender.com]

## Local Setup

**Clone the Repository:**
```python
git clone <https://github.com/Kammara-Kaveri/AI_Chatbot_Mental_Health_Support_Project.git>
cd AI_Chatbot_Mental_Health_Support
```

**Set Up a Virtual Environment**:
```python
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Install Dependencies:**
```python
pip install -r requirements.txt
```

**Run the App Locally:**
```python
python app.py
```

- Open http://localhost:5000 in your browser to interact with the chatbot.
- Note: To test the ML model locally, the output directory with model.safetensors is required. See the "Training the ML Model" section below to recreate it.

## Downloading the Dataset
-The synthetic_therapy_conversation.csv dataset is required for training the ML model but is excluded from the repository to reduce size. To download it:
1. **Set Up Kaggle API:**
    -Install the Kaggle package: 'pip install kaggle'

    -Place your kaggle.json API key in ~/.kaggle/ (or C:\Users\<kammarakaveri>\.kaggle\ on Windows).
2. **Run the Download Script:**
    ```python
    python download_syntheticchat.py
    ```
    - This downloads synthetic_therapy_conversation.csv from Kaggle and saves it in the project directory.


## Training the ML Model
1. **Download the Dataset:**
    - Run download_syntheticchat.py as described above to obtain synthetic_therapy_conversation.csv.
    - A processed version (processed_synthetic_chatbot_data.csv) was created during development but excluded to reduce repository size.
2. **Train the Model:**
    - Run the training script:
    ```python
    python train_model.py
    ```
    - This trains the ML model using synthetic_therapy_conversation.csv and generates the output directory with model weights (model.safetensors) and tokenizer files.
- Note: Training requires significant computational resources and is best performed on a powerful machine.


## Deployment on Render:
The chatbot has been successfully deployed on Render at [https://ai-chatbot-mental-health-support.onrender.comtext].To deploy a new instance:
1. Pushed the repository to GitHub (already done).
2. Created a new Web Service on Render.
3. Connected the GitHub repository and configure the build settings:
   -**Environment**: Docker
   -**Dockerfile**: Ensured a Dockerfile is present in the repository to install Java (required for language_tool_python).
4. Add the following environment variables:
   - FLASK_SECRET_KEY: A secure random string 
   - PORT: Set to 10000
5. Deployed the app and access it via the provided Render URL.

## Summary:
This project focuses on developing an ML-powered mental health support chatbot. The project begins with download_syntheticchat.py to retrieve the synthetic_therapy_conversation.csv dataset from Kaggle. The dataset is then processed into processed_synthetic_chatbot_data.csv, which serves as the input for training the machine learning model using train_model.py. The training process generates an output/ directory containing model weights and configuration files.

To optimize the repository for deployment, large files such as model.safetensors, tokenizers.pkl, processed_synthetic_chatbot_data.csv, and synthetic_therapy_conversation.csv are excluded, while essential configuration files are retained. 

A Flask-based web interface is implemented via app.py, styled using static/style.css, and presented through a user-friendly UI in templates/chat.html.Deployment is configured using a Procfile and requirements.txt, and the complete application is pushed to GitHub and successfully deployed on Render using a Docker environment to support language_tool_python (requiring Java). 
The app now runs seamlessly at [https://ai-chatbot-mental-health-support.onrender.com], ready to assist users with mental health support.. 

## Notes:
- The ML model weights (model.safetensors) were excluded from the repository to reduce size. To test the ML - functionality locally, run train_model.py to regenerate the output directory.
- The deployed chatbot on Render is fully functional, with ML capabilities available after training if needed on the Render instance.


