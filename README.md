**AI Chatbot for Mental Health Support**
# Overview
This project is an AI-powered chatbot designed to provide mental health support using machine learning (ML). The chatbot understands user emotions and generates empathetic responses through a pre-trained ML model. Built with Flask for the web interface, the chatbot integrates the transformers library to process user inputs and deliver supportive messages. The project has been successfully deployed on Render and is accessible for users to interact with, offering a seamless experience for mental health support.

**Project Structure**

app.py : The main Flask application that powers the web interface and handles user interactions.

test_model.py: Loads the ML model and processes user inputs to generate responses.

train_model.py: Script used to train the ML model, producing the model weights and configuration files.

download_syntheticchat.py: Script to download the synthetic_therapy_conversation.csv dataset from Kaggle.

templates/chat.html: The HTML template for the chat interface.

static/style.css: CSS styles for the chat interface.

requirements.txt: Lists the Python dependencies, including Flask, gunicorn, transformers, torch, and kaggle.

Procfile: Configuration file for deploying the app on Render using Gunicorn.

output/: Contains ML model configuration files (e.g., config.json, vocab.txt). Large files like model.safetensors and tokenizers.pkl were excluded to optimize deployment.

**Setup and Usage**
# Prerequisites

Python 3.8+
Git
A Kaggle account and API key (for downloading the dataset)
Access to the deployed app on Render (link provided by the project owner)

# Local Setup

Clone the Repository:git clone <https://github.com/Kammara-Kaveri/AI_Chatbot_Mental_Health_Support_Project.git>
cd AI_Chatbot_Mental_Health_Support


Set Up a Virtual Environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:pip install -r requirements.txt


Run the App Locally:python app.py


Open http://localhost:5000 in your browser to interact with the chatbot.
Note: To test the ML model locally, the output directory with model.safetensors is required. See the "Training the ML Model" section below to recreate it.



# Downloading the Dataset

The synthetic_therapy_conversation.csv dataset is required for training the ML model but is excluded from the repository to reduce size. To download it:
Set Up Kaggle API:
Install the Kaggle package: pip install kaggle
Place your kaggle.json API key in ~/.kaggle/ (or C:\Users\<kammarakaveri>\.kaggle\ on Windows).


Run the Download Script:python download_syntheticchat.py


This downloads synthetic_therapy_conversation.csv from Kaggle and saves it in the project directory.


# Training the ML Model

# Download the Dataset:
Run download_syntheticchat.py as described above to obtain synthetic_therapy_conversation.csv.
A processed version (processed_synthetic_chatbot_data.csv) was created during development but excluded to reduce repository size.


# Train the Model:
Run the training script:python train_model.py


This trains the ML model using synthetic_therapy_conversation.csv and generates the output directory with model weights (model.safetensors) and tokenizer files.
Note: Training requires significant computational resources and is best performed on a powerful machine.



**Deployment on Render**
The chatbot has been successfully deployed on Render. To deploy a new instance:

Push the repository to GitHub (already done).
Create a new Web Service on Render.
Connect the GitHub repository and configure the build settings:
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app (as specified in the Procfile)


Deploy the app and access it via the provided Render URL.
**
**Summary:****
This project was developed to create an ML-powered mental health support chatbot. We started by creating download_syntheticchat.py to download the synthetic_therapy_conversation.csv dataset from Kaggle. The dataset was processed to create processed_synthetic_chatbot_data.csv, which was used to train the ML model with train_model.py. The training process generated the output directory containing the model weights and configuration files. To optimize the repository for deployment, we excluded large files like model.safetensors, tokenizers.pkl, processed_synthetic_chatbot_data.csv, and synthetic_therapy_conversation.csv, while retaining essential configuration files in output/. We built a Flask web interface with app.py, styled it with static/style.css, and created a user-friendly chat UI in templates/chat.html. The project was configured for Render deployment with Procfile and requirements.txt, pushed to GitHub, and successfully deployed on Render, where it now runs seamlessly, ready to assist users with mental health support.

Notes
The ML model weights (model.safetensors) were excluded from the repository to reduce size. To test the ML functionality locally, run train_model.py to regenerate the output directory.
The deployed chatbot on Render is fully functional, with ML capabilities available after training if needed on the Render instance.


