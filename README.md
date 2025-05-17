# 🧮 Math Topic Classification Project 🔍

## Overview

This project implements an AI system that automatically classifies mathematical questions into 8 different topics using advanced NLP techniques. It fine-tunes pre-trained large language models with QLoRA (Quantized Low-Rank Adaptation) for efficient training on limited computational resources.

This project uses data from the Kaggle competition ["Classification of Math Problems by Kasut Academy"](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/overview).

The system consists of:
- A **model training pipeline** that uses fine-tuning with data augmentation to improve classification accuracy
- A **deployment workflow** on Azure ML that exposes the model as a REST API endpoint
- A **web application frontend** that provides a user-friendly interface to submit math questions and receive topic classifications

The solution achieves over 90% accuracy on the validation set while keeping response times under 500ms, making it suitable for real-time educational applications and automated content tagging systems.

## 🏆 Competition Results

This project achieved a **0.9114 F1-micro score** on the private test set of the Kaggle competition [Leaderboard](https://www.kaggle.com/competitions/classification-of-math-problems-by-kasut-academy/leaderboard), ranking **7th out of 341 participants**.

## 📋 Topics

The model classifies questions into these 8 math topics:

| ID | Topic |
|----|-------|
| 0 | Algebra |
| 1 | Geometry and Trigonometry |
| 2 | Calculus and Analysis |
| 3 | Probability and Statistics |
| 4 | Number Theory |
| 5 | Combinatorics and Discrete Math |
| 6 | Linear Algebra |
| 7 | Abstract Algebra and Topology |

## 🌟 Features

- 📊 Efficient fine-tuning with QLoRA (4-bit quantization + Low-Rank Adaptation)
- 📈 MLflow experiment tracking and model versioning
- 🔄 Data augmentation techniques (number substitution, back-translation)
- 📝 Detailed evaluation metrics and confusion matrix visualization
- ☁️ Azure ML deployment pipeline

## 🛠️ Tech Stack

- 🔥 PyTorch & 🤗 Transformers - Deep learning framework
- 📊 MLflow - Experiment tracking and model registry
- 🧠 QLoRA - Memory-efficient fine-tuning technique
- ☁️ Azure ML - Model deployment and endpoint hosting
- 🐳 Docker - Application containerization
- 🌐 FastAPI - Web API framework for the frontend

## 📁 Project Structure

```
math-topic-classification/
├── data/
│   ├── train.csv
│   ├── sample_submission.csv
│   └── test.csv
├── utils/
│   ├── models.py
│   ├── logging_utils.py
│   └── evaluation_utils.py
├── frontend/                   # Web application for the model
│   ├── app.py                  # Access to Azure ML endpoint
│   └── Dockerfile              # Dockerfile for containerization
├── train.py                    # Fine-tune model with QLoRA
├── predict.py                  # Generate predictions
├── evaluate.py                 # Evaluate checkpoints
├── split_data.py               # Split data
├── augment_data.py             # Augment training data
├── endpoint.yml                # Azure ML endpoint config
├── deployment.yml              # Azure ML deployment config
└── README.md
```

## 🛠️ Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
## 🚀 Usage

### Data Preparation

#### Splitting Data

Split your training data into train and validation sets:

```bash
python split_data.py --train_path data/train.csv --val_size 0.7 --random_state 46
```

This creates `train_split.csv` and `val_split.csv` in your data directory.

#### Data Augmentation 🔄

Expand your training data with augmentation techniques:

```bash
python augment_data.py --input_file data/train_split.csv --output_file data/augmented_train.csv
```

This script:
- Changes numbers in math problems
- Performs back-translation (translates to another language and back)
- Focuses augmentation on underrepresented classes

### Training 🧠

Train the model with QLoRA fine-tuning:

```bash
python train.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
                --train_path data/augmented_train.csv \
                --val_path data/val_split.csv \
                --num_epochs 10 \
                --learning_rate 1e-4 \
                --use_mlflow
```

Key parameters:
- `--model`: Hugging Face model ID to fine-tune
- `--use_mlflow`: Enable MLflow tracking
- `--num_epochs`: Number of training epochs
- `--early_stopping_patience`: Stop if validation metrics don't improve

### Evaluation 📊

Evaluate a specific checkpoint:

```bash
python evaluate.py --checkpoint_path results/20250501_220500_Qwen2.5-Math-7B/checkpoints/checkpoint-600 \
                  --val_path data/val_split.csv
```

This generates:
- Detailed metrics (accuracy, F1, precision, recall)
- Confusion matrix visualization
- Per-class performance analysis

### Prediction 🔮

Generate predictions on new data:

```bash
python predict.py --model_path results/20250501_220500_Qwen2.5-Math-7B/final_model \
                 --test_path data/test.csv \
                 --output_path predictions.csv
```

## ☁️ Azure Deployment

This project follows a two-step deployment process on Azure:

### Step 1: Azure ML Model Deployment

First, deploy your model as an online endpoint on Azure ML:

1. Create resource group and workspace:
   ```bash
   az group create -n ain3009-project -l francecentral 
   az ml workspace create --name ml-workspace -g ain3009-project
   ```

2. Register the model from your MLflow tracking:
   ```bash
   az ml model create --name math-topic-classification --type mlflow_model \
                      --path "mlruns/194569180057676461/f2d38696a6ac4f10b8f38638e8aa69f0/artifacts/transformers-model" \
                      -g ain3009-project -w ml-workspace
   ```

3. Create endpoint and deployment:
   ```bash
   az ml online-endpoint create -f endpoint.yml -g ain3009-project -w ml-workspace
   az ml online-deployment create -f deployment.yml -g ain3009-project -w ml-workspace
   ```

4. Update traffic allocation:
   ```bash
   az ml online-endpoint update --name math-topic-prediction-endpoint --traffic blue=100 \
                              -g ain3009-project -w ml-workspace
   ```

5. Test the deployed endpoint:
   ```bash
   curl -X POST "https://math-topic-prediction-endpoint.francecentral.inference.ml.azure.com/score" \
        -H "Authorization: Bearer <your-api-key>" \
        -H "Content-Type: application/json" \
        -d '{"input_data": ["Find the limit of (1+1/n)^n as n approaches infinity."]}'
   ```

### Step 2: Frontend Web App Deployment

Next, deploy the frontend web application that will use the ML endpoint:

1. Build and test the Docker image locally:
   ```bash
   cd frontend
   docker build -t math-topic-classification:latest .
   docker run -p 8000:8000 math-topic-classification:latest
   ```

2. Create and configure Azure Container Registry:
   ```bash
   az acr create --resource-group ain3009-project --name ain3009acr --sku Basic
   az acr login --name ain3009acr
   az acr update --name ain3009acr --admin-enabled true
   ```

3. Build and push the Docker image for Azure (multi-platform):
   ```bash
   docker buildx build --platform linux/amd64 -t ain3009acr.azurecr.io/math-topic-classification:latest . --push
   ```

4. Deploy to Azure Web App:
   ```bash
   # Get ACR credentials
   ACR_USERNAME=$(az acr credential show --name ain3009acr --query "username" -o tsv)
   ACR_PASSWORD=$(az acr credential show --name ain3009acr --query "passwords[0].value" -o tsv)

   # Create App Service plan
   az appservice plan create --name ain3009-webapp --resource-group ain3009-project --sku B1 --is-linux

   # Create and configure web app
   az webapp create \
     --resource-group ain3009-project \
     --plan math-topic-plan \
     --name math-topic-classifier \
     --deployment-container-image-name ain3009acr.azurecr.io/math-topic-classification:latest \
     --docker-registry-server-user $ACR_USERNAME \
     --docker-registry-server-password $ACR_PASSWORD
   ```

The web app connects to the Azure ML endpoint to provide a user-friendly interface for classifying math questions into topics.

