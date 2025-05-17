import os
import mlflow
from transformers import TrainingArguments, BitsAndBytesConfig
import shutil
from torch.serialization import add_safe_globals
from transformers import TrainingArguments, pipeline
from torch.serialization import add_safe_globals
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, ColSpec
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import mlflow.transformers
from peft import PeftConfig, PeftModel, get_peft_model

import torch
import json

labels=["Algebra", "Geometry and Trigonometry", "Calculus and Analysis", "Probability and Statistics", "Number Theory", "Combinatorics and Discrete Math", "Linear Algebra", "Abstract Algebra and Topology"]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {id: label for id, label in enumerate(labels)}

# === CONFIGURATION ===
CHECKPOINT_DIR = "checkpoint-1719"
EXPERIMENT_NAME = "math-topic-classification"

peft_config = PeftConfig.from_pretrained(CHECKPOINT_DIR)
base_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path, num_labels=8, device_map="cpu", id2label=id_to_label, label2id=label_to_id)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)


print("Loading PeftModel")
#model = Pt_peft_model(base_model, peft_config)eftModel.from_pretrained(model, CHECKPOINT_DIR)
model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model = model.merge_and_unload()  
model.eval()
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)
#model.load_adapter(CHECKPOINT_DIR, adapter_name="default", peft_config=peft_config)



print(pipe("What is the probability of getting a 5 on a fair die?"))


# mlflow models serve --model-uri runs:/6ca8c7512f3b486ba3f1e43a7621732c/transformers-model --port 5001 --no-conda
# curl -X POST -H "Content-Type: application/json" -d '{"inputs":["2x=8, solve for x"]}' http://127.0.0.1:5001/invocations

# curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" -d '{"columns": ["inputs"],"data":    [["∫ x² dx"]],"params":  {"return_all_scores": true}}'


# az ml online-endpoint get-credentials --name math-topic-prediction-endpoint -g ain3009-project -w  --query primaryKey -o tsv

# curl -X POST "https://math-topic-prediction-endpoint.francecentral.inference.ml.azure.com/score" -H "Authorization: Bearer 33zdSUzOXnJU1rjTyCV6vh0w8Nez5H7ObC8XpptSJUUlnGlNTcX4JQQJ99BEAAAAAAAAAAAAINFRAZML2cLM" -H "Content-Type: application/json" -d '{"input_data": ["Find the probability of getting a 5 on a fair die"]}'