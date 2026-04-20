# The code was fetched from the article: https://huggingface.co/learn/agents-course/unit1/what-are-llms
import torch
from transformers import pipeline

# Identifying the model
# Specify the model task and the pre-trained model to use
classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    dtype=torch.float16,
    device=0
)

result = classifier("I love using Hugging Face Transformers!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]