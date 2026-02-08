from transformers import pipeline

summarizer = pipeline("summarization")
print(summarizer("Hugging Face transformers provide great utilities for NLP.", max_length=50)[0]['summary_text'])