from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import spacy
from summa import summarizer
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import io
import numpy as np
import soundfile as sf 
import tempfile
import os

nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

@app.route('/')
def home():
    return "Backend API is running. Use the frontend for interaction."

whisper_model = None
nlp = None
summarizer_tokenizer = None
summarizer_model = None
entailment_pipeline = None
bias_pipeline = None
similarity_tokenizer = None
similarity_model = None

def load_models():
    global whisper_model, nlp, summarizer_tokenizer, summarizer_model
    global entailment_pipeline, bias_pipeline, similarity_tokenizer, similarity_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    if entailment_pipeline is None or bias_pipeline is None or summarizer_model is None:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        # Use direct seq2seq model to avoid pipeline task registry issues.
        if summarizer_tokenizer is None or summarizer_model is None:
            summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
            summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        if entailment_pipeline is None:
            # Use an NLI model for entailment; DialoGPT is not a classifier and will error here.
            entailment_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli")
        if bias_pipeline is None:
            bias_pipeline = pipeline("sentiment-analysis")  # For Bipol-like bias detection
    if similarity_model is None:
        from sentence_transformers import SentenceTransformer
        similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# Simulated functions for Chapter 2 algorithms (simplified implementations)
def txlasm_summarize(text):
    # Language-agnostic: Use TF-IDF for extractive summary
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf = vectorizer.fit_transform([text])
    scores = tfidf.toarray().sum(axis=0)
    words = vectorizer.get_feature_names_out()
    top_words = [words[i] for i in scores.argsort()[-10:]]
    return " ".join(top_words)  # Simplified summary

def mfmmr_bertsum(text):
    # Extractive with MMR
    load_models()
    sentences = [sent.text for sent in nlp(text).sents]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    similarity = cosine_similarity(tfidf)
    selected = [0] if sentences else []
    for _ in range(min(3, len(sentences) - len(selected))):
        scores = []
        candidate_indices = []
        for i in range(len(sentences)):
            if i not in selected:
                rel = similarity[i, selected].max()
                red = similarity[i, selected].mean()
                scores.append(rel - 0.5 * red)
                candidate_indices.append(i)
        if scores:
            best_idx = candidate_indices[int(np.argmax(scores))]
            selected.append(best_idx)
    return " ".join([sentences[i] for i in selected]) if selected else ""

def bert_bigru_summarize(text):
    load_models()
    inputs = summarizer_tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = summarizer_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        min_length=50,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def bart_summarize(text, max_len=180, min_len=60):
    load_models()
    inputs = summarizer_tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = summarizer_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
    )
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def athena_summarize(text):
    # Chunk long text
    load_models()
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    summaries = [bert_bigru_summarize(chunk) for chunk in chunks]
    return " ".join(summaries)

def bipol_bias_check(text):
    load_models()
    result = bias_pipeline(text)
    return {"bias_score": result[0]['score'], "label": result[0]['label']}  # Simplified

def kcwe_similarity(word1, word2):
    # Simplified word similarity using embeddings
    load_models()
    from sentence_transformers import util
    embeddings = similarity_model.encode([word1, word2], convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    # Already in -1..1; normalize to 0..1 for UI
    return float((sim + 1.0) / 2.0)

# Chapter 3 flow


@app.route('/transcribe', methods=['POST'])
def transcribe():
    load_models()
    file = request.files['audio']
    if file is None:
        return jsonify({"error": "No audio file uploaded"}), 400

    # Whisper expects a file path or 1D float32 numpy array.
    # Save the upload to a temp file to avoid shape/format issues.
    audio_bytes = file.read()
    if not audio_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Decode audio first to ensure it's non-empty.
        audio = whisper.load_audio(tmp_path)
        if audio is None or len(audio) == 0:
            return jsonify({"error": "Could not decode audio (empty/unsupported). Please upload WAV/MP3/OGG with audio content."}), 400

        result = whisper_model.transcribe(audio)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return jsonify({"transcription": result["text"]})
@app.route('/preprocess', methods=['POST'])
def preprocess():
    load_models()
    data = request.json
    text = data['text']
    doc = nlp(text)
    tokens = [token.text for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return jsonify({"tokens": tokens, "entities": entities})

@app.route('/summarize', methods=['POST'])
def summarize():
    load_models()
    data = request.json
    text = data['text']
    method = data['method']
    
    if method == 'bert_summa':
        # Use BART summarization for more coherent output than extractive Summa.
        summary = bart_summarize(text, max_len=180, min_len=60)
    elif method == 'mfmmr_bertsum':
        summary = mfmmr_bertsum(text)
    elif method == 'bert_bigru':
        summary = bert_bigru_summarize(text)
    elif method == 'athena':
        summary = athena_summarize(text)
    elif method == 'txlasm':
        summary = txlasm_summarize(text)
    # Add more as needed (e.g., BAS, ADSum simulated similarly)
    else:
        summary = "Method not implemented"
    
    return jsonify({"summary": summary})

@app.route('/evaluate', methods=['POST'])
def evaluate():
    load_models()
    data = request.json
    generated = data['generated']
    reference = data['reference']
    P, R, F1 = bert_score([generated], [reference], lang="en")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, generated)
    # Use proper text-pair input for NLI
    entailment_result = entailment_pipeline({"text": reference, "text_pair": generated})
    if isinstance(entailment_result, list):
        entailment = entailment_result[0]['label']
    else:
        entailment = entailment_result['label']
    return jsonify({
        "bertscore": {"precision": P.item(), "recall": R.item(), "f1": F1.item()},
        "rouge": {k: v.fmeasure for k, v in rouge_scores.items()},
        "entailment": entailment
    })

@app.route('/bias_check', methods=['POST'])
def bias_check():
    data = request.json
    text = data['text']
    return jsonify(bipol_bias_check(text))

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.json
    word1 = data['word1']
    word2 = data['word2']
    return jsonify({"similarity": kcwe_similarity(word1, word2)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
