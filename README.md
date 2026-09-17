# speech2summary
Full-stack app that transcribes audio with Whisper, preprocesses with spaCy, summarizes with BART and extractive methods, and evaluates with BERTScore, ROUGE, MNLI entailment, sentiment proxy, and Sentence-Transformers similarity.

## Setup
1. Install Python 3.8+ and Node.js 16+.
2. Backend: `cd backend && pip install -r requirements.txt && python app.py`
3. Frontend: `cd frontend && npm install && npm start`
4. Open `http://localhost:3000`.

## Features
- Audio transcription (Whisper)
- Preprocessing (spaCy)
- Summarization (BART + extractive)
- Evaluation (BERTScore, ROUGE, MNLI)
- Sentiment proxy + semantic similarity

## Deployment
Use `docker-compose up` for containerized run.
