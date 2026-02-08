# NLP-Driven Audio Transcription and Contextual Summary Generation Platform

This platform integrates AI algorithms from Chapters 2 and 3 of the project report, overcoming limitations like bias, redundancy, and language dependency.

## Setup
1. Install Python 3.8+ and Node.js 16+.
2. Backend: `cd backend && pip install -r requirements.txt && python app.py`
3. Frontend: `cd frontend && npm install && npm start`
4. Access at `http://localhost:3000`.

## Features
- Audio transcription (Whisper)
- Preprocessing (SpaCy)
- Summarization (Multiple algorithms)
- Evaluation (BERTScore, ROUGE)
- Bias check and word similarity

## Deployment
Use `docker-compose up` for containerized run.