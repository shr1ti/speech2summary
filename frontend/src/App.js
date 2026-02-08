import React, { useState } from 'react';
import axios from 'axios';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Paper,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Grid,
  Card,
  CardContent,
} from '@mui/material';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [preprocessed, setPreprocessed] = useState({});
  const [summary, setSummary] = useState('');
  const [evaluation, setEvaluation] = useState({});
  const [bias, setBias] = useState({});
  const [similarity, setSimilarity] = useState(0);
  const [method, setMethod] = useState('bert_summa');
  const [word1, setWord1] = useState('');
  const [word2, setWord2] = useState('');
  const [error, setError] = useState(null);
  const titleSx = {
    fontFamily: '"Cinzel", "Georgia", serif',
    fontWeight: 900,
    textTransform: 'uppercase',
    fontSize: '32px',
  };
  const titleSxSmall = { ...titleSx, fontSize: '24px' };

  const handleTranscribe = async () => {
    setError(null);
    if (!file) {
      setError('Please select an audio file first.');
      return;
    }
    const formData = new FormData();
    formData.append('audio', file);

    try {
      const response = await axios.post('http://localhost:5000/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setTranscription(response.data.transcription);
    } catch (err) {
      console.error('Transcription error:', err);
      setError('Network error: Could not connect to backend. Please ensure backend server is running.');
    }
  };

  const handlePreprocess = async () => {
    try {
      const response = await axios.post('http://localhost:5000/preprocess', { text: transcription });
      setPreprocessed(response.data);
    } catch (err) {
      console.error('Preprocess error:', err);
      setError('Failed to preprocess text.');
    }
  };

  const handleSummarize = async () => {
    try {
      const response = await axios.post('http://localhost:5000/summarize', { text: transcription, method });
      setSummary(response.data.summary);
    } catch (err) {
      console.error('Summarize error:', err);
      setError('Failed to summarize text.');
    }
  };

  const handleEvaluate = async () => {
    try {
      const response = await axios.post('http://localhost:5000/evaluate', { generated: summary, reference: transcription });
      setEvaluation(response.data);
    } catch (err) {
      console.error('Evaluate error:', err);
      setError('Failed to evaluate summary.');
    }
  };

  const handleBiasCheck = async () => {
    try {
      const response = await axios.post('http://localhost:5000/bias_check', { text: summary });
      setBias(response.data);
    } catch (err) {
      console.error('Bias check error:', err);
      setError('Failed to check bias.');
    }
  };

  const handleSimilarity = async () => {
    try {
      const response = await axios.post('http://localhost:5000/similarity', { word1, word2 });
      setSimilarity(response.data.similarity);
    } catch (err) {
      console.error('Similarity check error:', err);
      setError('Failed to check similarity.');
    }
  };

  return (
    <div className="app">
      <AppBar
        position="static"
        sx={{
          background: 'linear-gradient(90deg, #3a1026 0%, #7a1b4a 45%, #d63682 100%)',
          borderBottom: '1px solid #2a3352',
          boxShadow: '0 10px 30px rgba(0,0,0,0.35)',
        }}
      >
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="h6" sx={{ fontWeight: 700, letterSpacing: 0.3 }}>
            NLP-Driven Summarization Platform
          </Typography>
          <span className="accent-chip">Local • v1</span>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" className="app-shell">
        {error && (
          <Paper
            className="glass-card"
            sx={{
              p: 2,
              mb: 2,
              borderColor: '#5a1e2a',
              background: 'linear-gradient(180deg, rgba(255, 107, 107, 0.15), rgba(21, 26, 43, 0.8))',
            }}
          >
            <Typography color="error" variant="h6">
              ERROR
            </Typography>
            <Typography>{error}</Typography>
          </Paper>
        )}
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper className="glass-card" sx={{ p: 2 }}>
              <Typography variant="h5" className="section-title" sx={titleSx}>
                Upload Audio
              </Typography>
              <Typography className="muted" sx={{ mb: 2 }}>
                Whisper (base) • FFmpeg decode
              </Typography>
              <input type="file" accept="audio/*" onChange={(e) => setFile(e.target.files[0])} />
              <Button
                variant="contained"
                onClick={handleTranscribe}
                className="pink-button"
                sx={{ ml: 2 }}
              >
                Transcribe
              </Button>
              <TextField
                fullWidth
                label="Transcription"
                value={transcription}
                multiline
                rows={4}
                sx={{
                  mt: 2,
                  '& .MuiInputBase-root': { color: '#e6ecff' },
                  '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2a3352' },
                  '& .MuiInputLabel-root': { color: '#a7b4d8' },
                }}
              />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper className="glass-card" sx={{ p: 2 }}>
              <Typography variant="h5" className="section-title" sx={titleSx}>
                Preprocess
              </Typography>
              <Typography className="muted" sx={{ mb: 2 }}>
                spaCy `en_core_web_sm`
              </Typography>
              <Button
                variant="contained"
                onClick={handlePreprocess}
                className="pink-button"
              >
                Preprocess
              </Button>
              <Typography className="output-text">Tokens: {preprocessed.tokens?.join(', ')}</Typography>
              <Typography className="output-text">Entities: {JSON.stringify(preprocessed.entities)}</Typography>
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper className="glass-card" sx={{ p: 2 }}>
              <Typography variant="h5" className="section-title" sx={titleSx}>
                Summarize
              </Typography>
              <Typography className="muted" sx={{ mb: 2 }}>
                Abstractive and extractive options
              </Typography>
              <FormControl fullWidth sx={{ mt: 2 }}>
                <InputLabel sx={{ color: '#a7b4d8' }}>Algorithm</InputLabel>
                <Select
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  sx={{
                    color: '#e6ecff',
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2a3352' },
                    '& .MuiSvgIcon-root': { color: '#a7b4d8' },
                  }}
                >
                  <MenuItem value="bert_summa">BART Abstractive (facebook/bart-large-cnn)</MenuItem>
                  <MenuItem value="mfmmr_bertsum">MMR Extractive (TF-IDF + cosine)</MenuItem>
                  <MenuItem value="bert_bigru">BART Abstractive (beam search)</MenuItem>
                  <MenuItem value="athena">Chunked BART (long text)</MenuItem>
                  <MenuItem value="txlasm">TxLASM (TF-IDF top terms)</MenuItem>
                  {/* Add more options as needed */}
                </Select>
              </FormControl>
              <Button
                variant="contained"
                onClick={handleSummarize}
                className="pink-button"
                sx={{ mt: 2 }}
              >
                Summarize
              </Button>
              <TextField
                fullWidth
                label="Summary"
                value={summary}
                multiline
                rows={4}
                sx={{
                  mt: 2,
                  '& .MuiInputBase-root': { color: '#e6ecff' },
                  '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2a3352' },
                  '& .MuiInputLabel-root': { color: '#a7b4d8' },
                }}
              />
            </Paper>
          </Grid>

          <Grid item xs={6}>
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" className="section-title" sx={titleSxSmall}>
                  Evaluate
                </Typography>
                <Typography className="muted" sx={{ mb: 2 }}>
                  BERTScore • ROUGE • MNLI entailment
                </Typography>
                <Button
                  variant="contained"
                  onClick={handleEvaluate}
                  className="pink-button"
                >
                  Evaluate
                </Button>
                <Typography className="output-text">BERTScore F1: {evaluation.bertscore?.f1}</Typography>
                <Typography className="output-text">ROUGE-1: {evaluation.rouge?.rouge1}</Typography>
                <Typography className="output-text">Entailment (BART MNLI): {evaluation.entailment}</Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={6}>
            <Card className="glass-card">
              <CardContent>
                <Typography variant="h6" className="section-title" sx={titleSxSmall}>
                  Bias Check (Sentiment Proxy)
                </Typography>
                <Typography className="muted" sx={{ mb: 2 }}>
                  DistilBERT SST-2 sentiment
                </Typography>
                <Button
                  variant="contained"
                  onClick={handleBiasCheck}
                  className="pink-button"
                >
                  Check Bias
                </Button>
                <Typography className="output-text">
                  Score: {bias.bias_score}, Label: {bias.label}
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12}>
            <Paper className="glass-card" sx={{ p: 2 }}>
              <Typography variant="h5" className="section-title" sx={titleSx}>
                Word Similarity (Sentence-Transformers)
              </Typography>
              <Typography className="muted" sx={{ mb: 2 }}>
                all-MiniLM-L6-v2 embeddings
              </Typography>
              <TextField
                label="Word 1"
                value={word1}
                onChange={(e) => setWord1(e.target.value)}
                sx={{
                  '& .MuiInputBase-root': { color: '#e6ecff' },
                  '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2a3352' },
                  '& .MuiInputLabel-root': { color: '#a7b4d8' },
                }}
              />
              <TextField
                label="Word 2"
                value={word2}
                onChange={(e) => setWord2(e.target.value)}
                sx={{
                  ml: 2,
                  '& .MuiInputBase-root': { color: '#e6ecff' },
                  '& .MuiOutlinedInput-notchedOutline': { borderColor: '#2a3352' },
                  '& .MuiInputLabel-root': { color: '#a7b4d8' },
                }}
              />
              <Button
                variant="contained"
                onClick={handleSimilarity}
                className="pink-button"
                sx={{ ml: 2 }}
              >
                Check Similarity
              </Button>
              <Typography className="output-text">Similarity: {similarity}</Typography>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;
