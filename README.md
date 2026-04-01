# Multilingual Hand-Gesture Recognition & Voice-Translation System

An AI-driven communication bridge that translates Indian (ISL), American (ASL), and Bengali (BSL) sign languages into text and natural speech in real-time.

## 📌 Project Overview
This system utilizes computer vision and deep learning to break communication barriers for the hearing-impaired. By capturing 126-dimensional hand landmark data, the system predicts gestures using an LSTM network, refines the output via NLP, and provides multilingual audio feedback.

## 🏗️ System Architecture
The system follows a **Modular Pipeline** architecture:
1. **Capture Engine**: MediaPipe Hands extracts 21 3D landmarks per hand ($x, y, z$).
2. **Inference Engine**: A TensorFlow-based **LSTM (Long Short-Term Memory)** model processes temporal sequences for gesture classification.
3. **Semantic Mapper**: Translates gestures into Sign Language Gloss.
4. **NLP Refinement**: TextBlob performs grammar correction and spell-checking.
5. **Output Layer**: gTTS (Google Text-to-Speech) generates audible translations.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* FFmpeg (for Whisper/Audio processing)
* Standard HD Webcam

### Installation
```bash
git clone [https://github.com/yourusername/sign-language-translator.git](https://github.com/yourusername/sign-language-translator.git)
cd sign-language-translator
pip install -r libraries.txt






File,Function
1.dataset.py,Records 500 samples per gesture; saves 126-dimensional vectors to CSV.
2.train_model.py,Builds and trains the LSTM Sequential Model using TensorFlow/Keras.
3.predict_live.py,"Handles real-time inference, TextBlob correction, and gTTS output."
4.server_pipeline.py,Manages the WebSocket bridge for web-based rendering.
5.python -m http.server 8000 starts the server







🛠️ UsageData Collection: Run dataset.py to record new gestures into the dataset/ directory.Training: Run train_model.py and select the language (ISL/ASL/BSL/Gestures) to generate .h5 models.Live 
Execution:Terminal 1: python server_pipeline.py (Starts the Backend)
Terminal 2: python -m http.server 8000 (Starts the Frontend)
Open localhost:8000 in your browser.📊 
Technical SpecificationsFeature Vector: 126 elemts (21 landmarks X 3-coordinates X 2 hands) .
Neural Network: LSTM (128 units) → Dropout (0.3) → Dense (64) → Softmax.Sampling Rate: 30 FPS captured, resampled for inference stability.Accuracy: ~95.2% on test datasets.






### Key Technical Alignments Included:
* **Algorithm**: Correctly identifies the **LSTM** model and **TensorFlow** framework used in your `train_model.py`.
* **Data Flow**: Matches the 126-feature logic ($63 \times 2$) from your `get_combined_landmarks` function[cite: 778].
* **Setup**: Integrates the "Terminal 1/Terminal 2" launch sequence from your high-level architecture.