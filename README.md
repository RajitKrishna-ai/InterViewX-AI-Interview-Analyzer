# InterviewX – AI-Powered Interview Analyzer & Feedback Engine

InterviewX is an AI-driven interview analysis system designed to help candidates practice, improve, and refine their communication skills.  
Built with a production-grade architecture, it simulates the end-to-end pipeline used in modern AI interview tools — from audio 
transcription to NLP-based evaluation and automated feedback generation.

> 🚀 Designed, engineered, and optimized with real hiring workflows in mind — especially for UAE-based companies seeking AI-enabled
> training tools.

---

## 🌟 Key Features

### 🎤 **1. Real-Time Audio Transcription**
- Uses Whisper + CTranslate2 for fast, accurate speech-to-text.
- Supports noisy environments and accents.
- Ideal for mock HR and technical interviews.

### 🧹 **2. Smart Transcript Processing**
- Cleans filler words: *“uh”, “um”, “like”, “you know”*
- Segment-level timestamps  
- Sentence normalization for NLP processing

### 🧠 **3. Vector-Based Answer Analysis**
- Converts responses into embeddings (Sentence Transformers)
- Performs similarity search for:
  - **Best possible answers**
  - **Missing points**
  - **Improvement suggestions**

### 🤖 **4. LLM-Based Feedback Generation**
- Uses a custom LLM agent for:
  - Communication assessment  
  - Technical accuracy  
  - Confidence score  
  - Behavioral analysis (STAR method)

### 📊 **5. Interview Scorecard**
Outputs a structured report:
- Communication score  
- Technical depth  
- Relevance to question  
- Clarity & structure  
- Improvement recommendations  

### 🖥️ **6. Streamlit Web App**
A clean, minimal UI where users can:
- Upload audio  
- Get transcripts  
- View analytics  
- Download feedback  

---

