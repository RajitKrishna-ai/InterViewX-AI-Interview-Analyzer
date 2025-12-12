import streamlit as st
import tempfile
import pandas as pd
import altair as alt
import json

from asr import transcribe_audio
from rag_engine import analyze_with_context
from embeddings import set_segments, get_segment_by_index

# Streamlit Page Setup
# ---------------------
st.set_page_config(page_title="InterViewX AI Coach")
st.title("InterViewX - AI Interview Coach")
st.write("Upload your interview audio (.wav) and get instant feedback.")

# Session State for caching
# --------------------------
if "segments" not in st.session_state:
    st.session_state.segments = None
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "feedback" not in st.session_state:
    st.session_state.feedback = None

# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload your interview audio (.wav)", type=["wav"])

if uploaded_file is not None:
    if (st.session_state.audio_file != uploaded_file) or (st.session_state.segments is None):
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        # Step 1: Transcribe
        st.info("Transcribing audio...")
        segments = transcribe_audio(temp_audio_path)
        st.session_state.segments = segments
        st.session_state.audio_file = uploaded_file

        # Save segments globally for retrieval
        set_segments(segments)

    else:
        segments = st.session_state.segments

    # Show full transcript
    st.subheader(" Transcript")
    full_text = " ".join([seg["text"] for seg in segments])
    st.write(full_text)

    # Show top 2 segments for quick preview
    st.subheader(" Top Relevant Segments")
    for i, seg in enumerate(segments[:2]):
        st.write(f"{i+1}. {seg['text']}")

    # Analyze Button
    # ----------------------------
    if st.button("Analyze Interview"):
        st.info("Analyzing with AI...")

        feedback_raw = analyze_with_context(full_text, top_k=3)
        st.session_state.feedback = feedback_raw

        
        # Parse JSON feedback
        # ----------------------------
        try:
            scores = json.loads(feedback_raw)
        except:
            st.error("Failed to parse AI response. Showing raw output.")
            scores = {"clarity": 0, "structure": 0, "confidence": 0, "tip": feedback_raw}

        
        # Show Chart
        # ----------------------------
        st.subheader("AI Scores")
        df = pd.DataFrame({
            "Metric": ["Clarity", "Structure", "Confidence"],
            "Score": [scores.get("clarity",0), scores.get("structure",0), scores.get("confidence",0)]
        })

        chart = alt.Chart(df).mark_bar(color="#4CAF50").encode(
            x='Metric',
            y='Score'
        )

        st.altair_chart(chart, use_container_width=True)

        
        # Improvement Tip
        # ----------------------------
        st.subheader("Improvement Tip")
        st.write(scores.get("tip", "No tip available."))
