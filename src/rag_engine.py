import ollama
from embeddings import search_similar, get_segment_by_index

def analyze_with_context(query_text, top_k=3):

    # -------- Step 1: RAG retrieval 
    distances, indices = search_similar(query_text, top_k=top_k)

    rag_context = "\n".join(
        [get_segment_by_index(idx) for idx in indices[0]]
    )

    # -------- Step 2: Simple real metrics 
    word_count = len(query_text.split())
    sentence_count = max(1, query_text.count('.'))
    avg_sentence_length = word_count / sentence_count

    metrics_summary = f"""
Word count: {word_count}
Sentences: {sentence_count}
Avg sentence length: {avg_sentence_length:.1f}
"""

    # -------- Step 3: Partial full-answer context 
    full_answer_preview = query_text[:1500]

    # -------- Step 4: Smart prompt
    prompt = f"""
You are a senior FAANG-level interview coach.

Use this information:

FULL ANSWER:
{full_answer_preview}

MOST RELEVANT SEGMENTS:
{rag_context}

SIMPLE METRICS:
{metrics_summary}

Return ONLY valid JSON with this exact structure:

{{
  "clarity": 0,
  "structure": 0,
  "confidence": 0,
  "tip": "string"
}}
"""

    # -------- Step 5: LLaMA 3 call-
    response = ollama.chat(
        model="llama3:8b",
        messages=[
            {"role": "system", "content": "You only return valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


# -------- Local testing
if __name__ == "__main__":
    test_text = """
    I am a data scientist with experience in machine learning and NLP.
    I solve real-world problems using AI systems.
    """

    print(analyze_with_context(test_text))
