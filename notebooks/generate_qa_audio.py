from gtts import gTTS
import os

# Folder to save audio
audio_folder = "data"
os.makedirs(audio_folder, exist_ok=True)

# AI Engineer interview Q&A
qa_pairs = [
    ("Tell me about yourself.", "I am a data scientist with 3 years of experience in machine learning, NLP, and deep learning, passionate about building AI solutions that solve real-world problems."),
    ("Explain the difference between supervised and unsupervised learning.", "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data."),
    ("What is overfitting and how do you prevent it?", "Overfitting happens when a model learns training data too well. Techniques like regularization, dropout, and cross-validation help prevent it."),
    ("How do you deploy an ML model in production?", "Using APIs, cloud services, Docker containers, or pipeline orchestration tools like Airflow or Kubeflow.")
]

# Combine all Q&A into a single string
combined_text = ""
for i, (q, a) in enumerate(qa_pairs, start=1):
    combined_text += f"Question {i}: {q} ... Answer {i}: {a} ... "

# Output file
output_file = os.path.join(audio_folder, "sampple_interview.wav")

# Generate audio using gTTS
tts = gTTS(text=combined_text)
tts.save(output_file)

print(f"Saved single audio file with all Q&A: {output_file}")
