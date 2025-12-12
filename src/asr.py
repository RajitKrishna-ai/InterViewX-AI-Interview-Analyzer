#convert interview audio into timestamped segments


from faster_whisper import WhisperModel
model_size= 'tiny'
model = WhisperModel(model_size, device= "cpu")
print(f"ASR model '{model_size}' loaded successfully ")

def transcribe_audio(audio_path):
    """
    Input: path to audio file
    Output: list of segments with start, end, text
    """
    segments_list = []

    # Use the model.transcribe() correctly
    result = model.transcribe(audio_path, beam_size=5)
    
    # Depending on faster-whisper version, result may be: a tuple or  just a generator of segments
    try:
        segments, info = result
    except TypeError:
        segments = result  # it's a generator/list

    # Iterate and collect
    for segment in segments:
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    return segments_list

if __name__ == '__main__':
    audio_file = "data/sample_interview.wav"  
    results =transcribe_audio(audio_file)

    for seg in results:
        print(seg)