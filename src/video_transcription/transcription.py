import whisper

model = whisper.load_model("base")



def get_transcription(video_path: str):
    result = model.transcribe(video_path)
    return result["text"]