from youtube_downloader.yt_downloader import download_youtube_video
from video_transcription.transcription import get_transcription

url = input('Enter the URL of the video: ')
video_path = download_youtube_video(url)
video_transcription = get_transcription(video_path)
print(video_transcription)