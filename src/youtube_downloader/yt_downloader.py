import yt_dlp

url = 'https://youtu.be/UVR9lhUGAyU?si=csmW00sC_WTpMRyP'

# Configure download options
ydl_opts = {
    # Download highest quality available
    'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
    # Name the file based on the video title
    'outtmpl': 'E:/avd/assets/raw_video/%(title)s.%(ext)s', 
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
    info = ydl.extract_info(url, download=False)
    
    print(f"Title: {info.get('title')}")
    print(f"Duration: {info.get('duration')} seconds")
    print(f"View Count: {info.get('view_count')}")



