import yt_dlp



def download_youtube_video(url: str):
    ydl_opts = {
        # Download highest quality available
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        # Name the file based on the video title
        'outtmpl': 'E:/avd/assets/raw_video/%(title)s.%(ext)s', 
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        info = ydl.extract_info(url, download=False)
        print(f'E:/avd/assets/raw_video/{info.get("title")}.{info.get("ext")}')
        return f'E:/avd/assets/raw_video/{info.get("title")}.{info.get("ext")}'



