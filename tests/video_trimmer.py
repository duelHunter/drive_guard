from moviepy import VideoFileClip

video = VideoFileClip("data/videos/dashcam_footage_1.mp4")

cropped = video.subclipped("00:04", "00:53")  # 1 min 15 sec to 2 min 45 sec

cropped.write_videofile(
    "data/videos/output.mp4",
    codec="libx264",
    audio_codec="aac"
)

video.close()
cropped.close()