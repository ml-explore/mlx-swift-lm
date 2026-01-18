The two includes resources are for testing the MediaProcessing pipeline for correctness and validation

The video file was created via FFMPEG via

```
ffmpeg \
-f lavfi \
-i smptebars=size=1920x1080:rate=30:duration=5.0 \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-vf "drawtext=text='Frame\\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:alpha=0.5:box=1:boxborderw=4,drawtext=text='TC':x=(w-tw)/2:y=(lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:fontcolor=white:timecode='01\\:00\\:00\\:00':timecode_rate=(30)" \
-c:v libx264 \
-c:a aac \
-crf 23 \
-preset medium \
-pix_fmt yuv420p \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y 1080p_30.mov
```

and the audio only file 

```
ffmpeg \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-c:a aac \
-crf 23 \
-preset medium \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y audio_only.mov
```
