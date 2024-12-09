import logging
import pprint

import ffmpeg

from .__logging import trace

logger = logging.getLogger(__name__)


@trace(logger)
def probe(file):
    """Extract video metadata using ffmpeg.probe."""
    info = ffmpeg.probe(file)
    video_streams = [stream for stream in info['streams'] if stream['codec_type'] == 'video']
    if not video_streams:
        raise ValueError("No video stream found in file.")
    video_stream = video_streams[0]
    fps = eval(video_stream['avg_frame_rate'])  # avg_frame_rate is typically a fraction
    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'fps': fps}