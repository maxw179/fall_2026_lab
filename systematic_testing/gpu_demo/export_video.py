"""Export the exact pre-rendered HTML replay frames to a Google Slides MP4."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import shutil
import subprocess


def export_video(html_path, output=None, *, frame_seconds=.7, ffmpeg=None):
    html_path = Path(html_path)
    output = Path(output) if output else html_path.with_suffix('.mp4')
    if frame_seconds <= 0:
        raise ValueError('frame_seconds must be positive')
    match = re.search(r'const frames=(\[.*?\]);let index=', html_path.read_text(), re.DOTALL)
    if match is None:
        raise ValueError('No embedded replay frame array found')
    frames = json.loads(match.group(1))
    if not frames:
        raise ValueError('Replay has no frames')
    encoder = ffmpeg or shutil.which('ffmpeg')
    if not encoder:
        import imageio_ffmpeg
        encoder = imageio_ffmpeg.get_ffmpeg_exe()
    command = [str(encoder), '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'image2pipe', '-framerate', str(1/frame_seconds), '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-r', '30', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for frame in frames:
            prefix, encoded = frame.split(',', 1)
            if prefix != 'data:image/png;base64':
                raise ValueError('Expected embedded PNG frames')
            process.stdin.write(base64.b64decode(encoded, validate=True))
        process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg export failed')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frame-seconds', type=float, default=.7)
    parser.add_argument('--ffmpeg')
    args = parser.parse_args()
    print(export_video(args.html, args.output, frame_seconds=args.frame_seconds, ffmpeg=args.ffmpeg))


if __name__ == '__main__':
    main()
