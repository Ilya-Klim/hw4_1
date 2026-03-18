import pandas as pd
import subprocess
import os
from pathlib import Path

MAX_SAMPLES = 1500 


def download_audio(youtube_id, output_path, duration=10):
    cmd = [
        'yt-dlp',
        '-x', '--audio-format', 'wav',
        '--postprocessor-args', f'ffmpeg:-t {duration}',
        '-o', str(output_path),
        f'https://www.youtube.com/watch?v={youtube_id}'
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    df = pd.read_csv('data/raw/musiccaps.csv') 
    df = df.head(MAX_SAMPLES)
    audio_dir = Path('data/audio')
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []

    for idx, row in df.iterrows():
        youtube_id = row['ytid']
        caption = row['caption']
        output_file = audio_dir / f"{youtube_id}.wav"
        
        if output_file.exists():
            print(f"Skip {youtube_id}")
            continue
            
        print(f"Downloading {youtube_id} ({idx}/{len(df)})")
        if download_audio(youtube_id, output_file):
            metadata_list.append({
                "path": str(output_file),
                "original_caption": caption,
                "youtube_id": youtube_id
            })
        else:
            print(f"Failed {youtube_id}")

    import json
    with open('data/raw/downloaded_meta.json', 'w') as f:
        json.dump(metadata_list, f, indent=2)


if __name__ == "__main__":
    main()