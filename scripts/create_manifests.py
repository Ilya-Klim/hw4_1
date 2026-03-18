#!/usr/bin/env python3
import json
import gzip
import random
from pathlib import Path

def create_manifests(metadata_path: str, output_dir: str, train_ratio: float = 0.9):
    """Создаёт train/valid манифесты в формате .jsonl.gz"""
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data, valid_data = data[:split_idx], data[split_idx:]
    
    def format_prompt(item: dict) -> str:
        """Форматирует промпт из структурированных полей"""
        parts = []
        if item.get('description'):
            parts.append(f"Description: {item['description']}")
        if item.get('general_mood'):
            parts.append(f"Mood: {item['general_mood']}")
        if item.get('genre_tags'):
            tags = item['genre_tags'] if isinstance(item['genre_tags'], list) else [item['genre_tags']]
            parts.append(f"Genres: {', '.join(tags)}")
        if item.get('lead_instrument'):
            parts.append(f"Lead: {item['lead_instrument']}")
        if item.get('accompaniment'):
            parts.append(f"Accompaniment: {item['accompaniment']}")
        if item.get('tempo_and_rhythm'):
            parts.append(f"Tempo: {item['tempo_and_rhythm']}")
        if item.get('vocal_presence'):
            parts.append(f"Vocals: {item['vocal_presence']}")
        if item.get('production_quality'):
            parts.append(f"Quality: {item['production_quality']}")
        return " | ".join(parts)
    
    def write_jsonl_gz(items, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            for item in items:
                entry = {
                    'audio_path': item['audio_path'],
                    'text': format_prompt(item),
                    'duration': 10.0,
                    'sample_rate': 32000,
                    'youtube_id': item.get('youtube_id', '')
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    write_jsonl_gz(train_data, f"{output_dir}/train.jsonl.gz")
    write_jsonl_gz(valid_data, f"{output_dir}/valid.jsonl.gz")
    
    print(f"✅ Created manifests:")
    print(f"   Train: {len(train_data)} samples → {output_dir}/train.jsonl.gz")
    print(f"   Valid: {len(valid_data)} samples → {output_dir}/valid.jsonl.gz")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to enriched metadata JSON')
    parser.add_argument('--output', required=True, help='Output directory for manifests')
    args = parser.parse_args()
    create_manifests(args.input, args.output)