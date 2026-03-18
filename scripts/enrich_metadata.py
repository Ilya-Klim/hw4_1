import json
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a music metadata assistant. Convert the given raw music description into a structured JSON format.
Strictly follow this JSON schema:
{
  "description": "string (refined original)",
  "general_mood": "string",
  "genre_tags": ["string"],
  "lead_instrument": "string",
  "accompaniment": "string",
  "tempo_and_rhythm": "string",
  "vocal_presence": "string",
  "production_quality": "string"
}
Do not output any markdown or extra text, only valid JSON.
"""


def enrich_caption(raw_caption):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Raw caption: {raw_caption}"}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error processing caption: {e}")
        return None


def main():
    with open('data/raw/downloaded_meta.json', 'r') as f:
        data = json.load(f)
    
    output_dir = Path('data/metadata')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    enriched_data = []

    for item in data:
        youtube_id = item['youtube_id']
        json_path = output_dir / f"{youtube_id}.json"
        
        if json_path.exists():
            continue
            
        enriched = enrich_caption(item['original_caption'])
        if enriched:
            enriched["audio_path"] = item['path']
            
            with open(json_path, 'w') as f:
                json.dump(enriched, f, indent=2)
            enriched_data.append(enriched)
            print(f"Processed {youtube_id}")
        else:
            print(f"Skipped {youtube_id} due to error")

    with open('data/metadata/all_enriched.json', 'w') as f:
        json.dump(enriched_data, f, indent=2)


if __name__ == "__main__":
    main()