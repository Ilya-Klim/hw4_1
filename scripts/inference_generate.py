import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import json

MODEL_PATH = "path/to/your/checkpoint" # Или hf repo id если залили
model = MusicGen.get_pretrained('facebook/musicgen-small') # Замените на загрузку своих весов
model.set_generation_params(duration=15) # 10-15 секунд

# Ваши промпты из задания
PROMPTS = [
    {
  "description": "An epic and triumphant orchestral soundtrack featuring powerful brass and a sweeping string ensemble, driven by a fast march-like rhythm and an epic background choir, recorded with massive stadium reverb.",
  "general_mood": "Epic, heroic, triumphant, building tension",
  "genre_tags": ["Cinematic", "Orchestral", "Soundtrack"],
  "lead_instrument": "Powerful brass section (horns, trombones)",
  "accompaniment": "Sweeping string ensemble, heavy cinematic percussion, timpani",
  "tempo_and_rhythm": "Fast, driving, march-like rhythm",
  "vocal_presence": "Epic choir in the background (wordless chanting)",
  "production_quality": "High fidelity, wide stereo image, massive stadium reverb"
},
    {
  "description": "A relaxing lo-fi hip-hop instrumental with a muffled electric piano playing jazz chords over a dusty vinyl crackle, deep sub-bass, and a slow boom-bap drum loop.",
  "general_mood": "Relaxing, nostalgic, chill, melancholic",
  "genre_tags": ["Lo-Fi Hip Hop", "Chillhop", "Instrumental"],
  "lead_instrument": "Muffled electric piano (Rhodes) playing jazz chords",
  "accompaniment": "Dusty vinyl crackle, deep sub-bass, soft boom-bap drum loop",
  "tempo_and_rhythm": "Slow, laid-back, swinging groove",
  "vocal_presence": "None",
  "production_quality": "Lo-Fi, vintage, warm tape saturation, slightly muffled high frequencies"
},
    {
  "description": "An energetic progressive house dance track with a bright detuned synthesizer lead, pumping sidechain bass, and chopped vocal samples over a fast four-on-the-floor beat.",
  "general_mood": "Energetic, uplifting, party vibe, euphoric",
  "genre_tags": ["EDM", "Progressive House", "Dance"],
  "lead_instrument": "Bright, detuned synthesizer lead",
  "accompaniment": "Pumping sidechain bass, risers, crash cymbals",
  "tempo_and_rhythm": "Fast, driving, strict four-on-the-floor beat",
  "vocal_presence": "Chopped vocal samples used as a rhythmic instrument",
  "production_quality": "Modern, extremely loud, punchy, club-ready mix"
},  
    {
  "description": "An intimate acoustic folk instrumental featuring a fingerpicked acoustic guitar, light tambourine, and subtle upright bass, played in a gentle waltz-like rhythm.",
  "general_mood": "Intimate, warm, acoustic, peaceful",
  "genre_tags": ["Folk", "Acoustic", "Indie"],
  "lead_instrument": "Fingerpicked acoustic guitar",
  "accompaniment": "Light tambourine, subtle upright bass, distant ambient room sound",
  "tempo_and_rhythm": "Mid-tempo, gentle, waltz-like triple meter",
  "vocal_presence": "None",
  "production_quality": "Raw, organic, close-mic recording, natural room acoustics"
}, 
    {
  "description": "A dark cyberpunk synthwave instrumental driven by an aggressive distorted analog bass synthesizer, arpeggiated synth plucks, and a retro 80s drum machine.",
  "general_mood": "Dark, futuristic, gritty, mysterious",
  "genre_tags": ["Synthwave", "Cyberpunk", "Darkwave"],
  "lead_instrument": "Aggressive, distorted analog bass synthesizer",
  "accompaniment": "Arpeggiated synth plucks, retro 80s drum machine (gated snare)",
  "tempo_and_rhythm": "Driving, mid-tempo, robotic precision",
  "vocal_presence": "None",
  "production_quality": "Retro-futuristic, heavy compression, synthetic, 80s aesthetic"
}
]

def format_prompt(json_prompt):
    # Та же функция, что использовалась при обучении!
    parts = []
    parts.append(f"Description: {json_prompt.get('description', '')}")
    parts.append(f"Mood: {json_prompt.get('general_mood', '')}")
    parts.append(f"Genres: {', '.join(json_prompt.get('genre_tags', []))}")
    parts.append(f"Lead: {json_prompt.get('lead_instrument', '')}")
    parts.append(f"Accompaniment: {json_prompt.get('accompaniment', '')}")
    parts.append(f"Tempo: {json_prompt.get('tempo_and_rhythm', '')}")
    parts.append(f"Vocals: {json_prompt.get('vocal_presence', '')}")
    parts.append(f"Quality: {json_prompt.get('production_quality', '')}")
    return " | ".join(parts)

def main():
    for idx, prompt_json in enumerate(PROMPTS):
        text_prompt = format_prompt(prompt_json)
        print(f"Generating prompt {idx+1}: {text_prompt[:50]}...")
        
        # Генерация
        tokens = model.generate_unconditional(1) # Если нужно без текста
        # Но нам нужно с текстом:
        tokens = model.generate(text=[text_prompt], progress=True)
        
        # Сохранение
        for i, gen in enumerate(tokens):
            output_path = f"results/prompt_{idx+1}"
            # audio_write ожидает (path, wav, sample_rate, strategy)
            audio_write(output_path, gen.cpu(), model.sample_rate, strategy="loudness")
            print(f"Saved {output_path}.wav")

if __name__ == "__main__":
    with torch.no_grad():
        main()