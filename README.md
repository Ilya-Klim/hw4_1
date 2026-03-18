# Домашнее задание 4: Fine-tuning MusicGen

# Инструкция запуска
-  Клоинрование репозитория и установка зависимостей 
```  
    git clone https://github.com/Ilya-Klim/hw4_1.git
    cd hw4_1
    python -m venv venv
    source venv/bin/activate  
    pip install -r requirements.txt 

```
- Создайте файл .env в корне проекта

``` 
touch .env 
```
с содержанием 
    OPENAI_API_KEY=...
    WANDB_API_KEY=...
- Скачивание аудио: 
``` 
python scripts/download_musiccaps.py
```
- Обновить метаданные: 
```
python scripts/enrich_metadata.py
```
- Создать манифесты: 
``` 
python scripts/create_manifests.py \
    --input data/metadata/all_enriched.json \
    --output data/manifests 
```
- Запуск: 
перейдите в директорию к udiocraft
``` 
cd /path/to/audiocraft
```
```
python -m audiocraft.train \
    --config-dir=config \
    --config-name=config \
    conditioner=text2music \
    dataset.train.merge_text_p=0.3 \
    dataset.train.drop_desc_p=0.25 \
    dataset.train.drop_other_p=0.25 \
    experiment.name=musicgen_hw4 \
    experiment.dir=/path/to/hw4_1/experiments \
    solver.musicgen.musicgen_finetune.training.max_steps=5000 \
    solver.musicgen.musicgen_finetune.training.batch_size=4 \
    solver.musicgen.musicgen_finetune.training.learning_rate=1e-5 \
    logging.log_wandb=true \
    wandb.project=musicgen-hw4 
```

- Возвращаемся к корню проекта и делаем инференс
```
cd /path/to/hw4_1
```
```
python scripts/inference_generate.py \
    --model_path ./experiments/musicgen_hw4/checkpoints/checkpoint_last.pt
```
- Результаты в results

# Структура проекта
```
hw4_1/
├── scripts/
│   ├── download_musiccaps.py      # скрапт для скачивания аудио
│   ├── enrich_metadata.py         # обогащение мета
│   ├── create_manifests.py        # Создание манифестов
│   └── inference_generate.py      # Генерация треков
├── configs/training/
│   └── musicgen_finetune.yaml     # Конфиг обучения
├── data/   
│   ├── raw/                       # .csv данные
│   ├── audio/                     # .wav файлы
│   ├── metadata/                  # JSON метаданные
│   └── manifests/                 # .jsonl.gz
├── results/
│   └── prompt_*.wav               # Сгенерированное аудио
├── experiments/             
│   └── musicgen_hw4/checkpoints/  # Веса модели
├── audiocraft
│   ├── ...
│  
├── .gitignore
├── requirements.txt
├── README.md                     
```