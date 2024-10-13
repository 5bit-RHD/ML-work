import os
import time
import psutil
import numpy as np
import librosa
import stt
from sklearn.model_selection import train_test_split
import csv

# Функция для обучения модели
def train_model():
    os.system('stt-train --config config.yml \
        --train_files train.csv \
        --dev_files dev.csv \
        --test_files test.csv \
        --export_dir /Users/daniil/Хакатоны/ЦП СВФО/ML-work/Coqui-model/export/ \
        --test_output_file test_output.json \
        --log_level 0')

# Функция для транскрибации аудио
def transcribe_audio(model_path, scorer_path, audio_file):
    model = stt.Model(model_path)
    if scorer_path and os.path.exists(scorer_path):
        model.enableExternalScorer(scorer_path)
    
    # Загружаем аудио
    audio, sample_rate = librosa.load(audio_file, sr=16000)
    audio = (audio * 32767).astype(np.int16)
    
    # Транскрибация
    start_time = time.time()
    text = model.stt(audio)
    end_time = time.time()
    
    # Вычисление времени и использования памяти
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # В МБ
    inference_time = end_time - start_time
    
    print(f"Распознанный текст: {text}")
    print(f"Время инференса: {inference_time:.2f} секунд")
    print(f"Использование памяти: {memory_usage:.2f} МБ")

    return text

# Функция для оценки модели на тестовом наборе
def evaluate_model(test_csv, model_path, scorer_path):
    model = stt.Model(model_path)
    if scorer_path and os.path.exists(scorer_path):
        model.enableExternalScorer(scorer_path)
    
    references = []
    hypotheses = []
    
    with open(test_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            audio_file, transcript = row
            # Загружаем аудио
            audio, sample_rate = librosa.load(audio_file, sr=16000)
            audio = (audio * 32767).astype(np.int16)
            
            # Транскрибация
            text = model.stt(audio)
            
            references.append(transcript)
            hypotheses.append(text)
    
    # Вычисление WER
    from jiwer import wer
    error = wer(references, hypotheses)
    print(f"Word Error Rate (WER) на тестовом наборе: {error:.2f}")

# Основная функция
def main():
    # Обучение модели
    start_time = time.time()
    train_model()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Время обучения модели: {training_time / 60:.2f} минут")
    
    # Путь к обученной модели
    model_path = '/Users/daniil/Хакатоны/ЦП СВФО/ML-work/Coqui-model/export/output_graph.pb'
    scorer_path = '/Users/daniil/Хакатоны/ЦП СВФО/ML-work/Coqui-model/export/scorer.scorer'  # Если есть языковая модель
    
    # Оценка модели
    evaluate_model('test.csv', model_path, scorer_path)
    
    # Тестовое распознавание
    audio_file = input("Введите путь к аудиофайлу для распознавания: ")
    transcribe_audio(model_path, scorer_path, audio_file)

if __name__ == "__main__":
    main()
