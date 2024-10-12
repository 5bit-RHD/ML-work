import os
import wave
import json
import vosk
import soundfile as sf

# Функция для транскрибации одного аудиофайла
def transcribe_audio(audio_file, model_dir):
    # Открытие аудиофайла
    with sf.SoundFile(audio_file) as f:
        if f.channels != 1:
            raise ValueError("Аудиофайл должен быть монофоническим (1 канал)")
        wf = wave.open(audio_file, "rb")

    # Инициализация модели Vosk
    model = vosk.Model(model_dir)
    rec = vosk.KaldiRecognizer(model, wf.getframerate())

    # Чтение аудио и распознавание текста
    result = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result.append(res.get('text', ''))

    # Добавление финального результата
    final_res = json.loads(rec.FinalResult())
    result.append(final_res.get('text', ''))

    # Соединение всех распознанных фрагментов
    transcription = ' '.join(result)
    wf.close()
    return transcription

# Функция для обработки всех файлов в папке
def transcribe_folder(input_dir, output_dir, model_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_file = os.path.join(root, file)

                try:
                    # Транскрибация аудиофайла
                    transcription = transcribe_audio(audio_file, model_dir)

                    # Сохранение результата в текстовый файл
                    output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(transcription)

                    print(f"Файл {file} транскрибирован и сохранён как {output_file}")

                except Exception as e:
                    print(f"Ошибка при обработке файла {file}: {e}")

# Основная функция
def main():
    input_dir = '/Users/daniil/Хакатоны/ЦП СВФО/data/ржд 1/ESC_DATASET_v1.2/hr_bot_clear'  # Папка с очищенными аудиофайлами
    output_dir = '/Users/daniil/Хакатоны/ЦП СВФО/transcrub/result'  # Папка для сохранения транскрибаций
    model_dir = '//Users/daniil/Хакатоны/ЦП СВФО/transcrub/vosk-model-small-ru-0.22'  # Папка с языковой моделью Vosk

    # Запуск транскрибации всех файлов в папке
    transcribe_folder(input_dir, output_dir, model_dir)

if __name__ == "__main__":
    main()




