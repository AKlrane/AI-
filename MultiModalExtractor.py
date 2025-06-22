# MultiModalExtractor.py

import os
import json
import whisper
import easyocr
import ffmpeg
import numpy as np

# === 加载模型 ===
print("🎯 正在加载模型...")
whisper_model = whisper.load_model("base")
ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
print("✅ 模型加载完毕。")

# === 音频处理函数 ===
def transcribe_audio(audio_path: str):
    if not os.path.isfile(audio_path):
        return None, "音频文件不存在"
    try:
        print(f"🎧 正在解码音频: {audio_path}")
        process = (
            ffmpeg
            .input(audio_path, threads=0)
            .output('pipe:', format='f32le', ac=1, ar='16000')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        out, err = process.communicate()
        audio_np = np.frombuffer(out, np.float32)
        print("🤖 Whisper 正在识别...")
        result = whisper_model.transcribe(audio_np, language='zh')
        text = result["text"].strip()
        print("✅ 识别结果：", text)
        return text, None
    except Exception as e:
        return None, f"音频识别失败：{e}"

# === 图像处理函数 ===
def extract_image_text(image_path: str):
    if not os.path.isfile(image_path):
        return None, "图像文件不存在"
    try:
        print(f"🖼 正在识别图像: {image_path}")
        result = ocr_reader.readtext(image_path, detail=0)
        text = '\n'.join(result).strip()
        print("✅ 图像识别结果：", text)
        return text, None
    except Exception as e:
        return None, f"图像识别失败：{e}"
