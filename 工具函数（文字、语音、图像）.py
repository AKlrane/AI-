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

# === 音频处理（不依赖系统 ffmpeg） ===
def transcribe_audio_stream(audio_path: str, output_path: str):
    print(f"\n[🔍] 正在读取音频文件：{audio_path}")
    if not os.path.isfile(audio_path):
        print("❌ 音频文件不存在。")
        return

    try:
        print("🎧 解码音频中（通过 ffmpeg-python）...")
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

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "input_type": "audio",
                "file": os.path.basename(audio_path),
                "text": text
            }, f, ensure_ascii=False, indent=2)
        print(f"[💾] 已保存到 {output_path}")
    except Exception as e:
        print(f"[❌] 音频识别失败：{e}")

# === 图像处理 ===
def extract_text_from_image(image_path: str, output_path: str):
    print(f"\n[🖼] 正在读取图像文件：{image_path}")
    if not os.path.isfile(image_path):
        print("❌ 图像文件不存在。")
        return

    try:
        result = ocr_reader.readtext(image_path, detail=0)
        text = '\n'.join(result).strip()
        print("✅ 图像识别结果：", text)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "input_type": "image",
                "file": os.path.basename(image_path),
                "text": text
            }, f, ensure_ascii=False, indent=2)
        print(f"[💾] 已保存到 {output_path}")
    except Exception as e:
        print(f"[❌] 图像识别失败：{e}")


