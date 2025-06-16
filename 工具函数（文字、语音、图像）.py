import whisper
import easyocr

# === 初始化模型 ===
whisper_model = whisper.load_model("base")
ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

def transcribe_audio(audio_path: str) -> str:
    """
    将音频（WAV 格式）转成中文文字
    :param audio_path: 音频文件路径
    :return: 识别出的文字字符串
    """
    try:
        print(f"[🎙] 正在识别音频文件：{audio_path}")
        result = whisper_model.transcribe(audio_path, language='zh')
        print(f"[✅] 识别结果：{result['text']}")
        return result["text"]
    except Exception as e:
        print(f"[❌] 语音识别失败：{e}")
        return ""

def extract_text_from_image(image_path: str) -> str:
    """
    读取图像文件中的中文或英文文字
    :param image_path: 图像文件路径
    :return: 提取出的文字字符串
    """
    try:
        print(f"[🖼] 正在读取图像：{image_path}")
        result = ocr_reader.readtext(image_path, detail=0)
        text = '\n'.join(result).strip()
        print(f"[✅] 提取文字：{text}")
        return text
    except Exception as e:
        print(f"[❌] 图像识别失败：{e}")
        return ""
