# === 图像识别函数（OCR）===
def extract_text_from_image(image_path: str) -> str:
    try:
        print(f"[🖼] 正在读取图片：{image_path}")
        result = ocr_reader.readtext(image_path, detail=0)
        text = '\n'.join(result).strip()
        print("[✅] 提取文字：", text)
        return text
    except Exception as e:
        print(f"[❌] 图像识别失败：{e}")
        return ""

# === 音频转文字 ===
def transcribe_audio(audio_path: str) -> str:
    try:
        print(f"[🎙] 正在识别音频文件：{audio_path}")
        result = whisper_model.transcribe(audio_path, language='zh')
        print("[✅] 识别结果：", result["text"])
        return result["text"]
    except Exception as e:
        print(f"[❌] 语音识别失败：{e}")
        return ""

# === 向 DeepSeek 提问 ===
def ask_deepseek(user_input: str) -> str:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个多模态 AI，擅长分析用户语音、图片和文字信息，具有哲学气质。"},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"[❌] 请求失败 {response.status_code}：{response.text}")
            return "(无法获取回复)"
    except Exception as e:
        print(f"[❌] API 错误：{e}")
        return "(请求异常)"
