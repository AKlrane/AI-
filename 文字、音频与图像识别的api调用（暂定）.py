import os
import requests
import whisper
import easyocr

# === DeepSeek API 配置 ===
API_KEY = "sk-4c90f15d2a9841a59f972df9a8f72f32"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# === 模型加载 ===
whisper_model = whisper.load_model("base")
ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)

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

# === 主程序 ===
def main():
    print("=== 🤖 DeepSeek 多模态对话助手 ===")
    print("输入文字聊天，输入 v 使用语音，输入 i 读取图像，输入 q 退出")

    while True:
        user_input = input("你：").strip()

        if user_input.lower() in ["q", "quit", "exit"]:
            print("拜拜~")
            break

        elif user_input.lower() == "v":
            audio_path = input("请输入 WAV 文件路径：").strip()
            if not os.path.isfile(audio_path):
                print("❌ 文件不存在，请重试。")
                continue
            user_input = transcribe_audio(audio_path)
            if not user_input:
                continue

        elif user_input.lower() == "i":
            image_path = input("请输入图像路径（支持中英文 OCR）：").strip()
            if not os.path.isfile(image_path):
                print("❌ 图像不存在，请重试。")
                continue
            user_input = extract_text_from_image(image_path)
            if not user_input:
                continue

        reply = ask_deepseek(user_input)
        print("AI：", reply)

if __name__ == "__main__":
    main()
