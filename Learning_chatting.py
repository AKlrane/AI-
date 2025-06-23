import os
import json
from datetime import datetime
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

from Data_process import DataManager, DeepSeekProcessor

process = DeepSeekProcessor

def process_conversation(user_input: str):
    """处理单轮对话（简化存储版）"""
    # 获取AI回复
    response = process.agent.step(user_input)
    ai_response = response.msgs[0].content
    print(f"\n助手: {ai_response}")
      
    # 信息提取与保存
    try:
        # 提取并保存关键内容
        key_content = process._extract_key_contents(ai_response)
        process.dm.save_content(key_content)
          
        # 分析并保存语言风格
        style_data = process._analyze_linguistic_style(ai_response)
        process.dm.save_style(style_data)
    except Exception as e:
        print(f"数据保存失败: {str(e)}")
        # 写入错误日志
        with open("error.log", "a") as f:
            f.write(f"{datetime.now()} - {str(e)}\n")

# ====================== 主程序 ======================
def main():
    print("输入对话内容开始交流 (输入 'exit' 退出)")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("对话已保存，退出系统")
                break
            process_conversation(user_input)
        except KeyboardInterrupt:
            print("\n对话已自动保存")
            break
        except Exception as e:
            print(f"系统错误: {str(e)}")
            continue

if __name__ == "__main__":
    main()