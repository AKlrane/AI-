import os
import json
from datetime import datetime
from dotenv import load_dotenv
from camel.messages import BaseMessage
from camel.types import RoleType
from camel.memories import (
    ChatHistoryBlock,
    LongtermAgentMemory,
    MemoryRecord,
    ScoreBasedContextCreator,
    VectorDBBlock,
)
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent

# ====================== 配置初始化 ======================
load_dotenv(".env", encoding="utf-8-sig")
assert os.getenv("DEEPSEEK_API_KEY"), "请在 .env 文件中设置 DEEPSEEK_API_KEY"

# ====================== 数据管理系统 ======================
class DataManager:
    def __init__(self):
        self.memory = "chatting_memory.json"
        self._init_files()

    def _init_files(self):
        """初始化文件结构"""
        # 关键内容文件结构
        if not os.path.exists(self.memory):
            with open(self.memory, "w", encoding="utf-8") as f:
                json.dump([], f)  # 保存为数组格式

    def save_content(self, content_data: dict):
        """保存关键内容（直接追加）"""
        with open(self.memory, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(content_data)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
def format_data_to_str(data: any, name: str) -> str:
    formatted = [f"**{name}**:"]
    
    if isinstance(data, dict):
        # 处理字典
        formatted += [f"- {k}: {v}" for k, v in data.items()]
    elif isinstance(data, list):
        # 处理列表（元素可能是字典或普通值）
        for idx, item in enumerate(data, 1):
            if isinstance(item, dict):
                formatted.append(f"  Item {idx}:")
                formatted += [f"    - {k}: {v}" for k, v in item.items()]
            else:
                formatted.append(f"  - {item}")
    else:
        formatted.append(f"- {data}")
    
    return "\n".join(formatted)

# ====================== 功能处理器 ======================
class DeepSeekProcessor:
    def __init__(self):
        #jieba.initialize()
        self.dm = DataManager()
        self.conversation_buffer = []
        
        # 初始化模型
        self.model = ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=ModelType.DEEPSEEK_CHAT,
            model_config_dict={"temperature": 0.7, "max_tokens": 1024},
            url="https://api.deepseek.com/v1",
        )
        self.agent = ChatAgent(
            system_message="你是一个专业的智能助理",
            model=self.model,
            message_window_size=64
        )
        self.agentmsg = ChatAgent(
            system_message="你是一个专业的信息提取引擎，严格按要求返回结构化数据",
            model=self.model,
            message_window_size=64
        )
    
    def _extract_key_contents(self, text: str) -> dict:
        """使用 DeepSeek API 提取结构化关键信息"""
        # 构建结构化提取指令
        prompt = f"""请从以下文本中提取关键信息，按JSON格式返回：
        {text}

        要求：
        1. entities: 识别所有专业名词（人物/地点/机构/技术术语），用中文列出
        2. keywords: 提取5个最具代表性的关键词，用中文
        3. summary: 生成100字内的中文摘要
        4. 输出格式：
        {{
            "entities": [],
            "keywords": [],
            "summary": ""
        }}"""

        # 调用API获取结构化响应
        try:
            response = self.agentmsg.step(prompt)
            response_text = response.msgs[0].content
            
            # 解析JSON（兼容格式错误）
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "entities": result.get("entities", []),
                "keywords": result.get("keywords", []),
                "summary": result.get("summary", ""),
                "raw_text": text
            }
        except Exception as e:
            print(f"关键信息提取失败: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "entities": [],
                "keywords": [],
                "summary": "信息提取失败",
                "raw_text": text
            }
    
    def process_conversation(self, user_input: str):
        """处理单轮对话（简化存储版）"""
        with open('core_data.json', 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        with open('linguistic_style.json', 'r', encoding='utf-8') as f:
            data_b = json.load(f)
        with open('chatting_memory.json', 'r', encoding='utf-8') as f:
            data_c = json.load(f)
        try:
            with open('text.json', 'r', encoding='utf-8') as f:
                data_d = json.load(f)
                read_talk = 1
            os.remove('text.json')   
        except Exception as e:
            read_talk = 0
            pass

        # 获取AI回复
        data_context = "\n\n".join([
        format_data_to_str(data_a, "Dataset A"),
        format_data_to_str(data_b, "Dataset B"),
        format_data_to_str(data_c, "Dataset C"),
        #format_data_to_str(data_d, "Dataset D")if read_talk
    ])
        system_content = f"""
        你是一个数据分析助手，可以访问以下数据集：{data_context}请严格根据这些数据回答用户问题。
        """
        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=user_input  # 示例问题
            )
        sys_msg = BaseMessage.make_assistant_message(
            role_name="Data Assistant",
            content=system_content
            )
        camel_agent = ChatAgent(
            system_message=sys_msg,
            model=self.model,
            )
        response = camel_agent.step(user_msg)
        ai_response = response.msgs[0].content
        print(f"\n助手: {ai_response}")
        
        # 信息提取与保存
        try:
            # 提取并保存关键内容
            key_content = self._extract_key_contents(ai_response)
            self.dm.save_content(key_content)

        except Exception as e:
            print(f"数据保存失败: {str(e)}")
            # 写入错误日志
            with open("error.log", "a") as f:
                f.write(f"{datetime.now()} - {str(e)}\n")

# ====================== 主程序 ======================
def main():
    processor = DeepSeekProcessor()
    print("输入对话内容开始交流 (输入 'exit' 退出)")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'exit':
                print("对话已保存，退出系统")
                break
            processor.process_conversation(user_input)
        except KeyboardInterrupt:
            print("\n对话已自动保存")
            break
        except Exception as e:
            print(f"系统错误: {str(e)}")
            continue

if __name__ == "__main__":
    main()
