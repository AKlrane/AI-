import os
import json
from datetime import datetime
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.agents import ChatAgent


# ====================== 配置初始化 ======================
load_dotenv(".env", encoding="utf-8-sig")
assert os.getenv("DEEPSEEK_API_KEY"), "请在 .env 文件中设置 DEEPSEEK_API_KEY"

# 初始化模型
class DataManager:
    def __init__(self):
        self.core_file = "core_data.json"
        self.style_file = "linguistic_style.json"
        self._init_files()

    def _init_files(self):
        """初始化文件结构"""
        # 关键内容文件结构
        if not os.path.exists(self.core_file):
            with open(self.core_file, "w", encoding="utf-8") as f:
                json.dump([], f)  # 保存为数组格式

        # 语言风格文件结构
        if not os.path.exists(self.style_file):
            with open(self.style_file, "w", encoding="utf-8") as f:
                json.dump([], f)  # 保存为数组格式

    def save_content(self, content_data: dict):
        """保存关键内容（直接追加）"""
        with open(self.core_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(content_data)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_style(self, style_data: dict):
        """保存语言风格（直接追加）"""
        with open(self.style_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(style_data)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)

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
            message_window_size=6
        )
        self.agentmsg = ChatAgent(
            system_message="你是一个专业的信息提取引擎，严格按要求返回结构化数据",
            model=self.model,
            message_window_size=6
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
    
    def _analyze_linguistic_style(self, text: str) -> dict:
        """使用 DeepSeek 深度分析语言风格"""
        # 构建专业分析指令
        prompt = f"""请分析以下文本的语言风格特征，按JSON格式返回：
            {text}

            分析维度：
            1. sentence_structure: 句式结构特征（长句/短句/疑问句/排比句等）
            2. word_preference: 用词偏好（专业术语/口语词汇/形容词使用频率）
            3. complexity: 句子复杂度（1-5评分，1为简单，5为复杂）
            4. emotional_tone: 情感倾向（positive/neutral/negative）
            5. style_category: 整体风格分类（学术/正式/口语/文学/技术文档）

            输出要求：
            - 使用中文描述
            - 按以下JSON格式返回：
            {{
                "sentence_structure": [],
                "word_preference": [],
                "complexity": 0,
                "emotional_tone": "",
                "style_category": ""
            }}"""

        try:
            # 获取结构化响应
            response = self.agent.step(prompt)
            response_text = response.msgs[0].content
            
            # 提取JSON部分（容错处理）
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            result = json.loads(json_str)
            
            # 转换数据结构
            return {
                "sentence_features": result.get("sentence_structure", []),
                "vocab_features": result.get("word_preference", []),
                "complexity_level": result.get("complexity", 3),
                "emotional_tone": result.get("emotional_tone", "neutral"),
                "style_type": result.get("style_category", "正式")
            }
        except Exception as e:
            print(f"语言风格分析失败: {str(e)}")
            return {
                "sentence_features": [],
                "vocab_features": [],
                "complexity_level": 3,
                "emotional_tone": "neutral",
                "style_type": "未知"
            }
    