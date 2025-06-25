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
            #message_window_size=6
        )
        self.agentmsg = ChatAgent(
            system_message="你是一个专业的信息提取引擎，严格按要求返回结构化数据",
            model=self.model,
            #message_window_size=6
        )

    def _init_files(self):
        """初始化文件结构"""
        # 关键内容文件结构
        # 初始化核心知识库文件
        if not os.path.exists(self.core_file):
            with open(self.core_file, "w", encoding="utf-8") as f:
                # 使用符合规范的字典结构初始化
                initial_data = {
                    "entities": [],
                    "keywords": [],
                    "summary": ""
                }
                json.dump(initial_data, f, ensure_ascii=False, indent=2)

        if not os.path.exists(self.style_file):
            with open(self.style_file, "w", encoding="utf-8") as f:
                # 使用符合规范的字典结构初始化
                initial_data = {
                "sentence_structure": [],
                "word_preference": [],
                "emotional_tone": "",
                "style_category": ""
            }
                json.dump(initial_data, f, ensure_ascii=False, indent=2)
        # 语言风格文件结构
        if not os.path.exists(self.style_file):
            with open(self.style_file, "w", encoding="utf-8") as f:
                json.dump([], f)  # 保存为数组格式

    def save_content(self, content_data: dict):
        """保存关键内容（合并实体/关键词 + 智能压缩摘要）"""
        # 读取现有数据
        with open(self.core_file, "r+", encoding="utf-8") as f:
            try:
                core_data = json.load(f)
            except json.JSONDecodeError:
                core_data = {"entities": [], "keywords": [], "summary": ""}

            # 合并实体和关键词（直接追加）
            core_data["entities"].extend(content_data.get("entities", []))
            core_data["keywords"].extend(content_data.get("keywords", []))
            core_data["summary"] = f"{core_data['summary']}\n{content_data.get('summary')}".strip()
            core_data["entities"] = self.clean_list(core_data["entities"])
            core_data["keywords"] = self.clean_list(core_data["keywords"])
            core_data["summary"] = self.compress_text(core_data["summary"])

            # 写回文件
            f.seek(0)
            f.truncate()
            json.dump(core_data, f, ensure_ascii=False, indent=2)


    def save_style(self, style_data: dict):
        """保存语言风格（直接追加）"""
        with open(self.style_file, "r+", encoding="utf-8") as f:
            try:
                style_datas = json.load(f)
            except json.JSONDecodeError:
                style_data = {"sentence_structure": [], "word_preference": [], "emotional_tone": "", "style_category": ""}

            # 合并实体和关键词（直接追加）
            style_datas["sentence_structure"].extend(style_data.get("sentence_structure", []))
            style_datas["word_preference"].extend(style_data.get("word_preference", []))
            style_datas["emotional_tone"] = f"{style_datas['emotional_tone']}\n{style_data.get['emotional_tone']}".strip()
            style_datas["style_category"] = f"{style_datas['style_category']}\n{style_data.get['style_category']}".strip()
            style_datas["sentence_structure"] = self.clean_list(style_datas["sentence_structure"])
            style_datas["word_preference"]["keywords"] = self.clean_list(style_datas["word_preference"])
            style_datas["emotional_tone"] = self.compress_text(style_datas["emotional_tone"])
            style_datas["style_category"] = self.compress_text(style_datas["style_category"])

            # 写回文件
            f.seek(0)
            f.truncate()
            json.dump(style_datas, f, ensure_ascii=False, indent=2)
        with open(self.style_file, "r+", encoding="utf-8") as f:
            style_data = json.load(f)
            style_data.append(style_data)
            f.seek(0)
            json.dump(style_data, f, ensure_ascii=False, indent=2)
            
    # 增强版清洗逻辑
    def clean_list(self, raw_list: list) -> list:
        """使用大模型清洗实体列表"""
        prompt = f'''请严格按JSON列表格式返回去重后的结果，不要包含任何解释：
            原始数据：{raw_list}
            要求：
            1. 保持元素顺序不变
            2. 仅移除完全重复项
            3. 保留原始数据结构'''
        
        response = self.agentmsg.step(prompt)
        try:
            return json.loads(response.output)  # 强制JSON解析
        except json.JSONDecodeError:
            return raw_list  # 失败时返回原始数据

    # 摘要压缩改进
    def compress_text(self, text: str) -> str:
        prompt = f'''请用不超过1000字符的连贯文本概括以下内容，保持专业书面语：
            {text}'''
        return self.agentmsg.step(prompt).output.strip()

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
            #message_window_size=6
        )
        self.agentmsg = ChatAgent(
            system_message="你是一个专业的信息提取引擎，严格按要求返回结构化数据",
            model=self.model,
            #message_window_size=6
        )
    
    def _extract_key_contents(self, text: str) -> dict:
        """使用 DeepSeek API 提取结构化关键信息"""
        # 构建结构化提取指令
        prompt = f"""请从以下文本中提取关键信息，按JSON格式返回：
        {text}

        要求：
        1. entities: 识别所有专业名词与客观事实（人物/地点/机构/技术术语），用中文列出
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
                #"timestamp": datetime.now().isoformat(),
                "entities": result.get("entities", []),
                "keywords": result.get("keywords", []),
                "summary": result.get("summary", ""),
            }
        except Exception as e:
            print(f"关键信息提取失败: {str(e)}")
            return {
                #"timestamp": datetime.now().isoformat(),
                "entities": [],
                "keywords": [],
                "summary": "信息提取失败",
            }
    
    def _analyze_linguistic_style(self, text: str) -> dict:
        """使用 DeepSeek 深度分析语言风格"""
        # 构建专业分析指令
        prompt = f"""请分析以下文本的语言风格特征，按JSON格式返回：
            {text}

            分析维度：
            1. sentence_structure: 句式结构特征（长句/短句/疑问句/排比句等）
            2. word_preference: 用词偏好（专业术语/口语词汇/形容词使用频率）
            3. emotional_tone: 情感倾向（positive/neutral/negative）
            4. style_category: 整体风格分类（学术/正式/口语/文学/技术文档）

            输出要求：
            - 使用中文描述
            - 按以下JSON格式返回：
            {{
                "sentence_structure": [],
                "word_preference": [],
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
                "emotional_tone": result.get("emotional_tone", "neutral"),
                "style_type": result.get("style_category", "正式")
            }
        except Exception as e:
            print(f"语言风格分析失败: {str(e)}")
            return {
                "sentence_features": [],
                "vocab_features": [],
                "emotional_tone": "neutral",
                "style_type": "未知"
            }
    