import sqlite3
import requests
import json
import time
import concurrent.futures  # 用于异步执行
from datetime import datetime

API_KEY = "sk-ef3dcb4f0d754e93a0369dbbba1bda4c"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 初始化数据库（只需运行一次）
def init_content_db():
    conn = sqlite3.connect('chat_content.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT DEFAULT 'New Chat',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT,
        is_important BOOLEAN DEFAULT FALSE,
        tokens INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    )
    ''')
    conn.commit()
    conn.close()

def init_style_db():
    conn = sqlite3.connect('language_style.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS styles (
        style_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        message_id INTEGER,
        tone TEXT,
        sentiment TEXT,
        complexity INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions (session_id)
    )
    ''')
    conn.commit()
    conn.close()

# 初始化两个数据库
init_content_db()
init_style_db()

def chat(api_key):
    # 初始化会话
    content_conn = sqlite3.connect('chat_content.db')
    style_conn = sqlite3.connect('language_style.db')
    content_cursor = content_conn.cursor()
    style_cursor = style_conn.cursor()
    
    # 创建新会话
    content_cursor.execute("INSERT INTO sessions (title) VALUES ('Active Chat')")
    session_id = content_cursor.lastrowid
    style_cursor.execute("INSERT INTO sessions DEFAULT VALUES")
    content_conn.commit()
    style_conn.commit()
    
    while True:
        # 用户输入
        user_input = input("\nYou: ")
        if user_input.lower() in ('exit', 'quit'):
            break
        
        # 手动标记重要消息
        is_important = "记住" in user_input or "重要" in user_input
        save_content(session_id, "user", user_input, is_important)

        # 自动检测重要性（每3轮触发一次）
        if len(load_content(session_id)) % 3 == 0:
            auto_mark_importance(session_id, api_key)

        # 加载对话历史
        messages = load_content(session_id, max_tokens=3500)
        
        # Token超限时触发压缩
        if sum(count_tokens(msg["content"]) for msg in messages) > 3500:
            summarize_old_messages(session_id, api_key)
            messages = load_content(session_id, max_tokens=3500)

        # 保存内容
        message_id = save_content(session_id, "user", user_input)
        
        # 分析并保存语言风格
        save_style(session_id, message_id, user_input, api_key)

        # 调用DeepSeek API（流式）
        print("AI: ", end="", flush=True)  # 不换行，准备流式输出
        full_response = ""
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.7,
                "stream": True  # 启用流式
            },
            stream=True  # 确保requests支持流式
        )
        
        # 处理流式响应
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data:"):
                        json_data = decoded_line[5:].strip()
                        if json_data != "[DONE]":
                            chunk = json.loads(json_data)
                            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                                content = chunk["choices"][0]["delta"]["content"]
                                full_response += content
                                print(content, end="", flush=True)  # 逐字输出
            print()  # 换行
            save_content(session_id, "assistant", full_response)  # 保存完整回复
        else:
            print(f"API Error: {response.text}")
    
    content_conn.close()
    style_conn.close()

def save_content(session_id, role, content, is_important=False):
    conn = sqlite3.connect('chat_content.db')
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO messages (session_id, role, content, is_important)
        VALUES (?, ?, ?, ?)""",
        (session_id, role, content, is_important)
    )
    message_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return message_id  # 返回消息ID，用于关联风格记录

def load_content(session_id, max_tokens=3000):
    """加载消息，优先返回重要信息"""
    conn = sqlite3.connect('chat_content.db')
    cursor = conn.cursor()
    
    # 重要消息（不受Token限制）
    cursor.execute('''
    SELECT role, content FROM messages 
    WHERE session_id = ? AND is_important = TRUE
    ORDER BY message_id
    ''', (session_id,))
    important_msgs = [{"role": r, "content": c} for r, c in cursor.fetchall()]
    used_tokens = sum(count_tokens(m["content"]) for m in important_msgs)
    
    # 普通消息（剩余Token配额）
    normal_msgs = []
    if used_tokens < max_tokens:
        cursor.execute('''
        SELECT role, content FROM messages 
        WHERE session_id = ? AND is_important = FALSE
        ORDER BY message_id DESC
        LIMIT 100  -- 防止加载过多
        ''', (session_id,))
        
        remaining_tokens = max_tokens - used_tokens
        for role, content in cursor.fetchall():
            msg_tokens = count_tokens(content)
            if remaining_tokens - msg_tokens < 0:
                break
            normal_msgs.insert(0, {"role": role, "content": content})  # 保持时序
            remaining_tokens -= msg_tokens
    
    conn.close()
    return important_msgs + normal_msgs

def summarize_old_messages(session_id, api_key):
    """压缩非重要旧消息"""
    conn = sqlite3.connect('chat_content.db')
    cursor = conn.cursor()
    
    # 获取待摘要的普通消息（排除最近5条）
    cursor.execute('''
    SELECT role, content FROM messages 
    WHERE session_id = ? AND is_important = FALSE
    ORDER BY message_id DESC
    LIMIT 20 OFFSET 5
    ''', (session_id,))
    old_messages = [{"role": r, "content": c} for r, c in cursor.fetchall()]
    
    if not old_messages:
        return
    
    # 生成摘要
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "请用1句话总结以下对话的通用背景（排除重要信息）："},
                {"role": "user", "content": "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)}
            ],
            "temperature": 0.3
        }
    )
    
    if response.status_code == 200:
        summary = response.json()["choices"][0]["message"]["content"]
        save_content(session_id, "system", f"历史背景摘要：{summary}")
        
        # 删除已摘要的消息（保留重要消息和最近5条普通消息）
        cursor.execute('''
        DELETE FROM messages 
        WHERE session_id = ? AND is_important = FALSE AND message_id IN (
            SELECT message_id FROM messages 
            WHERE session_id = ? AND is_important = FALSE
            ORDER BY message_id DESC
            LIMIT 20 OFFSET 5
        )
        ''', (session_id, session_id))
        conn.commit()
    conn.close()

def auto_mark_importance(session_id, api_key):
    """自动标记重要消息"""
    conn = sqlite3.connect('chat_content.db')
    cursor = conn.cursor()
    
    # 检查未标记的近期消息
    cursor.execute('''
    SELECT message_id, content FROM messages 
    WHERE session_id = ? AND is_important = FALSE
    ORDER BY message_id DESC
    LIMIT 3
    ''', (session_id,))
    
    for msg_id, content in cursor.fetchall():
        # 规则1：包含关键词
        if any(kw in content for kw in ["密码", "过敏", "身份证号"]):
            cursor.execute('UPDATE messages SET is_important = TRUE WHERE message_id = ?', (msg_id,))
        
        # 规则2：模型判断
        else:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "如果以下内容包含需要长期记忆的关键信息（如个人信息、重要事实），回答YES，否则回答NO："},
                        {"role": "user", "content": content}
                    ],
                    "temperature": 0.0
                }
            )
            if response.json()["choices"][0]["message"]["content"].strip() == "YES":
                cursor.execute('UPDATE messages SET is_important = TRUE WHERE message_id = ?', (msg_id,))
    
    conn.commit()
    conn.close()

def count_tokens(text):
    """简易Token估算（实际应用建议使用tiktoken）"""
    return len(text.split())  # 中文大致1字=1.3 Token

def analyze_style(text, api_key):
    try:
        # 调用DeepSeek API分析风格
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """你是一个专业的语言风格分析器。请按以下JSON格式分析文本特征：
                        - tone: [formal/casual/technical/enthusiastic]
                        - sentiment: [positive/neutral/negative]
                        - complexity: 1-10的整数"""
                    },
                    {
                        "role": "user",
                        "content": f"分析以下文本风格：\n{text}"
                    }
                ],
                "temperature": 0.3,  # 保持分析结果稳定
                "max_tokens": 200
            }
        )

        # 检查响应状态码
        if response.status_code != 200:
            print(f"API请求失败: {response.status_code}")
            print(f"响应内容: {response.text}")
            return default_style()

        # 检查响应内容是否为空
        if not response.text:
            print("API返回内容为空")
            return default_style()

        # 解析API返回的JSON数据
        result = response.json()
        if "choices" not in result or not result["choices"]:
            print("API返回格式无效")
            return default_style()

        # 提取并标准化风格数据
        style_data = json.loads(result["choices"][0]["message"]["content"])
        return {
            "tone": style_data.get("tone", "formal").lower(),
            "sentiment": style_data.get("sentiment", "neutral").lower(),
            "complexity": max(1, min(10, int(style_data.get("complexity", 5))))
        }

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {str(e)}")
        print(f"原始响应内容: {response.text}")
        return default_style()
    except Exception as e:
        print(f"风格分析异常: {str(e)}")
        return default_style()

    
def default_style():
    """默认风格（API调用失败时使用）"""
    return {
        "tone": "formal",
        "sentiment": "neutral",
        "complexity": 5
    }

def save_style(session_id, message_id, text, api_key):
    style = analyze_style(text, api_key)
       
    # 数据库操作
    conn = None
    conn = sqlite3.connect('language_style.db', timeout=10)
    cursor = conn.cursor()
        
    cursor.execute(
        """INSERT OR REPLACE INTO styles 
        (session_id, message_id, tone, sentiment, complexity) 
        VALUES (?, ?, ?, ?, ?)""",
        (session_id, message_id, 
         style["tone"], 
         style["sentiment"], 
         style["complexity"])
    )
    conn.commit()
    conn.close()

# 启动对话
if __name__ == "__main__":
    API_KEY = "sk-1e739b6827fa443fb28da14b8c006365"
    chat(API_KEY)