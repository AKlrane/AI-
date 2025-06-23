import json
import os
from datetime import datetime
from typing import List, Dict
from Data_process import DataManager, DeepSeekProcessor

class BatchRestoreProcessor:
    def __init__(self):
        self.batch_size = 32  # 可调整批次大小
        self.memory_file = "memory.json"
        self._validate_files()

    def _validate_files(self):
        """验证必要文件存在"""
        if not os.path.exists(self.memory_file):
            raise FileNotFoundError(f"历史记录不存在")
        # 初始化目标文件
        open("core_data.json", 'a').close()
        open("linguistic_style.json", 'a').close()

    def _load_filtered_messages(self) -> List[Dict]:
        """加载并过滤用户消息"""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [msg for msg in data["messages"] 
                if msg.get("sender") == "我" and msg.get("content")]

    def _batch_generator(self, messages: List[Dict]):
        """生成批量数据块"""
        for i in range(0, len(messages), self.batch_size):
            yield messages[i:i+self.batch_size]

    def _process_batch(self, batch: List[Dict]) -> tuple:
        """处理单个批次"""
        # 合并批次内容
        combined_text = "\n".join([msg["content"] for msg in batch])
        
        # 提取关键内容
        core_data = DeepSeekProcessor._extract_key_contents(combined_text)
        
        # 分析语言风格
        style_data = DeepSeekProcessor._analyze_linguistic_style(combined_text)
        
        return core_data, style_data

    def execute(self):
        """执行完整处理流程"""
        messages = self._load_filtered_messages()
        total_batches = len(messages) // self.batch_size + (1 if len(messages) % self.batch_size else 0)
        
        print(f"开始处理 {len(messages)} 条用户消息，共 {total_batches} 个批次")
        
        for batch_index, batch in enumerate(self._batch_generator(messages), 1):
            try:
                # 处理批次
                core_data, style_data = self._process_batch(batch)
                
                # 添加元数据
                core_data["batch_index"] = batch_index
                style_data["batch_size"] = len(batch)
                
                # 保存数据
                DataManager.save_content(core_data)
                DataManager.save_style(style_data)
                
                print(f"进度: {batch_index}/{total_batches} 批次", end='\r')
            except Exception as e:
                print(f"\n批次 {batch_index} 处理失败: {str(e)}")
                continue
        
        print(f"\n处理完成！核心数据文件：core_data.json，风格数据文件：linguistic_style.json")

if __name__ == "__main__":
    processor = BatchRestoreProcessor()
    processor.execute()
