import os
from openai import OpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)

def get_model(model_name: str) -> BaseChatModel:
    """
    根据模型名称返回对应的LLM模型实例
    
    Args:
        model_name: 模型名称，可选值为 "DeepSeek" 或 "通义千问"
        
    Returns:
        对应的LLM模型实例
    """
    if model_name == "DeepSeek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到DeepSeek API密钥，请确保DEEPSEEK_API_KEY环境变量已设置")
        
        # 使用ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
            streaming=True
        )
    
    elif model_name == "通义千问":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未找到通义千问API密钥，请确保DASHSCOPE_API_KEY环境变量已设置")
        
        # 使用通义千问的标准导入方式，结合LangChain的ChatOpenAI
        return ChatOpenAI(
            api_key=api_key,
            model="qwen-plus",
            temperature=0.7,
            streaming=True,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    else:
        raise ValueError(f"不支持的模型: {model_name}，请选择 'DeepSeek' 或 '通义千问'") 