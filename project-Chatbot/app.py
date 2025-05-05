import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 导入自定义模型工具
from src.models import get_model

# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="双模型聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = None

if "token_count" not in st.session_state:
    st.session_state.token_count = 0

if "prompt_tokens" not in st.session_state:
    st.session_state.prompt_tokens = 0
    
if "completion_tokens" not in st.session_state:
    st.session_state.completion_tokens = 0
    
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
    
if "model_name" not in st.session_state:
    st.session_state.model_name = "通义千问"  # 默认使用通义千问

# 页面标题
st.title("🤖 双模型聊天机器人")

# 侧边栏 - 模型选择和状态显示
with st.sidebar:
    st.header("设置")
    model_name = st.selectbox(
        "选择模型", 
        ["通义千问", "DeepSeek"],  # 通义千问在前
        index=0
    )
    
    # 更新模型选择
    if model_name != st.session_state.model_name:
        st.session_state.model_name = model_name
        # 切换模型时重置记忆
        st.session_state.memory = None
        
    # 添加记忆长度控制
    max_messages = st.slider("保留对话轮数", min_value=1, max_value=10, value=5)
    
    # 显示Token使用情况
    st.header("Token使用统计")
    
    # 使用列布局展示数据
    col1, col2 = st.columns(2)
    with col1:
        st.metric("提示词tokens", f"{st.session_state.prompt_tokens}")
        st.metric("总tokens", f"{st.session_state.token_count}")
    with col2:
        st.metric("完成tokens", f"{st.session_state.completion_tokens}")
        st.metric("总费用($)", f"{st.session_state.total_cost:.6f}")
    
    # 添加清除会话按钮
    if st.button("清除会话", type="primary"):
        st.session_state.messages = []
        st.session_state.memory = None
        st.session_state.token_count = 0
        st.session_state.prompt_tokens = 0
        st.session_state.completion_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# 显示聊天历史
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# 构建系统提示词
system_prompt = """你是一个专业、友好且有用的AI助手。
请直接回答问题，保持回答简洁明了。
如果你不知道答案，请诚实地说出来，不要编造信息。
不要在回答中使用{input}或{output}这样的字符。"""

# 创建直接的提示模板，避免使用复杂的格式化
prompt_template = f"{system_prompt}\n\n"

# 获取用户输入
user_input = st.chat_input("在这里输入您的问题...")

if user_input:
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.write(user_input)
    
    # 获取模型
    llm = get_model(st.session_state.model_name)
    
    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 处理历史记录 - 控制历史长度，只保留最近的对话
                max_pairs = max_messages  # 从侧边栏的滑块获取值
                recent_messages = st.session_state.messages[-min(len(st.session_state.messages), max_pairs*2-1):-1]
                
                # 构建完整提示文本
                full_prompt = prompt_template
                
                # 添加历史消息
                for msg in recent_messages:
                    prefix = "用户: " if msg["role"] == "user" else "助手: "
                    full_prompt += f"{prefix}{msg['content']}\n"
                
                # 添加当前问题
                full_prompt += f"用户: {user_input}\n助手: "
                
                # 直接调用invoke方法，避免使用generate可能造成的复杂结构解析问题
                response = llm.invoke(full_prompt)
                
                # 根据返回类型安全地提取文本内容
                if hasattr(response, 'content'):
                    # 如果是对象且有content属性
                    response_text = response.content
                elif isinstance(response, str):
                    # 如果直接返回字符串
                    response_text = response
                else:
                    # 如果是其他类型，尝试转换为字符串
                    response_text = str(response)
                
                # 字符数估算token
                prompt_chars = len(full_prompt)
                response_chars = len(response_text)
                
                # 估算token数 (根据经验值，平均每个字符约占0.6-0.8个token)
                estimated_prompt_tokens = int(prompt_chars * 0.7) 
                estimated_completion_tokens = int(response_chars * 0.7)
                estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
                
                # 计算估算费用 (使用GPT-3.5 Turbo标准价格作为参考)
                prompt_cost = estimated_prompt_tokens * 0.0015 / 1000
                completion_cost = estimated_completion_tokens * 0.002 / 1000
                total_cost = prompt_cost + completion_cost
                
                # 更新会话状态中的计数器
                st.session_state.prompt_tokens += estimated_prompt_tokens
                st.session_state.completion_tokens += estimated_completion_tokens
                st.session_state.token_count += estimated_total_tokens
                st.session_state.total_cost += total_cost
                
                # 检查回复中是否包含占位符，如果有则替换
                if "{input}" in response_text:
                    response_text = response_text.replace("{input}", user_input)
                if "{output}" in response_text:
                    response_text = response_text.replace("{output}", "")
                
                # 显示回答
                st.write(response_text)
                
                # 保存助手回答到历史
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # 初始化或更新记忆
                if st.session_state.memory is None:
                    st.session_state.memory = ConversationBufferMemory(
                        return_messages=True,
                        memory_key="history"
                    )
                
                # 更新记忆
                st.session_state.memory.chat_memory.add_user_message(user_input)
                st.session_state.memory.chat_memory.add_ai_message(response_text)
                
            except Exception as e:
                st.error(f"生成回答时发生错误: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.write("抱歉，我现在无法处理您的请求。请稍后再试。")
                st.session_state.messages.append({"role": "assistant", "content": "抱歉，我现在无法处理您的请求。请稍后再试。"}) 