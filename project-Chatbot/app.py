import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
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
    
    # Few-Shot模式开关
    use_few_shot = st.checkbox("启用Few-Shot学习", value=True)
    
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

# 1. 定义示例数据集
examples = [
    {"input": "今天星期几？", 
     "output": "今天是星期二。\n Today is Tuesday."},
    {"input": "我刚才问你什么问题？", 
     "output": "你刚才问我今天星期几。\n You asked me what day it is."},
]

# 2. 定义示例格式化模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# 3. 构建few-shot模板主体
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 构建系统提示词
system_message = """你是一个专业、友好且有用的AI助手。
请用中文直接回答问题，保持回答简洁明了。并在下面重复一遍英语翻译"""

# 4. 整合到完整提示流
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    # 如果启用Few-Shot，则添加示例
    # few_shot_prompt将在运行时根据配置添加
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

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
                
                # 转换历史消息为LangChain消息格式
                history_messages = []
                for msg in recent_messages:
                    if msg["role"] == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    else:
                        history_messages.append(AIMessage(content=msg["content"]))
                
                # 根据Few-Shot开关决定是否使用Few-Shot模板
                if use_few_shot:
                    # 构建带Few-Shot示例的完整提示
                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        few_shot_prompt,  # 添加Few-Shot示例
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                else:
                    # 使用基本的提示模板
                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_message),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{input}")
                    ])
                
                # 使用LangChain提示模板API
                chain = full_prompt | llm
                
                # 调用模型获取响应
                response = chain.invoke({
                    "history": history_messages,
                    "input": user_input
                })
                
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
                
                # 估算token使用情况
                # 估算提示词token
                system_chars = len(system_message)
                few_shot_chars = len(str(examples)) if use_few_shot else 0
                history_chars = sum(len(msg["content"]) for msg in recent_messages)
                input_chars = len(user_input)
                
                prompt_chars = system_chars + few_shot_chars + history_chars + input_chars
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