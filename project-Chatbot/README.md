# 双模型聊天机器人

基于LangChain和Streamlit构建的聊天机器人应用，支持DeepSeek和通义千问双模型切换。

## 功能特点

1. **双模型切换**：支持在DeepSeek和通义千问模型间切换
2. **对话记忆**：使用ConversationSummaryBufferMemory保持对话上下文
3. **Token统计**：实时显示Token消耗情况
4. **简洁界面**：基于Streamlit构建的简单易用界面

## 安装步骤

1. 克隆项目到本地
2. **环境设置**:
   - 如果使用conda（推荐）:
   ```bash
   # 创建新的conda环境
   conda create -n chatbot python=3.10
   # 激活环境
   conda activate chatbot
   ```
3. 安装依赖：
   ```bash
   # 确保已激活正确的环境
   pip install -r requirements.txt
   ```
   - 如果遇到`Unknown option: r`或`Unknown or unsupported command 'install'`错误:
     - 打开Anaconda Prompt而不是普通命令提示符
     - 尝试使用`python -m pip install -r requirements.txt`命令

4. 配置API密钥：
   - 复制`.env.example`为`.env`
   - 在`.env`文件中填入你的API密钥：
```
DEEPSEEK_API_KEY=你的DeepSeek API密钥
DASHSCOPE_API_KEY=你的通义千问API密钥
```
   - 也可以直接设置系统环境变量

## 常见问题解决

### 导入错误
如果遇到`ImportError: cannot import name 'ChatDeepSeek' from 'langchain_community.chat_models'`错误:

1. 更新相关包:
```bash
pip install --upgrade langchain-community
pip install --upgrade langchain-openai
```

2. 已经在`models.py`中提供了备选导入方案，如果DeepSeek不可用，将自动使用OpenAI模型作为替代

### 运行错误
如果遇到其他运行错误，请检查:
1. 环境变量是否正确设置
2. 是否使用了正确版本的Python (推荐3.9或3.10)
3. 所有依赖是否安装完成

## 运行应用

```bash
# 确保已激活正确的环境
streamlit run app.py
```

## 使用方法

1. 在侧边栏选择要使用的模型
2. 在输入框中输入消息
3. 查看模型回复
4. 侧边栏显示Token使用统计
5. 需要时可以清除会话历史

## 项目结构

```
project-Chatbot/
├── .env.example       # 环境变量模板
├── .env               # 实际环境变量文件（需自行创建）
├── app.py             # 主应用文件
├── requirements.txt   # 依赖列表
└── src/
    └── models.py      # 模型封装模块
```

## 学习重点

1. LangChain的模型封装与API调用
2. 对话记忆机制的实现
3. 提示模板的设计与使用
4. Token使用量的统计
5. Streamlit界面的构建 