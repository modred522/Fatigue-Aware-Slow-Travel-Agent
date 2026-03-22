import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化通义千问大模型
llm = ChatTongyi(model="qwen-turbo")

# 发送一条简单的测试消息
response = llm.invoke("你好，请用一句话介绍一下什么是慢旅行（Slow Travel）？")

print("通义千问的回复：")
print(response.content)