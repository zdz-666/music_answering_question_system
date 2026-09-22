from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import Literal, List
from pydantic import BaseModel, Field

class Route(BaseModel):
    question_types: List[Literal["self_introduction", "music_analysis", "music_list"]] = Field(
        default_factory=list,
        description="问题类型列表，可包含：self_introduction（自我介绍）、music_analysis（音乐分析）、music_list（歌单列表）中的一个或多个"
    )

def get_router_collection(query):
    
    llm = ChatOpenAI(
    model="",
    api_key="",
    base_url="",
    temperature=0.1
)
    executor = llm.with_structured_output(Route)
    prompt = """
    你是一个智能路由模型，你的任务是根据用户的查询，判断应该将查询路由到哪个集合。
    集合包括：self_introduction、music_analysis、music_list。
    集合说明：
    self_introduction：自我介绍，包含个人喜欢的音乐。
    music_analysis：用户对一些音乐的私人分析。
    music_list：用户最近听过的一些歌单。
    用户查询：{question}
    请根据查询内容，判断应该将查询路由到哪个集合。
    输出格式：
    如果只返回一个集合，直接输出集合名称，如：self_introduction
    如果返回多个集合，用逗号分隔集合名称，如：self_introduction,music_analysis
    请不要包含其他任何解释性文本。
    """
    prompt_template = ChatPromptTemplate.from_template(prompt)
    messages = prompt_template.format_messages(question=query)
    result = executor.invoke(messages)

    return result.question_types

while True:
    query = input("用户问题：")
    if query == "exit":
        break
    collections = get_router_collection(query)
    print(f"\n路由集合：{collections}\n")
