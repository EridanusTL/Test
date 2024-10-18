from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-WLheelCpyFu48FQfGsjHi07lCjF4qJdO7qbGzG7dWILxFRzA",
    base_url="https://api.chatanywhere.tech/v1",
)


# 非流式响应
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def gpt_4o_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.8,
        presence_penalty=1,
        max_tokens=150,
        frequency_penalty=1,
        messages=messages,
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")


if __name__ == "__main__":
    score_1 = 4
    score_2 = 3
    score_3 = 2
    messages = [
        {"role": "system", "content": "你是一个优秀的美术老师，擅长国画。"},
        {
            "role": "user",
            "content": "学生名叫余兴华，上小学二年级，今天上了国画美术课，上课表现的分数是"
            + str(score_1)
            + "分，"
            + "请你根据现在的分数围绕学生的态度和专注力写三句评语；"
            + "上课画面的分数是"
            + str(score_2)
            + "分，"
            + "你根据现在的分数围绕学生的沟通能力、画画的用色水平和用墨水平写三句评语；"
            + "上课期望的分数是"
            + str(score_3)
            + "分，"
            + "你根据现在的分数围绕学生的沟通能力、画画的用色水平和用墨水平写三句评语；"
            + "老师的名字叫婷婷老师，评语是对着孩子家长说的，以第三人称的视角评价学生，"
            + "评语中不能有分数，评语的语气要可爱一点并配上颜文字与emoji，所有的评语最后一定要组成一段完整的话，不能有换行！。",
        },
        {
            "role": "assistant",
            "content": "分数的评价标准为：1分最低，5分最高。",
        },
    ]
    # 非流式调用
    data = gpt_4o_api(messages)
    print("data:\n", data)
    # 流式调用
    # gpt_35_api_stream(messages)
