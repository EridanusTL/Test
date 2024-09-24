import pandas as pd
import os
from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="sk-WLheelCpyFu48FQfGsjHi07lCjF4qJdO7qbGzG7dWILxFRzA",
    base_url="https://api.chatanywhere.tech/v1",
)


def GenerateInputMessage(name, score):
    messages = [
        {"role": "system", "content": "你是一个优秀的美术老师，擅长国画。"},
        {
            "role": "user",
            "content": "学生名叫"
            + name
            + "，上小学二年级，今天上了国画美术课，上课表现的分数是"
            + str(score[0])
            + "分，"
            + "请你根据现在的分数围绕学生的态度和专注力写三句评语；"
            + "上课画面的分数是"
            + str(score[1])
            + "分，"
            + "你根据现在的分数围绕学生的沟通能力、画画的用色水平和用墨水平写三句评语；"
            + "上课期望的分数是"
            + str(score[2])
            + "分，"
            + "你根据现在的分数围绕学生的沟通能力、画画的用色水平和用墨水平写三句评语；"
            + "老师的名字叫婷婷老师，评语是对着孩子家长说的，以第三人称的视角评价学生，"
            + "评语中不能有分数，评语的语气要可爱一点并配上颜文字与emoji，所有的评语最后必须一定要组成一段完整的话，不能有换行！。",
        },
        {
            "role": "assistant",
            "content": "分数的评价标准为：1分最低，5分最高。",
        },
    ]
    return messages


def GenerateHelloMsg(name):
    hello_msg = name[-2:] + "妈妈，晚上好吖[玫瑰]"
    return hello_msg


def gpt_4o_api(messages: list):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        presence_penalty=1,
        frequency_penalty=1,
        messages=messages,
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


if __name__ == "__main__":
    # get work path
    work_path = os.getcwd()

    # 替换为你的CSV文件路径
    file_path = "/dataset/students.csv"

    data = pd.read_csv(work_path + file_path)
    print(data)

    # Read generation switch
    all_generation = 1
    for index, row in data.iterrows():
        if row.iloc[4] == 1:
            all_generation = 0
            break

    numbers = []
    score = []
    output = []
    if all_generation:
        for index, row in data.iterrows():
            try:
                name = row.iloc[0]
                score.append(float(row.iloc[1]))  # 第二列
                score.append(float(row.iloc[2]))  # 第二列
                score.append(float(row.iloc[3]))  # 第二列
                numbers.append((score[0], score[1], score[2]))

            except ValueError:
                print(f"行 {index} 中数据格式不正确")
            except IndexError:
                print(f"行 {index} 中存在未打的分数")

            # Gpt output
            hello_msg = GenerateHelloMsg(name)
            message = GenerateInputMessage(name, score)
            output.append(hello_msg + gpt_4o_api(message))
    else:
        for index, row in data.iterrows():
            if row.iloc[4] == 0:
                continue
            try:
                name = row.iloc[0]
                score.append(float(row.iloc[1]))  # 第二列
                score.append(float(row.iloc[2]))  # 第二列
                score.append(float(row.iloc[3]))  # 第二列
                numbers.append((score[0], score[1], score[2]))

            except ValueError:
                print(f"行 {index} 中数据格式不正确")
            except IndexError:
                print(f"行 {index} 中存在未打的分数")

            # Gpt output
            hello_msg = GenerateHelloMsg(name)
            message = GenerateInputMessage(name, score)
            output.append(hello_msg + gpt_4o_api(message))

    with open(work_path + "/output/output.txt", "w") as file:
        if isinstance(output, list):
            for item in output:
                file.write(item + "\n\n")
        else:
            file.write(output)
