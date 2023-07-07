import openai

openai.api_key = "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"


def askchatGPT(question):
    prompt = question
    model_engine = "text-davinci-003"
    completions = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024, n=1,
        stop=None,
        temperature=0.5,
    )
    message = completions.choices[0].text
    print(message)


if __name__ == '__main__':
    print("系统已加载")
    while True:
        print("=========================================")
        askchatGPT(input("请输入问题，chatGPT可以查询！"))
