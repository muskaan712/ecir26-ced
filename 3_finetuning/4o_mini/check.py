from openai import OpenAI
c = OpenAI()
models = c.models.list()
for model in models:
    print(model.id)

