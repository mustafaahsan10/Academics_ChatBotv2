# from pydantic_ai import Agent
# from pydantic_ai.models.openai import OpenAIModel

# model = OpenAIModel(
#     "anthropic/claude-3.5-sonnet",  # or any other OpenRouter model
#     base_url="https://openrouter.ai/api/v1",
#     api_key="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",
# )

# agent = Agent(model)
# result = agent.run("What is the meaning of life?")
# print(result)



from openai import OpenAI
from os import getenv

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  model="openai/gpt-4o-mini",
  
  # pass extra_body to access OpenRouter-only arguments.
  # extra_body={
    # "models": [
    #   "${Model.GPT_4_Omni}",
    #   "${Model.Mixtral_8x_22B_Instruct}"
    # ]
  # },
  messages=[
    {
      "role": "user",
      "content": "Say this is a test",
    },
  ],
)
print(completion.choices[0].message.content)
