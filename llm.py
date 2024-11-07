import pathlib
import textwrap
import google.generativeai as genai
import google.generativeai as genai

GOOGLE_API_KEY= '<<ENTER YOUR API KEY>>'
genai.configure(api_key=GOOGLE_API_KEY)

# 모델 초기화
def LLM(text):
  model = genai.GenerativeModel('gemini-pro')
  while True:
    user_input = text
    if user_input=="q":
      break
    # 프롬프트와 instruction 정의
    else:
      instruction = "Please extract the keywords for the type of clothes and the color accordingly. Answer should be json format. If you can't find any keywords then leave blank. Note that one piece and dress are tops"
      prompt = """
      [Examples]
      User: I want to change the top of the woman to a blue sweatshirt and the bottom to a black skirt.
      You: {"top":["blue","sweatshirt"],"bottom":["black","skirt"]}
      User: change to shirt.
      You: {"top": ["","shirt"],"bottom":["",""]}
      User: """

      # 전체 프롬프트 생성 (instruction 포함)
      full_prompt = f"{instruction}\n{prompt}{user_input}\n  You: "

      # 모델 호출
      response = model.generate_content(full_prompt)

      # 응답 출력
      return response
