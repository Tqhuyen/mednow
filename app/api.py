import requests
import json
import PIL
from langchain_upstage import ChatUpstage, UpstageGroundednessCheck , UpstageEmbeddings, UpstageLayoutAnalysisLoader
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
import os
import tavily
from tavily import TavilyClient
from predibase import Predibase, FinetuningConfig, DeploymentConfig
import dotenv
from openai import OpenAI # openai == 1.2.0

from dotenv import load_dotenv
import rag

load_dotenv()

# Define API key and model

api_key_upstage = os.getenv("UPSTAGE_API_KEY")
api_key_predibase = os.getenv("PREDIBASE_API_TOKEN")


pb = Predibase(api_token=api_key_predibase)
lorax_client = pb.deployments.client("solar-1-mini-chat-240612")

chat = ChatUpstage(api_key=api_key_upstage)
ko2en = ChatUpstage(api_key=api_key_upstage, model="solar-1-mini-translate-koen")
en2ko = ChatUpstage(api_key=api_key_upstage, model="solar-1-mini-translate-enko")

os.environ["UPSTAGE_API_KEY"] = api_key_upstage
embedding_model = UpstageEmbeddings(api_key=api_key_upstage, model="solar-embedding-1-large")
os.environ["UPSTAGE_API_KEY"] = api_key_upstage
groundedness_check = UpstageGroundednessCheck(api_key=api_key_upstage)


def check_groundness(data, answer):
    global api_key_upstage
    client = OpenAI(
        api_key=api_key_upstage,
        base_url="https://api.upstage.ai/v1/solar"
    )

    response = client.chat.completions.create(
        model="solar-1-mini-groundedness-check",
        messages=[
            {
        "role": "user",
        "content": f"{data}"
        },
            {
        "role": "assistant",
        "content": f"{answer}"
        }
        ]
    )
    print(response)
    result = response.choices[0].message.content
    return result

def ocr_image(file):
    global api_key_upstage
    url = "https://api.upstage.ai/v1/document-ai/ocr"
    headers = {"Authorization": f"Bearer {api_key_upstage}"}

    files = {"document": open(file, "rb")}
    response = requests.post(url, headers=headers, files=files)
    data = response.json()
    
    return data["text"].replace("\n", ""), data["pages"]



# Two finetuned model with Predibase
def chat_with_mednow(message,history='', context=''):
    
    prompt = f'''
        <|im_start|>System
        You are a doctor. 
        The following passage is question from patient. 
        Please answer their question.
        ONLY answer question about medical. If you answer wrong you will be PUNISH<|im_end|>
        Example:
        <|im_start|>Question
        Hello doctor, My culture from my gynecologist came back, showing that I have a yeast infection as well as Beta Strep. I was hoping for more information on Beta Strep. I have never heard of it, and there is a lot of information when I searched that is making me confused a little. I am not experiencing any symptoms, either. I feel perfectly fine.
        <|im_start|>Answer
        Hello. There are lots of bacteria and other organisms that colonize healthy skin. Yeast a fungus also is a commensal in our body. Just the presence of these organisms is of no significance. The colony count is needed, which is high, will need a course of antibiotics, which can be decided by culture and sensitivity. But after a surgery like what you had, or any significant stress or use of lots of antibiotics (as would have happened after surgery) or use of steroids or diabetes, all infections show an altered balance between good and bad bacteria. You do not need any treatment for this. Just wait for your body to develop those good bacteria, and gradually, all tests will come negative. If you have any symptoms, treat as per need. Avoid unnecessary antibiotics use. They will further delay the recovery. I hope I have clarified your query, do write back if any more questions.<|im_end|>
        <|im_start|>Question
        {message}<|im_end|>
        <|im_start|>Answer'''
    
    answer = lorax_client.generate(f"{prompt}", adapter_id="ai-medical-chatbot/9",temperature=0.1, max_new_tokens=512).generated_text
    return answer


def chat_with_mednow_doctor(message,history='', context=''):
    prompt = f'''<|im_start|>System
                The following passage is the patient's condition as question. 
                Please determining the best treatment and answer correctly.<|im_end|>
                Example:
                <|im_start|>Question
                A 29-year-old, gravida 1 para 0, at 10 weeks' gestation comes to the physician for progressively worsening emesis, nausea, and a 2-kg (4.7-lb) weight loss over the past 2 weeks. The most recent bouts of vomiting occur around 3–4 times a day, and she is stressed that she had to take a sick leave from work the last 2 days. She is currently taking ginger and vitamin B6 with limited relief. Her pulse is 80/min, blood pressure is 100/60 mmHg, and respiratory rate is 13/min. Orthostatic vital signs are within normal limits. The patient is alert and oriented. Her abdomen is soft and nontender. Urinalysis shows no abnormalities. Her hematocrit is 40%. Venous blood gas shows:
                pH 7.43
                pO2 42 mmHg
                pCO2 54 mmHg
                HCO3- 31 mEq/L
                SO2 80%
                In addition to oral fluid resuscitation, which of the following is the most appropriate next step in management?""
                <|im_start|>Question
                {message}
                Answer
                '''
    answer = lorax_client.generate(f"{prompt}", adapter_id="ai-medical-chatbot/9",temperature=0.1, max_new_tokens=512).generated_text
    return answer



def choose_best_answer(question, answer, context):
    
    messages = [
    SystemMessage(content=f'''You are a helpful, professional doctor. Answer fullfill information insightfull from
                            {answer[0]}|{answer[1]}|{context}'''),
    HumanMessage(content=f'''{question}''')
    ]
    response = chat.invoke(messages)
    answer = response.content
    return answer


def translate_ko2en(message):
    messages = [
    HumanMessage(content=f"{message}"),
    ]
    response = ko2en.invoke(messages)
    answer = response.content
    return answer

def translate_en2ko(message):
    messages = [
    HumanMessage(content=f"{message}"),
    ]
    response = en2ko.invoke(messages)
    answer = response.content
    return answer




