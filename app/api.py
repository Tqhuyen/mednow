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
import PIL.Image
import rag
import google.generativeai as genai



load_dotenv()

# Define API key and model

api_key_google = os.getenv("GOOGLE_API_KEY")

def chat(text=None,context=None, images=None):
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f'''
    You are a doctor please answer question base on this context and from image: {context},
    Question: {text}
    '''
    input = [prompt]
    # input.extend(images)
    for image_path in images:
        image = PIL.Image.open(image_path)
        input.append(image)
    response = model.generate_content(input)
    return response.parts[0].text
















