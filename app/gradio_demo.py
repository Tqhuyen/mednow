import PIL.ImageDraw
import gradio as gr
import requests
import json
import PIL

import api
from api import chat
import rag

def get_anwser(history, question):
    history.append((question["text"], None))
    files = []
    for x in question['files']:
        image = (x,)
        history.append((image, None))
        files.append(x)
    context = rag.retrive_answer(message=question)
    
    ans = chat(text=question, context=context, images=files)
    

    history.append((None, ans))

    return history, gr.MultimodalTextbox(value=None, interactive=False)


def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)

def clear_history(history):
    history = None
    return history


def user_add_pdf_to_rag(filename=None):
    if filename != None:
        rag.add_pdf_data_to_rag(filename=filename)
    return None


with gr.Blocks(fill_height=False) as demo:
    with gr.Tab("MedNow"):
        gr.Markdown('''
        <h1> MedNow </h1>
        <p>A Chatbot can help you with good medical advice.</p>
        <p>Chatbot can help you:</p>
        <li>Give you information about any medical information like medicine, disease, treatment,...</li>
        <li>Upload your prescription and ask anything about it</li>
        <li>Assist healthcare professionals in finding relevant research or clinical guidelines.</li>
        <li>Educate patients on various health topics by pulling information from WHO's PDFs. </li>
        <li>Check your symptom, so you do not too worry about your conditons or you can know you must go to hospital </li>
        <h2>If you upload your Prescription. Please choose language of question the same with prescription and upload image (PNG,JPG...)</h2>''')
        chatbot = gr.Chatbot(
            label="MedNow",
            height=400,
            elem_id="chatbot",
            avatar_images=("https://cdn-icons-png.flaticon.com/512/3607/3607444.png","https://cdn.icon-icons.com/icons2/2122/PNG/512/doctor_medical_avatar_people_icon_131305.png"),
            bubble_full_width=True,
            scale=1,
            placeholder="Dialogue is here!"
        )
            
        chat_input = gr.MultimodalTextbox(interactive=True,
                                        label="Prompt",
                                        file_count="multiple",
                                        placeholder="Enter message or upload image file (png, jpef,etc)...", show_label=False)
        submit_button = gr.Button(value="Submit")
        clear_button = gr.ClearButton()
        chat_msg = chat_input.submit(get_anwser, [chatbot, chat_input], [chatbot, chat_input])
        chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        
        chatbot.like(print_like_dislike, None, None)
    
        example = gr.Examples(examples=[
        [{"text": "What is Peumonia?", "files":[]}],
        [{"text": "What is amoxicilin?", "files":[]}],
        [{"text": "페니실린은 무엇에 사용되나요?", "files": []}],
        [{"text": "What are these medicine and what are they used for?", "files": ["flagged/Prescription/9a0a5c21fb2a89215d2a/phpv2jvad.png"]}],
        [{"text": "이 처방전에는 어떤 약이 들어 있으며 그 용도는 무엇입니까?", "files": ["flagged/Prescription/70eec823ed8e254d796d/korean_sample.png"]}],
        [{"text": "What is anemia?", "files":[]}],
        [{"text": "빈혈이란?", "files": []}],
        
    ],
        inputs=[chat_input])
        
        chat_msg =  submit_button.click(get_anwser, [chatbot, chat_input], [chatbot, chat_input])
        chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        clear_button.click(fn=clear_history, inputs=chatbot, outputs=chatbot)
    with gr.Tab("Add data"): 
        gr.Markdown('''
        <h1> Upload data you believe can help other people </h1>
        <p>This tab is only used for add document about medical to expand knowledge of our bot.</p>
        <p>If you want to upload your prescription, please go to "MedNow" tab</p>''')
        files = gr.File()
        button = gr.Button(value="Upload")
        file_uploaded = button.click(user_add_pdf_to_rag, inputs=[files], outputs=[files])
        
demo.launch(debug=True)