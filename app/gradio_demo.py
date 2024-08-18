import PIL.ImageDraw
import gradio as gr
import requests
import json
import PIL

import api
from api import chat_with_mednow, chat_with_mednow_doctor, ocr_image, translate_en2ko, translate_ko2en, choose_best_answer, check_groundness
import rag

def get_anwser(language_prompt, history, question):
    history_scheme_user = '''Question {question}\n'''
    history_scheme_system = '''Answer {answer}\n'''
    chat_history = ''
    
    for user, system in history:
        if user != None:
            if language_prompt == "Korean":
                user = translate_ko2en(message=user)
            tmp_history = history_scheme_user.format(question=user)
            chat_history += tmp_history
        if system != None:
            if language_prompt == "Korean":
                system = translate_ko2en(message=system)
            tmp_history = history_scheme_system.format(answer=system)
            chat_history += tmp_history
    message = None
    files = []
    if question["text"] is not None:
        history.append((question['text'], None))
        message = question['text']
    for x in question['files']:
        image = (x,)
        history.append((image, None))
        files.append(x)
        
    if language_prompt == None:
        history.append((None, '''You need to choose your "Language" first'''))
        return history, gr.MultimodalTextbox(value=None, interactive=False)
    
    ans = None
    if language_prompt == "Korean":
        message = translate_en2ko(message=message)
        
    for image_path in files:
        if image_path != None:
            text_from_image, pages = ocr_image(image_path)
            image = PIL.Image.open(image_path)
            draw = PIL.ImageDraw.Draw(image)
            for page in pages:
                for word in page["words"]:
                    bounding_box = word["boundingBox"]
                    shape = []
                    for point in bounding_box["vertices"]:
                        temp = (point["x"], point["y"])
                        shape.append(temp)
                    draw.polygon(shape, outline ="blue")
            image.save(f'{image_path}_ocr.png')
            res = f'{image_path}_ocr.png'
            
            if language_prompt == "Korean":
                message = message + "\n 이것은 환자의 처방전입니다: \n" + text_from_image
            else:
                message = message + "\n This is patient's prescription: \n" + text_from_image
    

    ans_list = []
    if language_prompt == "Korean":
        message = translate_ko2en(message=message)
        
    ans = chat_with_mednow(message=message)
    ans_list.append(ans)
    ans = chat_with_mednow_doctor(message=message)
    ans_list.append(ans)
    
    
    context = rag.retrive_answer(message=message)
    ans = choose_best_answer(question=message, answer=ans_list, context=context)
    
    
    data_final = ans[0]+ans[1]+context
    check_truth = check_groundness(data=data_final, answer=ans)
    

         
    if len(files) != 0:
        history.append((None, (res,)))
    if language_prompt == "Korean":
        ans = translate_en2ko(message=ans)
    history.append((None, ans))
    
    if language_prompt == "Korean":
        if check_truth == "grounded":
            grounded_message = "이 정보는 우리 데이터베이스를 기반으로 작성되었습니다."
        else:
            grounded_message = "이 정보는 저희 데이터베이스를 기반으로 근거가 없습니다. 다시 한 번 확인하거나 '데이터 추가' 탭을 사용하여 근거를 만드는 데 도움을 줄 수 있습니다."
    else:
        if check_truth == "grounded":
            grounded_message ="This information is Grounded base on our database"
        else:
            grounded_message ="This information is NOT GROUNDED base on our database. You should double check it, or you can help us grounded this using 'Add data' tab"

    
    if language_prompt=="Korean":
        history.append((None, f"이건 단지 의료 정보일 뿐이에요, 만약 정말 문제가 있다면요. 가장 가까운 보건소로 가세요! \n {grounded_message}"))
    else:
        history.append((None, f"This is only medical information, if you really have any problem. Please go to nearest Health Center! \n {grounded_message}"))
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
        with gr.Row():
            # model_type= gr.Radio(["Normal Person", "Patient", "Doctor"], label="Use case", info="Who are you?")
            language = gr.Radio(["English", "Korean"], label="Language", info="Which language you are using?")
            
        chat_input = gr.MultimodalTextbox(interactive=True,
                                        label="Prompt",
                                        file_count="multiple",
                                        placeholder="Enter message or upload image file (png, jpef,etc)...", show_label=False)
        submit_button = gr.Button(value="Submit")
        clear_button = gr.ClearButton()
        chat_msg = chat_input.submit(get_anwser, [language ,chatbot, chat_input], [chatbot, chat_input])
        chat_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
        
        chatbot.like(print_like_dislike, None, None)
    
        example = gr.Examples(examples=[
        ["English",{"text": "What is Peumonia?", "files":[]}],
        ["English",{"text": "What is amoxicilin?", "files":[]}],
        ["Korean", {"text": "페니실린은 무엇에 사용되나요?", "files": []}],
        ["English",
         {"text": "What are these medicine and what are they used for?", "files": ["flagged/Prescription/9a0a5c21fb2a89215d2a/phpv2jvad.png"]}
         ],
        ["Korean", {"text": "이 처방전에는 어떤 약이 들어 있으며 그 용도는 무엇입니까?", "files": ["flagged/Prescription/70eec823ed8e254d796d/korean_sample.png"]}],
        ["English",{"text": "What is anemia?", "files":[]}],
        ["Korean", {"text": "빈혈이란?", "files": []}],
        
    ],
        inputs=[language,chat_input])
        
        chat_msg=  submit_button.click(get_anwser, [language ,chatbot, chat_input], [chatbot, chat_input])
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