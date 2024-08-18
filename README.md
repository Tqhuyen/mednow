<h1> MedNow </h1>
<p>A Chatbot can help you with good medical advice.</p>
<p>Chatbot can help you:</p>
<li>Give you information about any medical information like medicine, disease, treatment,...</li>
<li>Upload your prescription and ask anything about it</li>
<li>Assist healthcare professionals in finding relevant research or clinical guidelines.</li>
<li>Educate patients on various health topics by pulling information from WHO's PDFs. </li>
<li>Check your symptom, so you do not too worry about your conditons or you can know you must go to hospital </li>
<h2>If you upload your Prescription. Please choose language of question the same with prescription and upload image (PNG,JPG...)</h2>

# Upstage API use
OCR: OCR extracts text from prescription -> feeds into chatbot \
Embedding: Converts text into vector format to feed into RAG, and from RAG retrieves context \
Translation: Translates from Korean -> English and vice versa English -> Korean \
Chat: used to give answers based on context and answers from 2 finetuned models. Finetune 2 models for deep data and general data, to give the best answer \
Grounded Check: Checks if the information is correct in context \
Layout Analysis: Analyzes PDF files to convert to text, then embeds into Vector DB 

# Python Version
Use Python version 3.11 is the best
# Install Library
```
cd app
pip install -r requirements.txt
```
# Setup varible code in .env
```
PREDIBASE_API_TOKEN="PREDIBASE_API_TOKEN"
UPSTAGE_API_KEY="UPSTAGE_API_KEY"
TAVILY_API_KEY="TAVILY_API_KEY"
MONGODB_ATLAS_CLUSTER_URI="MONGODB_ATLAS_CLUSTER_URI"
```
# Run code
Make sure you are in app/ folder
```
gradio gradio_demo.py
```
And then go to : http://127.0.0.1:7860 \
or you can go to: https://huggingface.co/spaces/tqhuyen/MedNow for test without rerun
# Expected outcome
![Alt text](https://github.com/Tqhuyen/mednow/blob/main/img_github/Screenshot%202024-08-18%20220223.png)

