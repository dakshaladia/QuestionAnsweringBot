from flask import Flask, request, jsonify, send_file
import json
import os
import getpass
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import JSONLoader

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_or_json' not in request.files or 'json' not in request.files:
        return 'No file part', 400

    inputDoc = request.files['pdf_or_json']
    question_json = request.files['json']

    contextFile = saveInputsInTemp(inputDoc)
    questionJsonFile = saveInputsInTemp(question_json)

    if inputDoc.filename == '' or question_json.filename == '':
        return 'No selected file', 400

    processed_data = []
    type = ""

    if contextFile.endswith('.pdf'):        
        type = "pdf"
        processed_data = process_pdf(contextFile)
    elif contextFile.endswith('.json'):
        type = "json"
        processed_data = process_json(contextFile)
    else:
        return 'Unsupported file type', 400

    output_json = generateOutput(processed_data, questionJsonFile, type)

    return send_file(output_json, as_attachment=True)

def saveInputsInTemp(filename_Full):
        
        filename = filename_Full.filename
        save_path = os.path.join('tmp/', filename)
        filename_Full.save(save_path)

        return save_path

def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    data = loader.load_and_split()  

    return data

def process_json(json_file):
    loader = JSONLoader(
    file_path= json_file,
    jq_schema='.[]',
    text_content=False)
    data = loader.load()

    return data

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generateOutput(processed_data, question_json, type):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(processed_data)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    f = open(question_json)
    data = json.load(f)
    questions= data['Question'].split(';')

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    qa_pairs = {}
    for question in questions:
        chunks = []
        for chunk in rag_chain.stream(question):
            chunks.append(chunk)

        qa_pairs[question] = ' '.join(chunks)

    filename = 'qa_pairs_pdf.json' if type == "pdf" else 'qa_pairs_json.json'

    with open(filename, 'w') as file:
        json.dump(qa_pairs, file, indent=4)

    return filename

if __name__ == '__main__':
    app.run(debug=True)



