#from langchain.document_loaders.generic import GenericLoader
#from langchain.document_loaders.parsers import OpenAIWhisperParser
#from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import YoutubeLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
import os
import time
import shutil


def process_and_transcribe(url, openai_api_key, model):
    """Process and transcribe the video at a given url"""
    # Setting qa_chain as a global variable
    global qa_chain
    #save_dir = some_title
    #loader = GenericLoader(YoutubeAudioLoader(url, save_dir), OpenAIWhisperParser(api_key = openai_api_key))
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    vectordb = FAISS.from_texts(splits, embeddings)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name=model, temperature=0, openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )

    return text

def response(message, history):
    return qa_chain.run(message)

transcribe_interface = gr.Interface(
    fn=process_and_transcribe, # Triggers the transcibe function
    inputs=['text', 'text', gr.components.Radio(['gpt-3.5-turbo', 'gpt-3.5-turbo-16k'])], # Takes in the URL, OpenAI API Key and Model
    outputs=['text'], # Returns the Transcribed Text
    title="Summarize",
    description="Let's Summarize a Video, Paste the link of the YouTube Video, Give a Title to the Video and Enter your OpenAI API Key For Chatting"
)

chat_interface = gr.ChatInterface(fn=response, title="Chat with the AI about the topic", description="Chat with the AI about the video you just summarized.")
demo = gr.TabbedInterface([transcribe_interface, chat_interface], ["Summarize", "Chat"]) # Creates a Tabbed Interface

demo.queue()
demo.launch()