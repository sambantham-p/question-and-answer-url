from langchain.chains import LLMChain
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
import fitz

def load_pdf_text(file_path):
    pdf_document = fitz.open(file_path)
    extracted_text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        extracted_text += page.get_text()
    pdf_document.close()
    cleaned_text = extracted_text.replace('\n', ' ').strip()
    cleaned_text = ' '.join(word for word in cleaned_text.split() if not word.startswith(('http://', 'https://')))
    return cleaned_text

def qaretriever(question,urls):
  modelPath = "all-MiniLM-L6-v2"
  model_kwargs = {'device':'cpu'}
  load_dotenv()


  tokenizer = AutoTokenizer.from_pretrained("flan-t5-large")

  model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-large")

  pipe = pipeline("text2text-generation", model=model,tokenizer=tokenizer,max_length=512)
#   pipe = pipeline(
#     "question-answering",
#     model=model,
#     tokenizer=tokenizer,
#     return_tensors='pt'
# )


  llm = HuggingFacePipeline(
    pipeline = pipe,
    model_kwargs={"temperature": 0.9,"max_length":512},
  )

  youtube =  urls[0].startswith('https://www.youtube.com')
  pdfFile = urls[0].endswith('.pdf')
  print("PDF FILE IS:",pdfFile)
  if youtube:
    loaders = YoutubeLoader.from_youtube_url(urls[0],add_video_info=True)
    data = loaders.load()
  if pdfFile:
    pdfLoader = PyPDFLoader(urls[0])
    data = pdfLoader.load()
  else:
    loaders = UnstructuredURLLoader(urls=urls)
    data = loaders.load()

  embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)

  print("DATA IS:",data)
  main_loader = st.empty()
  main_loader.text('Data Loading... ')
  text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n','\n',',','.',' '],chunk_size=1000,chunk_overlap=150)
  print("Text splitter is:",text_splitter)
  main_loader.text('Recursive text splitter started Loading... ')
  docs = text_splitter.split_documents(data)
  print('docs',docs)
  template = """Use the following pieces of context to answer the question at the end. You can modify answer but it should be relevant and give full described answer. Context: {context} Question: {question} Helpful Answer:"""
  QA_CHAIN = PromptTemplate.from_template(template)
  main_loader.text('...Embeddings started ... ')
  if youtube:
    vector_index_y = FAISS.from_documents(docs,embeddings)
    vector_index_y.save_local("yt_index")
    db_y = FAISS.load_local("yt_index", embeddings)

    retriever_y = db_y.as_retriever()
    if question:
      chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_y,
            chain_type_kwargs={"prompt": QA_CHAIN}
        )
      output = chain({"query": question})
    return output

  else:
    vector_index = FAISS.from_documents(docs, embeddings)
    vector_index.save_local("faiss_index")
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever()
    st.session_state.vector_index = retriever

          # result = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
          # print("RES:::",result)
          # context = "Context information related to the question"

          # result = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_index.as_retriever())
          # output = result({"question":question},return_only_outputs=True)
    if question:
      chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN}
        )
      output = chain({"query": question})
    return output