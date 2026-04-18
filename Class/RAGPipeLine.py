import atexit
import multiprocessing
import re

import torch
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from Common.config import LLM_Model

os.environ['HF_HOME'] = './models'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''
cache_dir = os.getenv("HF_HOME")
HF_KEY = "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"
os.environ["HF_TOKEN"] = HF_KEY

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

load_dotenv()

SYSTEM_MESSAGE = "You are a helpful Islamic scholar. Answer ONLY based on context."
#INSTRUCTION_MESSAGE = "Provide a concise answer in 1-3 complete sentences."
INSTRUCTION_MESSAGE = "Provide a thorough and detailed answer, covering all relevant aspects based on the context."

class RAGPipeLine():
    def __init__(self, device, doc_name, log_fn=None):
        self.config = LLM_Model
        self.device = device
        self.doc_name = doc_name
        print(self.config)

        self._load_embeddings()
        self._load_llm()

        self.vector_stores = None
        self.rag_chain = None
        atexit.register(self.cleanup)

    def cleanup(self):
        try:
            multiprocessing.active_children()
        except Exception as rx:
            print(rx)

    def log(self, msg):
        print(msg)

    def _load_embeddings(self):
        self.log("Loading embedding model: BAAI/bge-large-en-v1.5...")
        self.embedding = HuggingFaceEmbeddings(model_name=self.config["embedding_model"],
                                               model_kwargs={"device": "cpu"}, multi_process=False,
                                               encode_kwargs={"normalize_embeddings": True})
        self.log("Embedding model ready...")

    def _load_llm(self):
        self.log(f"Loading LLM: {self.config['llm']}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["llm"])

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config["llm"],
            quantization_config=bnb_config,
            device_map={"":0}
        )

        model.generation_config.max_new_tokens = self.config["max_new_tokens"]
        model.generation_config.do_sample = False
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        model.generation_config.eos_token_id = self.tokenizer.eos_token_id

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=self.tokenizer,
            return_full_text=False
            )

        #self.llm = HuggingFacePipeline(pipeline=pipe)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.log("LLM ready...")

        '''
        pipe = pipeline(
            "text-generation",
            model=self.config["llm"],  # ← only this changed
            token=os.getenv("HF_TOKEN"),
            max_new_tokens=self.config["max_new_tokens"],
            pad_token_id=2,
            eos_token_id = 2,
            do_sample=False,  # deterministic
            return_full_text=False,  # return only the answer, not the prompt
            device=-1
            # model_kwargs={
            #     "quantization_config": bnb_config
            # }
        )
        '''

    def save_vector_store(self, chunks, path):
        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embedding)
        vector_store.save_local(path)
        self.log("Vector saved")
        return vector_store

    def load_vector_store(self, chunks,  path):
        self.log("Loading vector store...")
        if os.path.exists(path):
            self.log("Vector exist. Loading...")
            return FAISS.load_local(path, self.embedding, allow_dangerous_deserialization=True)
        else:
            self.log("Vector Does not exist. Creating Vectors...")
            return self.save_vector_store(chunks, path)

    def load_pdf(self, pdf_path):
        self.log("Loading pdf")
        loader = PyMuPDFLoader(pdf_path)
        documents = list(loader.lazy_load())[11:]

        formatted_document = []
        current_content=[]
        current_header = ""
        for i, doc in enumerate(documents):
            lines = list(doc.page_content.split("\n"))[1:-1]
            line = 0
            while line < len(lines):
                m = re.search(r'(\d+)\.\s([A-Z]+(?:\s[A-Z]+)*)$', lines[line].strip())
                if m:
                    if current_header and current_content:
                        content_doc = Document(page_content="\n".join(current_content),
                                               metadata={'section': current_header,
                                                         'page': doc.metadata.get('page', 0)})
                        formatted_document.append(content_doc)
                    current_header = m.group(0) + " " + lines[line+1].strip()
                    line += 2
                    current_content.clear()
                else:
                    if lines[line].strip():
                        current_content.append(lines[line].strip())
                    line += 1
        if current_content and current_header:
            content_doc = Document(
                page_content="\n".join(current_content),
                metadata={'section': current_header}
            )
            formatted_document.append(content_doc)

        # Split and chunk them
        chunk_size = self.config['chunk_size']
        chunk_overlap = self.config['chunk_overlap']
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  separators=["\n\n", "\n", ".", " "])
        chunks = splitter.split_documents(formatted_document)
        #print(f"Chunk Length: {len(chunks)}")

        # Create or load vector store
        self.vector_stores = self.load_vector_store(chunks, f"faiss_index_{self.doc_name}_{self.config['max_new_tokens']}")
        self.log(type(self.vector_stores))
        self.log("Vector store ready...")

        #Building Rag Chain
        self.log("Building Rag Chain")
        self.build_rag_chain()

    def token_count(self, user_input):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful Islamic scholar. Answer ONLY based on context."),
            ("human", "Context:\n{context}\n\nQuestion: {input}\n\n" + INSTRUCTION_MESSAGE)
        ])

        docs = user_input['context']

        context_text = "\n".join(doc.page_content for doc in docs)

        context_token = self.tokenizer.encode(context_text)
        question_token = self.tokenizer.encode(user_input['input'])
        instruction_prompt_token = self.tokenizer.encode(INSTRUCTION_MESSAGE)

        total_token = len(context_token) + len(question_token) + len(instruction_prompt_token)
        print(f"total_token: {total_token}")

        return prompt

    def build_rag_chain(self):
        retriever = self.vector_stores.as_retriever(search_type="similarity",
                                                search_kwargs={"k":self.config["top_k"]})

        retriever = RunnableParallel({"context":retriever, "input":RunnablePassthrough()})
        self.rag_chain = (retriever
                          | RunnableParallel(answer = RunnableLambda(self.token_count)
                          | self.llm, source_docs = RunnableLambda(lambda x: x['context'])))

    def query(self, user_input):
        if self.rag_chain is None:
            self.log("Error: RAG chain not built. Call load_pdf() first.")
            return None
        self.log(f"Processing: {user_input}")

        result = self.rag_chain.invoke(user_input)
        answer = result["answer"].strip().replace("Answer: ", "")
        source_docs = result["source_docs"]

        answer += "\nSources: \n"

        for i, doc in enumerate(source_docs):
            section = doc.metadata.get('section', 'No Section')
            page = str(doc.metadata.get('page', 'No Page'))
            answer += f"  [{i + 1}] Section: {section} | Page: {page}\n"

        return answer