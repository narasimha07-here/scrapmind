__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from typing import List, Dict, Any, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader, UnstructuredMarkdownLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.retrievers.base import BaseRetriever
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class ChromaRAGSystem:
    def __init__(self, openrouter_api_key: str = None, openai_api_key: str = None):
        self.openrouter_api_key = openrouter_api_key
        self.openai_api_key = openai_api_key
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
        if openai_api_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_base="https://openrouter.ai/api/v1",
                openai_api_key=openrouter_api_key,
                model="openai/text-embedding-ada-002"
            )
    
    def setup_vectorstore(self, bot_id: str, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = f"data/vectorstores/{bot_id}"
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name=f"bot_{bot_id}_docs"
        )
        
        return self.vectorstore
    
    def setup_retriever(self, search_type: str = "similarity", search_kwargs: Dict = None):
        if self.vectorstore is None:
            raise ValueError("Vectorstore must be setup first")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        return self.retriever
    
    def similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.0):
        if self.vectorstore is None:
            return []
        
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query=query, 
            k=k
        )
        
        filtered_docs = [
            (doc, score) for doc, score in docs_and_scores 
            if score >= score_threshold
        ]
        
        return filtered_docs
    
    def add_documents(self, documents: List[Document]):
        if self.vectorstore is None:
            raise ValueError("Vectorstore must be setup first")
        
        self.vectorstore.add_documents(documents)
        
        self.vectorstore.persist()
    
    def process_uploaded_files(self, uploaded_files: List, chunk_size: int = 1000, 
                              chunk_overlap: int = 200, chunking_strategy: str = "recursive_character"):
        documents = []
        
        for uploaded_file in uploaded_files:
            temp_path = f"data/temp/{uploaded_file.name}"
            os.makedirs("data/temp", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(temp_path)
            elif uploaded_file.name.endswith('.csv'):
                loader = CSVLoader(temp_path)
            elif uploaded_file.name.endswith('.md'):
                loader = UnstructuredMarkdownLoader(temp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
                doc.metadata['file_type'] = uploaded_file.type
            
            documents.extend(docs)
            
            os.remove(temp_path)
        
        if chunking_strategy == "recursive_character":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        split_docs = text_splitter.split_documents(documents)
        
        self.add_documents(split_docs)
        
        return len(split_docs)
    
    def setup_qa_chain(self, llm, chain_type: str = "stuff", return_source_documents: bool = True):
        if self.retriever is None:
            raise ValueError("Retriever must be setup first")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=return_source_documents,
            verbose=False
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict:
        if self.qa_chain is None:
            raise ValueError("QA chain must be setup first")
        
        try:
            result = self.qa_chain({"query": question})
            
            return {
                "success": True,
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "answer": "Sorry, I encountered an error while searching the documents.",
                "sources": []
            }
