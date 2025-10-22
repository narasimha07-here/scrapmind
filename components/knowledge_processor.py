__import__('pysqlite3')

import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

import tempfile

import streamlit as st

import uuid

from datetime import datetime

from typing import Dict, List, Optional, Any, TYPE_CHECKING

import logging

import shutil

import re

import hashlib

import json

import pickle

from langchain_core.runnables import RunnableLambda

import time

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

try:

    from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader

    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

    from langchain.schema import Document

    from langchain_core.runnables import RunnablePassthrough, RunnableParallel

    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

    from langchain_core.output_parsers import StrOutputParser

    from langchain_core.runnables.base import Runnable

    try:

        from langchain_experimental.text_splitter import SemanticChunker

        SEMANTIC_AVAILABLE = True

    except ImportError:

        SEMANTIC_AVAILABLE = False

    CHROMA_AVAILABLE = False

    CHROMA_IMPORT = None

    try:

        from langchain_chroma import Chroma

        CHROMA_IMPORT = "langchain_chroma"

        CHROMA_AVAILABLE = True

    except ImportError:

        try:

            from langchain_community.vectorstores import Chroma

            CHROMA_IMPORT = "langchain_community"

            CHROMA_AVAILABLE = True

        except ImportError:

            try:

                from langchain.vectorstores import Chroma

                CHROMA_IMPORT = "langchain"

                CHROMA_AVAILABLE = True

            except ImportError:

                CHROMA_AVAILABLE = False

    EMBEDDINGS_AVAILABLE = False

    try:

        from langchain_community.embeddings import HuggingFaceEmbeddings

        EMBEDDINGS_AVAILABLE = True

    except ImportError:

        try:

            from langchain.embeddings import HuggingFaceEmbeddings

            EMBEDDINGS_AVAILABLE = True

        except ImportError:

            EMBEDDINGS_AVAILABLE = False

    LANGCHAIN_AVAILABLE = CHROMA_AVAILABLE and EMBEDDINGS_AVAILABLE

except ImportError as e:

    LANGCHAIN_AVAILABLE = False

    CHROMA_AVAILABLE = False

    EMBEDDINGS_AVAILABLE = False

    Document = None  # Set Document to None if import fails


class KnowledgeProcessor:

    def __init__(self, data_manager=None):

        self.embeddings = None

        self.vectorstore = None

        self.documents_processed = 0

        self._persist_directory = None

        self._collection_name = None

        self._processed_files = set()

        self.data_manager = data_manager or self._get_data_manager()

        self.user_id = st.session_state.get('user_id')

        self._initialize_storage_paths()

    def _get_data_manager(self):

        """Get DataManager instance from session state or create new one"""

        try:

            if 'data_manager' in st.session_state and st.session_state.data_manager:

                return st.session_state.data_manager

            else:

                from components.data_manager import DataManager

                dm = DataManager()

                st.session_state.data_manager = dm

                return dm

        except Exception as e:

            logger.warning(f"Could not get DataManager: {e}")

            return None

    def _initialize_storage_paths(self):

        """Initialize storage paths for knowledge processor data"""

        if not self.data_manager:

            return

        try:

            base_dir = self.data_manager.data_dir

            self.kb_data_dir = os.path.join(base_dir, "knowledge_base")

            self.vectordb_dir = os.path.join(base_dir, "vectordb")

            self.processed_files_dir = os.path.join(base_dir, "processed_files")

            for directory in [self.kb_data_dir, self.vectordb_dir, self.processed_files_dir]:

                os.makedirs(directory, exist_ok=True)

        except Exception as e:

            logger.error(f"Error initializing storage paths: {e}")

    def save_processor_state(self, bot_id: str):

        if not self.data_manager or not self.user_id:

            return False

        try:

            state_file = os.path.join(self.kb_data_dir, f"{self.user_id}_{bot_id}_state.json")

            processor_state = {

                'user_id': self.user_id,

                'bot_id': bot_id,

                'documents_processed': self.documents_processed,

                'persist_directory': self._persist_directory,

                'collection_name': self._collection_name,

                'processed_files': list(self._processed_files),

                'embedding_model': getattr(self.embeddings, 'model_name', None) if self.embeddings else None,

                'last_updated': datetime.now().isoformat(),

                'version': '1.0'

            }

            with open(state_file, 'w', encoding='utf-8') as f:

                json.dump(processor_state, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved processor state for bot {bot_id}")

            return True

        except Exception as e:

            logger.error(f"Error saving processor state: {e}")

            return False

    def load_processor_state(self, bot_id: str) -> bool:

        if not self.data_manager or not self.user_id:

            return False

        try:

            state_file = os.path.join(self.kb_data_dir, f"{self.user_id}_{bot_id}_state.json")

            if not os.path.exists(state_file):

                logger.info(f"No saved state found for bot {bot_id}")

                return False

            with open(state_file, 'r', encoding='utf-8') as f:

                processor_state = json.load(f)

            self.documents_processed = processor_state.get('documents_processed', 0)

            self._persist_directory = processor_state.get('persist_directory')

            self._collection_name = processor_state.get('collection_name')

            self._processed_files = set(processor_state.get('processed_files', []))

            embedding_model = processor_state.get('embedding_model')

            if embedding_model and not self.embeddings:

                self.initialize_embeddings(embedding_model)

            logger.info(f"Loaded processor state for bot {bot_id}")

            st.info(f"Restored knowledge base state: {self.documents_processed} processed documents.")

            return True

        except Exception as e:

            logger.error(f"Error loading processor state: {e}")

            return False

    def save_vectorstore_metadata(self, bot_id: str, metadata: Dict):

        if not self.data_manager or not self.user_id:

            return False

        try:

            metadata_file = os.path.join(self.vectordb_dir, f"{self.user_id}_{bot_id}_metadata.json")

            metadata_to_save = {

                **metadata,

                'user_id': self.user_id,

                'bot_id': bot_id,

                'saved_at': datetime.now().isoformat()

            }

            with open(metadata_file, 'w', encoding='utf-8') as f:

                json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:

            logger.error(f"Error saving vectorstore metadata: {e}")

            return False

    def load_vectorstore_metadata(self, bot_id: str) -> Optional[Dict]:

        if not self.data_manager or not self.user_id:

            return None

        try:

            metadata_file = os.path.join(self.vectordb_dir, f"{self.user_id}_{bot_id}_metadata.json")

            if os.path.exists(metadata_file):

                with open(metadata_file, 'r', encoding='utf-8') as f:

                    return json.load(f)

            return None

        except Exception as e:

            logger.error(f"Error loading vectorstore metadata: {e}")

            return None

    def save_processed_documents(self, bot_id: str, documents: List['Document']):

        """Save processed documents - using string annotation for type hint"""

        if not self.data_manager or not self.user_id:

            return False

        try:

            docs_file = os.path.join(self.processed_files_dir, f"{self.user_id}_{bot_id}_documents.pkl")

            serializable_docs = []

            for doc in documents:

                doc_data = {

                    'page_content': doc.page_content,

                    'metadata': doc.metadata

                }

                serializable_docs.append(doc_data)

            with open(docs_file, 'wb') as f:

                pickle.dump(serializable_docs, f)

            logger.info(f"Saved {len(documents)} processed documents for bot {bot_id}")

            return True

        except Exception as e:

            logger.error(f"Error saving processed documents: {e}")

            return False

    def load_processed_documents(self, bot_id: str) -> List['Document']:

        """Load processed documents - using string annotation for type hint"""

        if not self.data_manager or not self.user_id:

            return []

        try:

            docs_file = os.path.join(self.processed_files_dir, f"{self.user_id}_{bot_id}_documents.pkl")

            if not os.path.exists(docs_file):

                return []

            with open(docs_file, 'rb') as f:

                serializable_docs = pickle.load(f)

            documents = []

            for doc_data in serializable_docs:

                if Document is None:

                    logger.error("Document class not available")

                    continue

                doc = Document(

                    page_content=doc_data['page_content'],

                    metadata=doc_data['metadata']

                )

                documents.append(doc)

            logger.info(f"Loaded {len(documents)} processed documents for bot {bot_id}")

            return documents

        except Exception as e:

            logger.error(f"Error loading processed documents: {e}")

            return []

    def initialize_with_persistence(self, bot_id: str, embedding_model: str = 'BAAI/bge-small-en-v1.5'):

        try:

            state_loaded = self.load_processor_state(bot_id)

            if not self.embeddings:

                if self.initialize_embeddings(embedding_model):

                    st.success("Embeddings initialized successfully!")

                else:

                    st.error("Failed to initialize embeddings")

                    return False

            vectorstore = self.setup_vectorstore_with_persistence(bot_id)

            if vectorstore:

                if state_loaded:

                    st.success(f"Knowledge base restored! Found {self.documents_processed} processed documents.")

                else:

                    st.info("New knowledge base initialized and ready for documents.")

                return True

            else:

                st.error("Failed to setup vectorstore")

                return False

        except Exception as e:

            st.error(f"Error initializing knowledge processor: {e}")

            return False

    def setup_vectorstore_with_persistence(self, bot_id: str):

        try:

            if not self.embeddings:

                raise ValueError("Embeddings must be initialized first.")

            safe_bot_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(bot_id))

            self._persist_directory = os.path.join(self.vectordb_dir, self.user_id, safe_bot_id)

            os.makedirs(self._persist_directory, exist_ok=True)

            self._collection_name = self._create_safe_collection_name(bot_id)

            st.info(f"Setting up persistent vectorstore: {self._collection_name}")

            st.info(f"Persist directory: {self._persist_directory}")

            self.vectorstore = Chroma(

                collection_name=self._collection_name,

                embedding_function=self.embeddings,

                persist_directory=self._persist_directory

            )

            self._load_processed_files_tracking()

            doc_count = self.vectorstore._collection.count()

            st.success(f"Vectorstore ready. Found {doc_count} existing documents.")

            metadata = {

                'collection_name': self._collection_name,

                'persist_directory': self._persist_directory,

                'total_documents': doc_count,

                'embedding_model': getattr(self.embeddings, 'model_name', 'unknown')

            }

            self.save_vectorstore_metadata(bot_id, metadata)

            return self.vectorstore

        except Exception as e:

            st.error(f"Critical error during vectorstore setup: {str(e)}")

            return None

    def _load_processed_files_tracking(self):

        """Load list of already processed files to prevent duplicates"""

        try:

            if self.vectorstore:

                existing_docs = self.vectorstore.get()

                if existing_docs and 'metadatas' in existing_docs:

                    for metadata in existing_docs['metadatas']:

                        if metadata and 'file_hash' in metadata:

                            self._processed_files.add(metadata['file_hash'])

                        elif metadata and 'source' in metadata:

                            self._processed_files.add(metadata['source'])

        except Exception as e:

            logger.error(f"Error loading processed files tracking: {e}")

    def _create_safe_collection_name(self, bot_id: str) -> str:

        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(bot_id))

        if len(safe_name) > 50:

            safe_name = safe_name[:40] + "_" + uuid.uuid4().hex[:8]

        return f"bot_{safe_name}_docs"

    def _get_file_hash(self, file_content: bytes, filename: str) -> str:

        """Generate unique hash for file to prevent duplicates"""

        content_hash = hashlib.md5(file_content).hexdigest()

        name_hash = hashlib.md5(filename.encode()).hexdigest()

        return f"{content_hash}_{name_hash}"

    def _is_file_already_processed(self, file_hash: str, filename: str) -> bool:

        """Check if file is already processed"""

        return file_hash in self._processed_files or filename in self._processed_files

    def get_available_embedding_models(self):

        return {

            'BAAI/bge-small-en-v1.5': {

                'name': 'BGE Small English v1.5 (Recommended)',

                'size': '120MB',

                'description': 'Good performance, moderate size, latest version'

            },

            'sentence-transformers/all-MiniLM-L6-v2': {

                'name': 'All MiniLM L6 v2',

                'size': '80MB',

                'description': 'Fast, lightweight, good quality'

            },

            'sentence-transformers/all-mpnet-base-v2': {

                'name': 'All MPNet Base v2',

                'size': '420MB',

                'description': 'High quality, larger model'

            },

            'BAAI/bge-base-en-v1.5': {

                'name': 'BGE Base English',

                'size': '420MB',

                'description': 'High performance, larger size'

            }

        }

    def initialize_embeddings(self, model_name: str = 'BAAI/bge-small-en-v1.5'):

        try:

            if not EMBEDDINGS_AVAILABLE:

                raise ImportError("HuggingFace embeddings not available")

            st.info(f"Loading embedding model: {model_name}")

            self.embeddings = HuggingFaceEmbeddings(

                model_name=model_name,

                model_kwargs={'device': 'cpu'},

                encode_kwargs={'normalize_embeddings': True}

            )

            test_embedding = self.embeddings.embed_query("test")

            if len(test_embedding) > 0:

                st.success(f"Embeddings initialized! Dimension: {len(test_embedding)}")

                return True

            else:

                st.error("Embedding test failed - empty result")

                return False

        except Exception as e:

            logger.error(f"Embedding initialization error: {e}")

            st.error(f"Embedding initialization failed: {str(e)}")

            return False

    def get_text_splitter(self, chunking_strategy: str, chunk_size: int, chunk_overlap: int):

        """Get the appropriate text splitter based on strategy"""

        if st.session_state.get('debug_mode', False):

            st.info(f"Creating text splitter: {chunking_strategy}, size: {chunk_size}, overlap: {chunk_overlap}")

        if chunking_strategy == "recursive_character":

            st.info("Using Recursive Character Splitting (separates on paragraphs, then sentences, then words)")

            return RecursiveCharacterTextSplitter(

                chunk_size=chunk_size,

                chunk_overlap=chunk_overlap,

                length_function=len,

                separators=["\n\n", "\n", ". ", " ", ""]

            )

        elif chunking_strategy == "character":

            st.info("Using Character Splitting (splits on newlines only)")

            return CharacterTextSplitter(

                chunk_size=chunk_size,

                chunk_overlap=chunk_overlap,

                length_function=len,

                separator="\n"

            )

        elif chunking_strategy == "semantic":

            st.info("Using Semantic Splitting (LangChain Experimental - understands content meaning)")

            try:

                if not SEMANTIC_AVAILABLE:

                    st.warning("LangChain Experimental not available for true semantic chunking")

                    raise ImportError("langchain_experimental not installed")

                if not self.embeddings:

                    st.error("Embeddings required for semantic chunking")

                    raise ValueError("No embeddings available")

                semantic_chunker = SemanticChunker(

                    embeddings=self.embeddings,

                    breakpoint_threshold_type="percentile",

                    breakpoint_threshold_amount=80,

                    buffer_size=1

                )

                st.success("✅ True semantic chunker initialized")

                return semantic_chunker

            except (ImportError, ValueError):

                st.info("Using enhanced fallback semantic chunking")

                return RecursiveCharacterTextSplitter(

                    chunk_size=int(chunk_size * 1.5),

                    chunk_overlap=int(chunk_overlap * 1.5),

                    length_function=len,

                    separators=["\n\n\n", "\n\n", ".\n", ". ", "! ", "? ", "; ", ", ", " "]

                )

        else:

            st.warning(f"Unknown chunking strategy '{chunking_strategy}', using recursive_character")

            return RecursiveCharacterTextSplitter(

                chunk_size=chunk_size,

                chunk_overlap=chunk_overlap,

                length_function=len,

                separators=["\n\n", "\n", ". ", " ", ""]

            )

    def process_uploaded_files_with_persistence(self, uploaded_files: List, bot_id: str = None,

                                                chunk_size: int = 1000, chunk_overlap: int = 200,

                                                chunking_strategy: str = "recursive_character"):

        if not self.vectorstore:

            st.error("Vectorstore not initialized. Call initialize_with_persistence() first.")

            return 0

        if not uploaded_files:

            st.warning("No files to process")

            return 0

        processed_count = self.process_uploaded_files(

            uploaded_files, bot_id, chunk_size, chunk_overlap, chunking_strategy

        )

        if processed_count > 0:

            self.save_processor_state(bot_id)

            if self.data_manager and hasattr(self.data_manager, 'save_uploaded_file'):

                for i, uploaded_file in enumerate(uploaded_files):

                    try:

                        self.data_manager.save_uploaded_file(self.user_id, bot_id, i, uploaded_file)

                    except Exception as e:

                        logger.warning(f"Could not save file {uploaded_file.name}: {e}")

            self.auto_save_after_processing(bot_id)

            st.success(f"✅ Successfully processed and saved {processed_count} documents permanently!")

        return processed_count

    def process_uploaded_files(self, uploaded_files: List, bot_id: str = None,

                               chunk_size: int = 1000, chunk_overlap: int = 200,

                               chunking_strategy: str = "recursive_character"):

        """Process files and ADD documents to the vectorstore"""

        if not self.vectorstore:

            st.error("Vectorstore not initialized. Call setup_vectorstore() first.")

            return 0

        if not uploaded_files:

            st.warning("No files to process")

            return 0

        documents = []

        failed_files = []

        for uploaded_file in uploaded_files:

            try:

                st.info(f"Processing file: {uploaded_file.name}...")

                file_extension = uploaded_file.name.split('.')[-1].lower()

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:

                    temp_file.write(uploaded_file.getvalue())

                    temp_path = temp_file.name

                loader = None

                if file_extension == 'pdf':

                    loader = PyPDFLoader(temp_path)

                elif file_extension == 'txt':

                    loader = TextLoader(temp_path, encoding='utf-8')

                elif file_extension == 'csv':

                    loader = CSVLoader(temp_path)

                elif file_extension in ['md', 'markdown']:

                    loader = TextLoader(temp_path, encoding='utf-8')

                else:

                    st.warning(f"Unsupported file type: {uploaded_file.name}")

                    failed_files.append(uploaded_file.name)

                    os.unlink(temp_path)

                    continue

                docs = loader.load()

                if not docs:

                    st.warning(f"No content extracted from {uploaded_file.name}")

                    failed_files.append(uploaded_file.name)

                    os.unlink(temp_path)

                    continue

                for i, doc in enumerate(docs):

                    doc.metadata.update({

                        'source': uploaded_file.name,

                        'bot_id': bot_id,

                        'file_type': uploaded_file.type,

                        'chunk_id': f"{uploaded_file.name}_{i}",

                        'processed_at': str(datetime.now()),

                        'file_size': len(uploaded_file.getvalue()),

                        'chunking_strategy': chunking_strategy

                    })

                documents.extend(docs)

                st.success(f"Loaded {uploaded_file.name}: {len(docs)} pages")

                os.unlink(temp_path)

            except Exception as e:

                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                failed_files.append(uploaded_file.name)

                try:

                    os.unlink(temp_path)

                except:

                    pass

        if not documents:

            st.warning("No documents to process")

            return 0

        try:

            st.info(f"Splitting documents using '{chunking_strategy}' strategy...")

            text_splitter = self.get_text_splitter(chunking_strategy, chunk_size, chunk_overlap)

            split_docs = text_splitter.split_documents(documents)

            st.success(f"Documents split: {len(split_docs)} chunks")

        except Exception as e:

            st.error(f"Error splitting documents: {str(e)}")

            return 0

        if not split_docs:

            st.warning("No document chunks created")

            return 0

        try:

            st.info(f"Adding {len(split_docs)} chunks to vectorstore...")

            initial_count = self.vectorstore._collection.count()

            batch_size = 25

            for i in range(0, len(split_docs), batch_size):

                batch = split_docs[i:i+batch_size]

                try:

                    self.vectorstore.add_documents(batch)

                except Exception as batch_error:

                    st.warning(f"Error adding batch: {batch_error}")

            final_count = self.vectorstore._collection.count()

            added_count = final_count - initial_count

            if added_count > 0:

                st.success(f"✅ Successfully added {added_count} new chunks!")

                st.info(f"Total documents in vectorstore: {final_count}")

                self.documents_processed = final_count

                return added_count

            else:

                st.error("No documents were added")

                return 0

        except Exception as e:

            st.error(f"Critical error adding documents: {str(e)}")

            return 0

    def format_docs(self, docs: List) -> str:

        """Convert retrieved documents to formatted string"""

        if not docs:

            return "No relevant documents found."

        formatted_docs = []

        for i, doc in enumerate(docs, 1):

            source = doc.metadata.get('source', f'Document {i}')

            content = doc.page_content.strip()

            if len(content) > 1000:

                content = content[:1000] + "..."

            formatted_docs.append(f"**Source {i}: {source}**\n{content}")

        return "\n\n---\n\n".join(formatted_docs)

    def query_with_rag_chain(self, question: str, llm: Any, chat_history: List = None,

                            use_conversational: bool = True):

        """Query using the RAG chain"""

        try:

            if not self.vectorstore:

                return {

                    "success": False,

                    "error": "Vectorstore not initialized",

                    "answer": "I cannot access the document database. Please ensure your knowledge base is properly set up."

                }

            if use_conversational and chat_history:

                rag_chain = self.create_conversational_rag_chain(llm)

                response = rag_chain.invoke({

                    "question": question,

                    "chat_history": "\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in chat_history[-5:]])

                })

            else:

                rag_chain = self.create_simple_rag_chain(llm)

                response = rag_chain.invoke(question)

            try:

                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

                source_docs = retriever.invoke(question)

            except Exception as retrieval_error:

                logger.warning(f"Could not retrieve source docs: {retrieval_error}")

                source_docs = []

            return {

                "success": True,

                "answer": response,

                "source_docs": [

                    {

                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,

                        "source": doc.metadata.get('source', 'Unknown'),

                        "metadata": {k: v for k, v in doc.metadata.items() if k in ['source', 'file_type']}

                    } for doc in source_docs

                ]

            }

        except Exception as e:

            logger.error(f"RAG chain error: {e}")

            return {

                "success": False,

                "error": str(e),

                "answer": f"I encountered an error: {str(e)}"

            }

    def similarity_search(self, query: str, k: int = 4):

        """Perform similarity search"""

        if not self.vectorstore:

            return []

        try:

            docs_and_scores = self.vectorstore.similarity_search_with_score(query=query, k=k)

            return docs_and_scores

        except Exception as e:

            logger.error(f"Similarity search error: {e}")

            return []

    def create_simple_rag_chain(self, llm: Any, retriever_kwargs: Dict = None):

        """Create a simple RAG chain"""

        if retriever_kwargs is None:

            retriever_kwargs = {"k": 4}

        retriever = self.vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs=retriever_kwargs

        )

        prompt = PromptTemplate.from_template(

            """Use the following context to answer the question.

            Context: {context}

            Question: {question}

            Answer:"""

        )

        rag_chain = (

            {"context": retriever | (lambda docs: self.format_docs(docs)), "question": RunnablePassthrough()}

            | prompt

            | llm

            | StrOutputParser()

        )

        return rag_chain

    def create_conversational_rag_chain(self, llm: Any, retriever_kwargs: Dict = None):

        """Create a conversational RAG chain"""

        if retriever_kwargs is None:

            retriever_kwargs = {"k": 4}

        retriever = self.vectorstore.as_retriever(

            search_type="similarity",

            search_kwargs=retriever_kwargs

        )

        prompt = ChatPromptTemplate.from_template(

            """You are a helpful AI assistant. Use the following context to answer questions.

            Context: {context}

            Chat History: {chat_history}

            Question: {question}

            Answer:"""

        )

        rag_chain = (

            {"context": retriever | (lambda docs: self.format_docs(docs)), "question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}

            | prompt

            | llm

            | StrOutputParser()

        )

        return rag_chain

    def get_vectorstore_stats(self) -> Dict:

        """Get statistics about the vectorstore"""

        if not self.vectorstore:

            return {"initialized": False}

        try:

            doc_count = self.vectorstore._collection.count()

            return {

                "initialized": True,

                "total_documents": doc_count,

                "collection_name": self._collection_name,

                "persist_directory": self._persist_directory

            }

        except Exception as e:

            return {"initialized": True, "error": str(e), "total_documents": 0}

    def validate_knowledge_base(self, kb_config: Dict) -> tuple:

        """Validates the knowledge base configuration"""

        try:

            if not kb_config.get('enabled'):

                return True, "Knowledge base is disabled."

            if not LANGCHAIN_AVAILABLE:

                return False, "LangChain dependencies are not installed."

            chunk_size = kb_config.get('chunk_size', 1000)

            if not 100 <= chunk_size <= 4000:

                return False, "Chunk size must be between 100 and 4000."

            return True, "Knowledge base configuration is valid."

        except Exception as e:

            return False, f"Validation error: {e}"

    def test_vectorstore(self):

        """Test if vectorstore is working"""

        if not self.vectorstore:

            return False, "Vectorstore not initialized"

        try:

            doc_count = self.vectorstore._collection.count()

            return True, f"Vectorstore working! {doc_count} documents"

        except Exception as e:

            return False, f"Vectorstore test failed: {str(e)}"

    def auto_save_after_processing(self, bot_id: str):

        """Automatically save data after processing"""

        try:

            if self.data_manager and self.user_id:

                self.save_processor_state(bot_id)

            return True

        except Exception as e:

            logger.error(f"Auto-save error: {e}")

            return False

    def cleanup_bot_data(self, bot_id: str):

        """Clean up bot data"""

        try:

            if not self.data_manager or not self.user_id:

                return False

            state_file = os.path.join(self.kb_data_dir, f"{self.user_id}_{bot_id}_state.json")

            if os.path.exists(state_file):

                os.remove(state_file)

            st.success(f"✅ Cleaned up all data for bot {bot_id}")

            return True

        except Exception as e:

            st.error(f"Error cleaning up bot data: {e}")

            return False


def get_or_create_knowledge_processor(bot_id: str, embedding_model: str = 'BAAI/bge-small-en-v1.5'):

    """Get existing or create new knowledge processor"""

    try:

        processor = KnowledgeProcessor()

        if processor.initialize_with_persistence(bot_id, embedding_model):

            return processor

        return None

    except Exception as e:

        st.error(f"Error creating knowledge processor: {e}")

        return None
