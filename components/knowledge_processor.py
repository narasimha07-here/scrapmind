__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import tempfile
import streamlit as st
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
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

    def save_processed_documents(self, bot_id: str, documents: List[Document]):
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

    def load_processed_documents(self, bot_id: str) -> List[Document]:

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
        """Get the appropriate text splitter based on strategy with enhanced debugging"""

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
        
        elif chunking_strategy == "token":
            st.info("Using Token-based Splitting (sentence boundary focused)")
            try:
                from langchain.text_splitter import TokenTextSplitter
                return TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            except ImportError:

                st.warning("TokenTextSplitter not available, using fallback")
                return RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separators=[". ", "! ", "? ", "\n\n", "\n", " "]
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
        """Process files and ADD documents to the vectorstore - HANDLES CHUNKING STRATEGY CHANGES"""
        if not self.vectorstore:
            st.error("Vectorstore not initialized. Call setup_vectorstore() first.")
            return 0

        if not uploaded_files:
            st.warning("No files to process")
            return 0

        current_stored_strategy = None
        try:
            existing_docs = self.vectorstore.get(limit=1)
            if existing_docs and 'metadatas' in existing_docs and existing_docs['metadatas']:
                current_stored_strategy = existing_docs['metadatas'][0].get('chunking_strategy')
        except:
            current_stored_strategy = None

        strategy_changed = (current_stored_strategy is not None and 
                           current_stored_strategy != chunking_strategy)
        
        if strategy_changed:
            st.warning(f"🔄 Chunking strategy changed from '{current_stored_strategy}' to '{chunking_strategy}'")
            st.info("⚠️ **COMPLETELY REBUILDING** vectorstore to avoid mixing chunking methods...")
            

            if self.force_rebuild_vectorstore(bot_id, chunking_strategy):
                st.success("✅ Vectorstore completely rebuilt from scratch!")

                self._processed_files.clear()
                st.info("🔄 All files will be reprocessed with new chunking strategy")
            else:
                st.error("❌ Failed to rebuild vectorstore")
                return 0

        if strategy_changed:
            st.info("🔄 Reprocessing ALL files with new chunking strategy...")
            files_to_process = [(f, self._get_file_hash(f.getvalue(), f.name)) for f in uploaded_files]
            for _, file_hash in files_to_process:
                self._processed_files.add(file_hash)
        else:

            files_to_process = []
            skipped_files = []
            
            for uploaded_file in uploaded_files:
                file_content = uploaded_file.getvalue()
                file_hash = self._get_file_hash(file_content, uploaded_file.name)
                
                if self._is_file_already_processed(file_hash, uploaded_file.name):
                    skipped_files.append(uploaded_file.name)
                    st.info(f"Skipping {uploaded_file.name} (already processed)")
                else:
                    files_to_process.append((uploaded_file, file_hash))
            
            if skipped_files:
                st.info(f"Skipped {len(skipped_files)} already processed files")
            
            if not files_to_process:
                st.success("All files already processed - no duplicates added!")
                return 0

        documents = []
        failed_files = []
        
        for uploaded_file, file_hash in files_to_process:
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
                        'file_hash': file_hash,
                        'chunk_id': f"{uploaded_file.name}_{i}",
                        'processed_at': str(datetime.now()),
                        'file_size': len(uploaded_file.getvalue()),
                        'chunking_strategy': chunking_strategy,
                        'chunk_size_config': chunk_size,
                        'chunk_overlap_config': chunk_overlap
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
                continue

        if not documents:
            st.warning("No documents to process")
            return 0


        try:
            st.info(f"Splitting documents using '{chunking_strategy}' strategy...")
            text_splitter = self.get_text_splitter(chunking_strategy, chunk_size, chunk_overlap)
            split_docs = text_splitter.split_documents(documents)


            for i, doc in enumerate(split_docs):
                doc.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(split_docs),
                    'chunking_strategy': chunking_strategy, 
                    'chunk_size_used': chunk_size,
                    'chunk_overlap_used': chunk_overlap,
                    'original_length': len(doc.page_content)
                })

            st.success(f"Documents split using '{chunking_strategy}' strategy: {len(split_docs)} chunks")
        except Exception as e:
            st.error(f"Error splitting documents: {str(e)}")
            return 0

        if not split_docs:
            st.warning("No document chunks created")
            return 0


        try:
            st.info(f"Adding {len(split_docs)} chunks to vectorstore...")
            
            initial_count = self.vectorstore._collection.count()
            if strategy_changed:
                st.info(f"Starting with empty vectorstore (strategy changed)")
                expected_initial = 0
            else:
                expected_initial = initial_count


            batch_size = 25
            progress_bar = st.progress(0)
            
            for i in range(0, len(split_docs), batch_size):
                batch = split_docs[i:i+batch_size]
                try:
                    self.vectorstore.add_documents(batch)
                    progress = (i + batch_size) / len(split_docs)
                    progress_bar.progress(min(progress, 1.0))
                except Exception as batch_error:
                    st.warning(f"Error adding batch: {batch_error}")
                    continue

            final_count = self.vectorstore._collection.count()
            added_count = final_count - expected_initial
            progress_bar.progress(1.0)

            if added_count > 0:
                if strategy_changed:
                    st.success(f"✅ Successfully rebuilt vectorstore with {final_count} chunks using '{chunking_strategy}' strategy!")
                else:
                    st.success(f"✅ Successfully added {added_count} new chunks using '{chunking_strategy}' strategy!")
                st.info(f"Total documents in vectorstore: {final_count}")

                self.documents_processed = final_count
                return added_count
            else:
                st.error("No documents were added")
                return 0

        except Exception as e:
            st.error(f"Critical error adding documents: {str(e)}")
            return 0

    def force_rebuild_vectorstore(self, bot_id: str, chunking_strategy: str = None):
        """Force rebuild the entire vectorstore - COMPLETELY REBUILT VERSION"""
        try:
            st.info(f"🔄 **COMPLETELY REBUILDING** vectorstore for bot: {bot_id}")

            if self.vectorstore:
                try:
                    collection_name = self._collection_name
                    if hasattr(self.vectorstore, '_collection'):
                        self.vectorstore._collection.delete() 
                        st.info(f"Deleted collection: {collection_name}")

                    if hasattr(self.vectorstore, '_client'):
                        self.vectorstore._client.reset()
                except Exception as e:
                    st.warning(f"Error cleaning existing collection: {e}")

            if self._persist_directory and os.path.exists(self._persist_directory):
                try:
                    import stat
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    
                    shutil.rmtree(self._persist_directory, onerror=handle_remove_readonly)
                    st.info(f"✅ Deleted persist directory: {self._persist_directory}")

                    time.sleep(1)
                except Exception as e:
                    st.warning(f"Could not delete persist directory: {e}")

            self._processed_files.clear()
            self.vectorstore = None
            self._persist_directory = None
            self._collection_name = None

            timestamp = int(datetime.now().timestamp())
            new_bot_id = f"{bot_id}_{timestamp}"
            st.info(f"Creating new vectorstore with ID: {new_bot_id}")
            
            new_vectorstore = self.setup_vectorstore_with_persistence(new_bot_id)
            
            if new_vectorstore:

                try:
                    doc_count = new_vectorstore._collection.count()
                    if doc_count == 0:
                        st.success(f"✅ Vectorstore completely rebuilt - verified empty ({doc_count} documents)")

                        if chunking_strategy:
                            self._last_chunking_strategy = chunking_strategy
                            st.info(f"Updated chunking strategy to: {chunking_strategy}")
                        
                        return True
                    else:
                        st.error(f"❌ Rebuild failed - new vectorstore still contains {doc_count} documents")
                        return False
                except Exception as e:
                    st.warning(f"Could not verify empty vectorstore: {e}")

                    return True
            else:
                st.error("❌ Failed to create new vectorstore")
                return False

        except Exception as e:
            st.error(f"❌ Critical rebuild error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def process_manual_text_with_persistence(self, manual_text: str, bot_id: str,
                                             chunk_size: int = 1000, chunk_overlap: int = 200,
                                             chunking_strategy: str = "recursive_character"):
        processed_count = self.process_manual_text(
            manual_text, bot_id, chunk_size, chunk_overlap, chunking_strategy
        )
        
        if processed_count > 0:

            self.save_processor_state(bot_id)

            self._save_manual_text_permanently(bot_id, manual_text)

            self.auto_save_after_processing(bot_id)
            st.success(f"✅ Manual text processed and saved permanently!")
        
        return processed_count

    def process_manual_text(self, manual_text: str, bot_id: str,
                            chunk_size: int = 1000, chunk_overlap: int = 200,
                            chunking_strategy: str = "recursive_character"):
        """Process manual text input and add to vectorstore - PREVENTS DUPLICATES"""
        if not manual_text.strip():
            return 0

        if not self.vectorstore:
            st.error("Vectorstore not initialized")
            return 0

        text_hash = hashlib.md5(manual_text.encode()).hexdigest()
        manual_identifier = f"manual_text_{text_hash}"
        
        if self._is_file_already_processed(manual_identifier, manual_identifier):
            st.info("Manual text already processed - skipping to prevent duplicates")
            return 0

        try:
            st.info(f"Processing NEW manual text using '{chunking_strategy}' chunking...")
            
            doc = Document(
                page_content=manual_text,
                metadata={
                    'source': 'manual_text',
                    'bot_id': bot_id,
                    'type': 'manual',
                    'file_hash': manual_identifier,
                    'processed_at': str(datetime.now()),
                    'chunking_strategy': chunking_strategy
                }
            )

            if len(manual_text) > chunk_size:

                text_splitter = self.get_text_splitter(chunking_strategy, chunk_size, chunk_overlap)
                split_docs = text_splitter.split_documents([doc])
            else:
                split_docs = [doc]

            for i, split_doc in enumerate(split_docs):
                split_doc.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(split_docs),
                    'chunking_strategy': chunking_strategy
                })

            initial_count = self.vectorstore._collection.count()
            self.vectorstore.add_documents(split_docs)
            final_count = self.vectorstore._collection.count()
            added_count = final_count - initial_count

            self._processed_files.add(manual_identifier)
            st.success(f"Added {added_count} NEW manual text chunks using '{chunking_strategy}' strategy")
            
            return len(split_docs)

        except Exception as e:
            st.error(f"Error processing manual text: {str(e)}")
            return 0

    def _save_manual_text_permanently(self, bot_id: str, manual_text: str):
        """Save manual text to permanent storage"""
        try:
            if not self.data_manager or not self.user_id:
                return False

            manual_text_file = os.path.join(
                self.processed_files_dir,
                f"{self.user_id}_{bot_id}_manual_text.json"
            )

            manual_texts = []
            if os.path.exists(manual_text_file):
                with open(manual_text_file, 'r', encoding='utf-8') as f:
                    manual_texts = json.load(f)

            text_entry = {
                'text': manual_text,
                'added_at': datetime.now().isoformat(),
                'text_hash': hashlib.md5(manual_text.encode()).hexdigest()
            }
            manual_texts.append(text_entry)

            with open(manual_text_file, 'w', encoding='utf-8') as f:
                json.dump(manual_texts, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            logger.error(f"Error saving manual text permanently: {e}")
            return False

    def load_manual_texts(self, bot_id: str) -> List[str]:
        """Load saved manual texts"""
        try:
            if not self.data_manager or not self.user_id:
                return []

            manual_text_file = os.path.join(
                self.processed_files_dir,
                f"{self.user_id}_{bot_id}_manual_text.json"
            )

            if os.path.exists(manual_text_file):
                with open(manual_text_file, 'r', encoding='utf-8') as f:
                    manual_texts = json.load(f)
                return [entry['text'] for entry in manual_texts]
            
            return []
        except Exception as e:
            logger.error(f"Error loading manual texts: {e}")
            return []

    def format_docs(self, docs: List[Document]) -> str:
        """Convert retrieved documents to formatted string with better formatting"""
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
        """Query using the RAG chain with enhanced error handling and proper input formatting"""
        try:
            if not self.vectorstore:
                return {
                    "success": False,
                    "error": "Vectorstore not initialized",
                    "answer": "I cannot access the document database. Please ensure your knowledge base is properly set up."
                }

            if use_conversational and chat_history:

                history_text = ""
                for msg in chat_history[-5:]:
                    role = msg.get('role', 'unknown').title()
                    content = msg.get('content', '')
                    if content and role in ['User', 'Assistant']:
                        history_text += f"{role}: {content[:200]}...\n" if len(content) > 200 else f"{role}: {content}\n"

                rag_chain = self.create_conversational_rag_chain(llm)
                response = rag_chain.invoke({
                    "question": question,
                    "chat_history": history_text
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
                        "metadata": {k: v for k, v in doc.metadata.items() if k in ['source', 'file_type', 'chunk_index']}
                    } for doc in source_docs
                ]
            }

        except Exception as e:
            logger.error(f"RAG chain error: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"I encountered an error while searching the documents: {str(e)}"
            }
        
    def similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.0):
        """Perform similarity search with scores and better error handling"""
        if not self.vectorstore:
            return []

        try:
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )

            filtered_docs = [
                (doc, score) for doc, score in docs_and_scores
                if (score <= score_threshold if score_threshold > 0 else True) 
            ]

            filtered_docs.sort(key=lambda x: x[1])

            return filtered_docs
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            st.error(f"Search error: {str(e)}")
            return []


    def improve_similarity_search(self, query: str, k: int = 4, score_threshold: float = 0.7):
        """Enhanced similarity search with multiple strategies and better filtering."""
        if not self.vectorstore:
            st.warning("Vectorstore not initialized for improved search.")
            return []

        all_results = []
        seen_content_hashes = set()

        try:

            initial_results = self.vectorstore.similarity_search_with_score(query=query, k=k * 2) 
            
            for doc, score in initial_results:
            
                doc_identifier = hashlib.md5((doc.page_content + doc.metadata.get('source', '')).encode()).hexdigest()
                if doc_identifier not in seen_content_hashes:
                    seen_content_hashes.add(doc_identifier)
                    all_results.append((doc, score))

            if not all_results or (all_results and all_results[0][1] > score_threshold) or len(all_results) < k:
                keywords = re.findall(r'\b\w{4,}\b', query.lower()) 
                
                for keyword in list(set(keywords)): 
                    try:
                        keyword_results = self.vectorstore.similarity_search_with_score(query=keyword, k=k)
                        for doc, score in keyword_results:
                            doc_identifier = hashlib.md5((doc.page_content + doc.metadata.get('source', '')).encode()).hexdigest()
                            if doc_identifier not in seen_content_hashes:
                                seen_content_hashes.add(doc_identifier)
                                all_results.append((doc, score))
                    except Exception as kw_e:
                        logger.warning(f"Keyword search for '{keyword}' failed: {kw_e}")

            all_results.sort(key=lambda x: x[1])

            final_filtered_results = []
            for doc, score in all_results:
                if score <= score_threshold:
                    final_filtered_results.append((doc, score))
                if len(final_filtered_results) >= k: 
                    break

            return final_filtered_results

        except Exception as e:
            logger.error(f"Improved similarity search error: {e}")
            st.error(f"Enhanced search error: {str(e)}")
            return []


    def create_simple_rag_chain(self, llm: Any, retriever_kwargs: Dict = None):
        """Create a simple RAG chain that properly handles the data flow - FIXED VERSION"""
        if retriever_kwargs is None:
            retriever_kwargs = {"k": 4}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=retriever_kwargs
        )

        prompt = PromptTemplate.from_template(
            """Use the following context to answer the question. If you cannot answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        )


        from langchain_core.runnables import RunnableLambda

        rag_chain = (
            {
                "context": retriever | RunnableLambda(self.format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    def create_conversational_rag_chain(self, llm: Any, retriever_kwargs: Dict = None):
        """Create a conversational RAG chain that properly handles chat history - FIXED VERSION"""
        if retriever_kwargs is None:
            retriever_kwargs = {"k": 4}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=retriever_kwargs
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful AI assistant. Use the following context from documents and conversation history to answer the user's question.

Previous conversation:
{chat_history}

Context from documents:
{context}

Current question: {question}

Instructions:
- Answer based primarily on the provided context
- If the information is not in the context, say so clearly
- Reference specific sources when possible
- Maintain conversation continuity
- Be concise but comprehensive

Answer:"""
        )


        def extract_question_for_retrieval(inputs):
            """Extract question for retrieval"""
            if isinstance(inputs, dict):
                return inputs.get("question", "")
            return str(inputs)

        def extract_question(inputs):
            """Extract question for prompt"""
            if isinstance(inputs, dict):
                return inputs.get("question", "")
            return str(inputs)

        def extract_chat_history(inputs):
            """Extract chat history for prompt"""
            if isinstance(inputs, dict):
                return inputs.get("chat_history", "")
            return ""

        rag_chain = (
            {
                "context": RunnableLambda(extract_question_for_retrieval) | retriever | RunnableLambda(self.format_docs),
                "question": RunnableLambda(extract_question),
                "chat_history": RunnableLambda(extract_chat_history)
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    def create_fallback_rag_chain(self, llm: Any, retriever_kwargs: Dict = None):
        """Fallback RAG implementation that doesn't use complex chaining"""
        if retriever_kwargs is None:
            retriever_kwargs = {"k": 4}

        def rag_query(question: str, chat_history: str = "") -> str:
            """Direct RAG query without complex chaining"""
            try:
                search_results = self.improve_similarity_search(question, k=retriever_kwargs["k"])
                if not search_results:
                    return "I couldn't find any relevant information in your documents to answer this question."
                
                context = self.format_docs([doc for doc, score in search_results])

                if chat_history:
                    prompt_text = f"""You are a helpful AI assistant. Use the following context from documents and conversation history to answer the user's question.

Previous conversation:
{chat_history}

Context from documents:
{context}

Current question: {question}

Instructions:
- Answer based primarily on the provided context
- If the information is not in the context, say so clearly
- Reference specific sources when possible
- Maintain conversation continuity
- Be concise but comprehensive

Answer:"""
                else:
                    prompt_text = f"""Use the following context to answer the question. If you cannot answer based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

                response = llm.invoke(prompt_text)
                return response

            except Exception as e:
                return f"Error generating response: {str(e)}"

        return rag_query

    def get_vectorstore_stats(self) -> Dict:
        """Get statistics about the vectorstore"""
        if not self.vectorstore:
            return {"initialized": False}

        try:
            doc_count = self.vectorstore._collection.count()

            sample_docs = self.vectorstore.get(limit=10)
            sources = set()
            file_types = set()
            if sample_docs and 'metadatas' in sample_docs:
                for metadata in sample_docs['metadatas']:
                    if metadata:
                        if 'source' in metadata:
                            sources.add(metadata['source'])
                        if 'file_type' in metadata:
                            file_types.add(metadata['file_type'])

            return {
                "initialized": True,
                "total_documents": doc_count,
                "unique_sources": len(sources),
                "file_types": list(file_types),
                "collection_name": self._collection_name,
                "persist_directory": self._persist_directory
            }

        except Exception as e:
            return {
                "initialized": True,
                "error": str(e),
                "total_documents": 0
            }

    def validate_knowledge_base(self, kb_config: Dict) -> tuple[bool, str]:
        """Validates the knowledge base configuration to ensure it's usable"""
        try:
            if not kb_config.get('enabled'):
                return True, "Knowledge base is disabled."

            if not LANGCHAIN_AVAILABLE:
                return False, "LangChain dependencies are not installed."

            data_source = kb_config.get('data_source')
            if not data_source:
                return False, "No data source was specified."

            if data_source == "Upload Documents":
                if not kb_config.get('uploaded_files') and not kb_config.get('file_names'):
                    return False, "No documents were uploaded for processing."

            elif data_source == "Enter Text Manually":
                manual_text = kb_config.get('manual_text', '').strip()
                if not manual_text:
                    return False, "Manual text content is empty."
                if len(manual_text) < 10:
                    return False, "Manual text is too short (minimum 10 characters)."

            elif data_source == "Web URLs":
                urls = kb_config.get('urls', [])
                if not urls:
                    return False, "No URLs were provided."
                for url in urls:
                    if not url.strip().startswith(('http://', 'https://')):
                        return False, f"Invalid URL format: {url}"

            elif data_source == "FAQ Database":
                faqs = kb_config.get('faqs', [])
                if not faqs:
                    return False, "No FAQ entries have been provided."
                for i, faq in enumerate(faqs):
                    if not faq.get('question', '').strip() or not faq.get('answer', '').strip():
                        return False, f"FAQ #{i+1} is missing a question or answer."

            chunk_size = kb_config.get('chunk_size', 1000)
            if not 100 <= chunk_size <= 4000:
                return False, "Chunk size must be between 100 and 4000."

            chunk_overlap = kb_config.get('chunk_overlap', 200)
            if not 0 <= chunk_overlap < chunk_size:
                return False, "Chunk overlap must be less than chunk size."

            return True, "Knowledge base configuration is valid."

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, f"An unexpected error occurred during validation: {e}"

    def test_vectorstore(self):
        """Test if vectorstore is working properly with enhanced testing"""
        if not self.vectorstore:
            return False, "Vectorstore not initialized"

        try:
            # Test basic functionality
            doc_count = self.vectorstore._collection.count()
            if doc_count == 0:
                return True, "Vectorstore working but empty (0 documents)"

            # Test search functionality
            test_query = "test search query"
            search_results = self.vectorstore.similarity_search(test_query, k=1)

            if search_results:
                return True, f"Vectorstore working perfectly! {doc_count} documents, search functional"
            else:
                return True, f"Vectorstore working with {doc_count} documents (search returned no results for test query)"

        except Exception as e:
            return False, f"Vectorstore test failed: {str(e)}"

    def auto_save_after_processing(self, bot_id: str):
        """Automatically save all data after processing"""
        try:
            if self.data_manager and self.user_id:

                self.save_processor_state(bot_id)
                
           
                if 'bots' in st.session_state and bot_id in st.session_state.bots:
                    self.data_manager.save_user_bots(self.user_id, st.session_state.bots)

                self.data_manager.save_all_user_data()
                return True
        except Exception as e:
            logger.error(f"Auto-save error: {e}")
            return False

    def rebuild_knowledge_base_from_storage(self, bot_id: str):

        try:
            st.info("Rebuilding knowledge base from stored data...")
            
            if not self.force_rebuild_vectorstore(bot_id):
                st.error("Failed to rebuild vectorstore")
                return False

            total_processed = 0

            if self.data_manager and hasattr(self.data_manager, 'load_uploaded_file'):

                bots = st.session_state.get('bots', {})
                if bot_id in bots:
                    kb = bots[bot_id].get('knowledge_base', {})
                    file_metadata = kb.get('file_metadata', [])
                    
                    if file_metadata:
                        st.info(f"Found {len(file_metadata)} stored files to reload...")
                        restored_files = []
                        
                        for file_meta in file_metadata:
                            file_content = self.data_manager.load_uploaded_file(
                                self.user_id, bot_id, file_meta['index']
                            )
                            
                            if file_content:

                                class MockFile:
                                    def __init__(self, name, content, file_type):
                                        self.name = name
                                        self.type = file_type
                                        self._content = content
                                    
                                    def getvalue(self):
                                        return self._content

                                mock_file = MockFile(
                                    file_meta['name'],
                                    file_content,
                                    file_meta.get('type', 'unknown')
                                )
                                restored_files.append(mock_file)

                        if restored_files:
                            chunk_size = kb.get('chunk_size', 1000)
                            chunk_overlap = kb.get('chunk_overlap', 200)
                            chunking_strategy = kb.get('chunking_strategy', 'recursive_character')
                            
                            processed = self.process_uploaded_files(
                                restored_files, bot_id, chunk_size, chunk_overlap, chunking_strategy
                            )
                            total_processed += processed
                            st.success(f"Reprocessed {processed} documents from stored files")

            manual_texts = self.load_manual_texts(bot_id)
            if manual_texts:
                st.info(f"Found {len(manual_texts)} stored manual texts...")
                for i, manual_text in enumerate(manual_texts):
                    processed = self.process_manual_text(manual_text, bot_id)
                    total_processed += processed
                st.success(f"Reprocessed {len(manual_texts)} manual text entries")

            if total_processed > 0:
                self.save_processor_state(bot_id)
                st.success(f"✅ Knowledge base rebuilt! Total: {total_processed} documents")
                return True
            else:
                st.warning("No stored data found to rebuild knowledge base")
                return False

        except Exception as e:
            st.error(f"Error rebuilding knowledge base: {e}")
            return False

    def cleanup_bot_data(self, bot_id: str):
        try:
            if not self.data_manager or not self.user_id:
                return False

            state_file = os.path.join(self.kb_data_dir, f"{self.user_id}_{bot_id}_state.json")
            if os.path.exists(state_file):
                os.remove(state_file)

            metadata_file = os.path.join(self.vectordb_dir, f"{self.user_id}_{bot_id}_metadata.json")
            if os.path.exists(metadata_file):
                os.remove(metadata_file)

            docs_file = os.path.join(self.processed_files_dir, f"{self.user_id}_{bot_id}_documents.pkl")
            if os.path.exists(docs_file):
                os.remove(docs_file)

            manual_text_file = os.path.join(self.processed_files_dir, f"{self.user_id}_{bot_id}_manual_text.json")
            if os.path.exists(manual_text_file):
                os.remove(manual_text_file)

            vectorstore_dir = os.path.join(self.vectordb_dir, self.user_id, bot_id)
            if os.path.exists(vectorstore_dir):
                shutil.rmtree(vectorstore_dir)

            if hasattr(self.data_manager, 'delete_bot_files'):
                self.data_manager.delete_bot_files(self.user_id, bot_id)

            st.success(f"✅ Cleaned up all data for bot {bot_id}")
            return True
        except Exception as e:
            st.error(f"Error cleaning up bot data: {e}")
            return False

    def get_storage_stats(self, bot_id: str) -> Dict:
        
        try:
            stats = {
                'vectorstore_size': 0,
                'processed_files_size': 0,
                'manual_texts_count': 0,
                'total_documents': 0,
                'storage_path': None
            }

            if not self.data_manager or not self.user_id:
                return stats

            vectorstore_dir = os.path.join(self.vectordb_dir, self.user_id, bot_id)
            if os.path.exists(vectorstore_dir):
                stats['storage_path'] = vectorstore_dir
                for root, dirs, files in os.walk(vectorstore_dir):
                    stats['vectorstore_size'] += sum(
                        os.path.getsize(os.path.join(root, file)) for file in files
                    )

            docs_file = os.path.join(self.processed_files_dir, f"{self.user_id}_{bot_id}_documents.pkl")
            if os.path.exists(docs_file):
                stats['processed_files_size'] = os.path.getsize(docs_file)

            manual_texts = self.load_manual_texts(bot_id)
            stats['manual_texts_count'] = len(manual_texts)

            if self.vectorstore:
                try:
                    stats['total_documents'] = self.vectorstore._collection.count()
                except:
                    pass

            return stats
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}

    @classmethod
    def create_with_persistence(cls, bot_id: str, embedding_model: str = 'BAAI/bge-small-en-v1.5'):
        """Class method to create knowledge processor with persistence enabled"""
        try:

            data_manager = None
            if 'data_manager' in st.session_state:
                data_manager = st.session_state.data_manager
            else:
                from components.data_manager import DataManager 
                data_manager = DataManager()
                st.session_state.data_manager = data_manager

            processor = cls(data_manager=data_manager)

            if processor.initialize_with_persistence(bot_id, embedding_model):
                st.success("✅ Knowledge processor created with persistent storage!")
                return processor
            else:
                st.error("❌ Failed to create persistent knowledge processor")
                return None
        except Exception as e:
            st.error(f"Error creating persistent knowledge processor: {e}")
            return None

    @classmethod
    def restore_from_storage(cls, bot_id: str):
        """Restore knowledge processor from permanent storage"""
        try:

            data_manager = None
            if 'data_manager' in st.session_state:
                data_manager = st.session_state.data_manager
            else:
                from components.data_manager import DataManager # Corrected import path
                data_manager = DataManager()
                st.session_state.data_manager = data_manager

            # Create processor
            processor = cls(data_manager=data_manager)
            
            # Try to load existing state
            if processor.load_processor_state(bot_id):
                # Try to restore vectorstore
                if processor._persist_directory and os.path.exists(processor._persist_directory):
                    # Initialize embeddings first
                    metadata = processor.load_vectorstore_metadata(bot_id)
                    embedding_model = metadata.get('embedding_model', 'BAAI/bge-small-en-v1.5') if metadata else 'BAAI/bge-small-en-v1.5'
                    
                    if processor.initialize_embeddings(embedding_model):
                        # Setup vectorstore with existing persistence
                        vectorstore = processor.setup_vectorstore_with_persistence(bot_id)
                        if vectorstore:
                            st.success(f"✅ Knowledge processor restored from storage!")
                            return processor

            # If restoration failed, return new processor
            st.info("No existing knowledge base found, creating new one...")
            return cls.create_with_persistence(bot_id)

        except Exception as e:
            st.error(f"Error restoring knowledge processor: {e}")
            return None

    def export_knowledge_base(self, bot_id: str) -> Optional[str]:
        """Export all knowledge base data to a file"""
        try:
            if not self.data_manager or not self.user_id:
                return None

            export_data = {
                'bot_id': bot_id,
                'user_id': self.user_id,
                'processor_state': {
                    'documents_processed': self.documents_processed,
                    'processed_files': list(self._processed_files),
                    'embedding_model': getattr(self.embeddings, 'model_name', None) if self.embeddings else None
                },
                'vectorstore_stats': self.get_vectorstore_stats(),
                'storage_stats': self.get_storage_stats(bot_id),
                'manual_texts': self.load_manual_texts(bot_id),
                'export_date': datetime.now().isoformat(),
                'version': '1.0'
            }

            export_file = os.path.join(
                self.kb_data_dir,
                f"{self.user_id}_{bot_id}_kb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            st.success(f"Knowledge base exported to: {export_file}")
            return export_file

        except Exception as e:
            st.error(f"Error exporting knowledge base: {e}")
            return None

    def get_knowledge_base_info(self, bot_id: str) -> Dict:
        """Get comprehensive information about the knowledge base"""
        try:
            info = {
                'bot_id': bot_id,
                'user_id': self.user_id,
                'initialized': bool(self.vectorstore),
                'embeddings_loaded': bool(self.embeddings),
                'documents_processed': self.documents_processed,
                'processed_files_count': len(self._processed_files),
                'vectorstore_stats': self.get_vectorstore_stats(),
                'storage_stats': self.get_storage_stats(bot_id),
                'manual_texts_count': len(self.load_manual_texts(bot_id)),
                'persistence_enabled': bool(self.data_manager),
                'last_updated': datetime.now().isoformat()
            }

            if self.vectorstore:
                working, message = self.test_vectorstore()
                info['vectorstore_working'] = working
                info['vectorstore_status'] = message

            return info
        except Exception as e:
            logger.error(f"Error getting knowledge base info: {e}")
            return {'error': str(e)}

    def health_check(self, bot_id: str) -> Dict:
        """Perform comprehensive health check"""
        health = {
            'overall_status': 'healthy',
            'checks': {},
            'recommendations': [],
            'errors': []
        }

        try:

            if not self.data_manager:
                health['checks']['data_manager'] = 'missing'
                health['errors'].append('DataManager not available - persistence disabled')
                health['recommendations'].append('Initialize DataManager for persistent storage')
            else:
                health['checks']['data_manager'] = 'ok'

            if not self.user_id:
                health['checks']['user_id'] = 'missing'
                health['errors'].append('User ID not set - cannot save data')
                health['recommendations'].append('Set user_id in session state')
            else:
                health['checks']['user_id'] = 'ok'

            if not self.embeddings:
                health['checks']['embeddings'] = 'missing'
                health['errors'].append('Embeddings not initialized')
                health['recommendations'].append('Initialize embeddings before processing documents')
            else:
                health['checks']['embeddings'] = 'ok'

            if not self.vectorstore:
                health['checks']['vectorstore'] = 'missing'
                health['errors'].append('Vectorstore not initialized')
                health['recommendations'].append('Setup vectorstore before adding documents')
            else:
                working, message = self.test_vectorstore()
                health['checks']['vectorstore'] = 'ok' if working else 'error'
                if not working:
                    health['errors'].append(f'Vectorstore error: {message}')

            if hasattr(self, 'kb_data_dir') and os.path.exists(self.kb_data_dir):
                health['checks']['storage_paths'] = 'ok'
            else:
                health['checks']['storage_paths'] = 'missing'
                health['recommendations'].append('Initialize storage paths')

            if self.data_manager and self.user_id:
                state_file = os.path.join(self.kb_data_dir, f"{self.user_id}_{bot_id}_state.json")
                if os.path.exists(state_file):
                    health['checks']['saved_state'] = 'found'
                else:
                    health['checks']['saved_state'] = 'none'

            if health['errors']:
                health['overall_status'] = 'error'
            elif health['recommendations']:
                health['overall_status'] = 'warning'

            return health

        except Exception as e:
            health['overall_status'] = 'critical'
            health['errors'].append(f'Health check failed: {e}')
            return health

    def debug_vectorstore_contents(self):
        """Debug function to check what's actually in the vectorstore"""
        if not self.vectorstore:
            st.error("Vectorstore not initialized")
            return

        try:

            doc_count = self.vectorstore._collection.count()
            st.write(f"Total documents in vectorstore: {doc_count}")

            sample_docs = self.vectorstore.get(limit=5)
            if sample_docs and 'documents' in sample_docs:
                st.write("Sample document contents:")
                for i, (doc, metadata) in enumerate(zip(sample_docs['documents'], sample_docs.get('metadatas', []))):
                    st.write(f"**Document {i+1}:**")
                    st.write(f"Source: {metadata.get('source', 'Unknown') if metadata else 'No metadata'}")
                    st.write(f"Content preview: {doc[:200]}...")
                    st.write("---")

            test_query = st.text_input("Test search query:", "")
            if test_query:
                results = self.vectorstore.similarity_search_with_score(test_query, k=3)
                st.write(f"Search results for '{test_query}':")
                for i, (doc, score) in enumerate(results):
                    st.write(f"**Result {i+1} (Score: {score:.3f}):**")
                    st.write(f"Source: {doc.metadata.get('source', 'Unknown')}")
                    st.write(f"Content: {doc.page_content[:300]}...")
                    st.write("---")

        except Exception as e:
            st.error(f"Debug error: {e}")

    def quick_search_test(self, query: str):
        """Quick test to see what documents are retrieved for a query"""
        if not self.vectorstore:
            return "Vectorstore not initialized"

        try:

            total_docs = self.vectorstore._collection.count()
            results = self.vectorstore.similarity_search_with_score(query, k=5)

            output = f"Total documents: {total_docs}\n"
            output += f"Query: '{query}'\n"
            output += f"Results found: {len(results)}\n\n"

            for i, (doc, score) in enumerate(results):
                output += f"Result {i+1} (Score: {score:.3f}):\n"
                output += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                output += f"Content: {doc.page_content[:200]}...\n"
                output += "-" * 50 + "\n"

            return output

        except Exception as e:
            return f"Search test failed: {str(e)}"



def get_or_create_knowledge_processor(bot_id: str, embedding_model: str = 'BAAI/bge-small-en-v1.5'):
    """Get existing or create new knowledge processor with persistence"""
    try:

        session_key = f"knowledge_processor_{bot_id}"
        if session_key in st.session_state:
            processor = st.session_state[session_key]

            if processor and processor.vectorstore:
                return processor

        processor = KnowledgeProcessor.restore_from_storage(bot_id)
        if not processor:
            processor = KnowledgeProcessor.create_with_persistence(bot_id, embedding_model)

        if processor:
            st.session_state[session_key] = processor

        return processor

    except Exception as e:
        st.error(f"Error getting knowledge processor: {e}")
        return None


def cleanup_knowledge_processor(bot_id: str):
    """Clean up knowledge processor and its data"""
    try:

        session_key = f"knowledge_processor_{bot_id}"
        processor = st.session_state.get(session_key)

        if processor:
            processor.cleanup_bot_data(bot_id)
            del st.session_state[session_key]

        return True
    except Exception as e:
        st.error(f"Error cleaning up knowledge processor: {e}")
        return False


def save_all_knowledge_processors():
    """Save state of all active knowledge processors"""
    try:
        saved_count = 0
        for key, processor in st.session_state.items():
            if key.startswith("knowledge_processor_") and isinstance(processor, KnowledgeProcessor):
                bot_id = key.replace("knowledge_processor_", "")
                if processor.save_processor_state(bot_id):
                    saved_count += 1

        if saved_count > 0:
            st.success(f"✅ Saved {saved_count} knowledge processors")
        return saved_count > 0

    except Exception as e:
        st.error(f"Error saving knowledge processors: {e}")
        return False

