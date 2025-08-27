import streamlit as st
import json
import time
from typing import Dict, List, Optional, Any, Iterator
from datetime import datetime
import base64

from langchain_core.language_models.llms import LLM
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk

try:
    from utils.openrouter_client import OpenRouterClient
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    from components.knowledge_processor import KnowledgeProcessor
    KNOWLEDGE_PROCESSOR_AVAILABLE = True
except ImportError:
    KNOWLEDGE_PROCESSOR_AVAILABLE = False

try:
    from components.data_manager import DataManager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False

try:
    from components.voice_config import VoiceConfig
    VOICE_CONFIG_AVAILABLE = True
except ImportError:
    VOICE_CONFIG_AVAILABLE = False

class OpenRouterLLM:
    def __init__(self, openrouter_client: OpenRouterClient, model: str = "meta-llama/llama-3.2-3b-instruct:free", 
                 temperature: float = 0.7):
        self.openrouter_client = openrouter_client
        self.model = model
        self.temperature = temperature
        self.retry_models = [
            "meta-llama/llama-3.2-3b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "google/gemma-2-9b-it:free"
        ]
    
    def invoke_with_retry(self, prompt: str, **kwargs) -> str:
        models_to_try = [self.model] + [m for m in self.retry_models if m != self.model]
        
        for attempt, model in enumerate(models_to_try):
            try:
                print(f"Attempt {attempt + 1}: Trying model {model}")
                
                messages = [{"role": "user", "content": prompt}]
                
                response = self.openrouter_client.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=self.temperature,
                    **kwargs
                )
                
                if response.get('success', False):
                    content = response.get('content', '').strip()
                    if content:
                        return content
                    else:
                        print(f"Empty response from {model}")
                        continue
                else:
                    error_msg = response.get('error', 'Unknown error')
                    print(f"Model {model} failed: {error_msg}")
                    continue
                    
            except Exception as e:
                print(f"Exception with {model}: {str(e)}")
                continue
        
        return "I apologize, but I'm experiencing technical difficulties with the AI service. Please try again in a moment."

    def invoke(self, prompt: str, **kwargs) -> str:
        try:
            if len(prompt) > 8000:
                prompt = prompt[:8000] + "\n\n[Content truncated due to length]"
            
            return self.invoke_with_retry(prompt, **kwargs)
            
        except Exception as e:
            print(f"Critical error in invoke: {str(e)}")
            return f"Error calling AI service: {str(e)}"

    def stream(self, prompt: str, **kwargs):
        try:
            if len(prompt) > 8000:
                prompt = prompt[:8000] + "\n\n[Content truncated due to length]"
                
            messages = [{"role": "user", "content": prompt}]
            
            for chunk_text in self.openrouter_client.chat_completion_stream(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                **kwargs
            ):
                yield chunk_text
                
        except Exception as e:
            yield f"Streaming error: {str(e)}"

class ChatInterface:
    def __init__(self):
        self.openrouter_client = None
        self.rag_system = None
        self.data_manager = None
        self.voice_config_manager = None
        self.session_id = self.get_or_create_session_id()
        self.initialize_session_state()
        self.initialize_data_manager()
        self.initialize_voice_config_manager()

    def get_or_create_session_id(self):
        if 'chat_session_id' not in st.session_state:
            st.session_state.chat_session_id = f"session_{int(time.time() * 1000)}_{hash(str(time.time())) % 10000}"
        return st.session_state.chat_session_id

    def generate_unique_key(self, base_key: str, context: str = ""):
        bot_id = st.session_state.get('current_bot', 'default')
        return f"{base_key}_{context}_{bot_id}_{self.session_id}"

    def initialize_session_state(self):
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        if 'streaming_enabled' not in st.session_state:
            st.session_state.streaming_enabled = True
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = {}
        
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'user_config' not in st.session_state:
            st.session_state.user_config = {}
        if 'bots' not in st.session_state:
            st.session_state.bots = {}
        
        if 'auto_save_enabled' not in st.session_state:
            st.session_state.auto_save_enabled = True
        if 'last_save_time' not in st.session_state:
            st.session_state.last_save_time = None

    def initialize_data_manager(self):
        try:
            if DATA_MANAGER_AVAILABLE:
                if 'data_manager' not in st.session_state:
                    st.session_state.data_manager = DataManager()
                self.data_manager = st.session_state.data_manager
            else:
                st.warning("Data Manager not available. Data will not persist between sessions.")
                self.data_manager = None
        except Exception as e:
            st.error(f"Failed to initialize data manager: {e}")
            self.data_manager = None

    def initialize_voice_config_manager(self):
        try:
            if VOICE_CONFIG_AVAILABLE:
                if 'voice_config_manager' not in st.session_state:
                    if self.data_manager is None:
                        self.initialize_data_manager()
                    st.session_state.voice_config_manager = VoiceConfig(data_manager=self.data_manager)
                self.voice_config_manager = st.session_state.voice_config_manager
            else:
                st.warning("VoiceConfig not available. Voice features will be disabled.")
                self.voice_config_manager = None
        except Exception as e:
            st.error(f"Failed to initialize voice config manager: {e}")
            self.voice_config_manager = None

    def ensure_user_authenticated(self):
        if not st.session_state.get('authenticated', False):
            self.show_authentication_interface()
            return False
        return True

    def show_authentication_interface(self):
        st.markdown("""
        <div style="padding: 2rem; border-radius: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 2rem; text-align: center;">
            <h2 style="margin: 0; color: white;">User Authentication</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Please sign in to save your chat history and settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("authentication_form"):
                st.subheader("Sign In / Sign Up")
                
                username = st.text_input(
                    "Username", 
                    placeholder="Enter your username",
                    help="Your username will be used to save your data across sessions"
                )
                
                email = st.text_input(
                    "Email (Optional)", 
                    placeholder="Enter your email",
                    help="Email is optional but recommended for account recovery"
                )
                
                col_login, col_register = st.columns(2)
                
                with col_login:
                    login_clicked = st.form_submit_button("Sign In", use_container_width=True)
                
                with col_register:
                    register_clicked = st.form_submit_button("Sign Up", use_container_width=True)
                
                if login_clicked or register_clicked:
                    if username.strip():
                        self.authenticate_user(username.strip(), email.strip() if email else None)
                        st.rerun()
                    else:
                        st.error("Please enter a username")

    def authenticate_user(self, username: str, email: str = None):
        try:
            if self.data_manager:
                user_id = self.data_manager.initialize_user_session(username, email)
                st.success(f"Successfully authenticated as {username}")
                
                self.load_user_chat_history(user_id)
                
            else:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_id = f"temp_{hash(username) % 10000}"
                st.warning("Data will not persist between sessions (DataManager unavailable)")
                
        except Exception as e:
            st.error(f"Authentication failed: {e}")

    def load_user_chat_history(self, user_id: str):
        try:
            if self.data_manager:
                st.session_state.chat_history = self.data_manager.load_chat_history(user_id)
                
                if st.session_state.chat_history:
                    bot_count = len(st.session_state.chat_history)
                    total_messages = sum(len(messages) for messages in st.session_state.chat_history.values() if isinstance(messages, list))
                    st.info(f"Loaded chat history: {bot_count} conversations with {total_messages} total messages")
                    
        except Exception as e:
            st.error(f"Failed to load chat history: {e}")

    def save_user_data(self, force_save: bool = False):
        try:
            if not self.data_manager or not st.session_state.get('authenticated'):
                return False
            
            current_time = datetime.now()
            last_save = st.session_state.get('last_save_time')
            
            should_save = force_save or not last_save or (current_time - last_save).seconds > 30
            
            if should_save and st.session_state.get('auto_save_enabled', True):
                user_id = st.session_state.get('user_id')
                if user_id:
                    success = self.data_manager.save_chat_history(user_id, st.session_state.chat_history)
                    
                    if success:
                        st.session_state.last_save_time = current_time
                        return True
                        
        except Exception as e:
            st.error(f"Failed to save user data: {e}")
            
        return False

    def show_chat_interface(self, bot_config: Dict):
        if not self.ensure_user_authenticated():
            return
        
        self.initialize_clients(bot_config)

        if not self.openrouter_client:
            st.error("No OpenRouter API key configured. Please set up your API key in Settings.")
            if st.button("Go to Settings", key=self.generate_unique_key("go_to_settings", "main")):
                st.switch_page("pages/3_Settings.py")
            return

        if not bot_config.get('knowledge_base', {}).get('enabled', False):
            st.info("Knowledge Base disabled - using plain AI mode.")
            self.rag_system = None

        self.render_chat_header(bot_config)

        self.render_chat_controls(bot_config)

        chat_key = st.session_state.get('current_bot')
        if not chat_key:
            st.error("No bot selected")
            return

        self.initialize_bot_chat_history(chat_key, bot_config)

        self.render_chat_messages(chat_key, bot_config)

        self.handle_chat_input(bot_config, chat_key)
        
        self.save_user_data(force_save=False)

    def initialize_bot_chat_history(self, chat_key: str, bot_config: Dict):
        if chat_key not in st.session_state.chat_history:
            welcome_msg = bot_config.get('welcome_message', 'Hello! I can help you with questions based on your uploaded documents.')
            st.session_state.chat_history[chat_key] = [
                {
                    "role": "assistant", 
                    "content": welcome_msg, 
                    "timestamp": datetime.now().isoformat(),
                    "bot_id": chat_key,
                    "user_id": st.session_state.get('user_id')
                }
            ]
            self.save_user_data(force_save=True)

    def initialize_clients(self, bot_config: Dict):
        user_config = bot_config.get('user_config', st.session_state.get('user_config', {}))
        
        if OPENROUTER_AVAILABLE and user_config.get('openrouter_api_key'):
            try:
                self.openrouter_client = OpenRouterClient(user_config['openrouter_api_key'])
            except Exception as e:
                st.error(f"Error initializing OpenRouter client: {str(e)}")

        if bot_config.get('knowledge_base', {}).get('enabled'):
            knowledge_processor = bot_config.get('_knowledge_processor')
            if knowledge_processor and hasattr(knowledge_processor, 'vectorstore') and knowledge_processor.vectorstore:
                self.rag_system = knowledge_processor
                kb_status = bot_config.get('_kb_status', {})
                if kb_status.get('initialized'):
                    doc_count = kb_status.get('document_count', 0)
                    st.success(f"RAG system loaded with {doc_count} documents")
                else:
                    st.info("RAG system loaded but may be empty")
            else:
                try:
                    bot_id = st.session_state.get('current_bot')
                    if bot_id and KNOWLEDGE_PROCESSOR_AVAILABLE:
                        self.rag_system = KnowledgeProcessor()
                        embedding_model = bot_config.get('knowledge_base', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                        embeddings_success = self.rag_system.initialize_embeddings(model_name=embedding_model)
                        if embeddings_success:
                            vectorstore = self.rag_system.setup_vectorstore_with_persistence(bot_id)
                            if vectorstore:
                                test_success, test_msg = self.rag_system.test_vectorstore()
                                if test_success:
                                    st.info("RAG system reinitialized from saved vectorstore")
                                else:
                                    st.warning(f"RAG system loaded but may have issues: {test_msg}")
                            else:
                                st.warning("Could not load vectorstore")
                                self.rag_system = None
                        else:
                            st.error("Could not initialize embeddings for RAG system")
                            self.rag_system = None
                except Exception as e:
                    st.warning(f"RAG system reinitialization failed: {str(e)}")
                    self.rag_system = None

    def render_chat_header(self, bot_config: Dict):
        username = st.session_state.get('username', 'User')
        user_id = st.session_state.get('user_id', 'Unknown')
        last_save = st.session_state.get('last_save_time')
        save_status = f"Last saved: {last_save.strftime('%H:%M:%S')}" if last_save else "Not saved yet"
        
        voice_config = bot_config.get('voice_config', {})
        voice_status = "Voice Enabled" if voice_config.get('enabled', False) else "Text Only"
        
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; color: white;">{bot_config.get('name', 'AI Assistant')} - RAG Mode</h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Document-Enhanced AI Assistant | {voice_status}</p>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.8; font-size: 0.9rem;">{bot_config.get('description', 'AI Assistant with access to your documents')[:150]}...</p>
                </div>
                <div style="text-align: right; opacity: 0.9;">
                    <p style="margin: 0; font-size: 0.9rem;">{username}</p>
                    <p style="margin: 0; font-size: 0.8rem;">{save_status}</p>
                    <p style="margin: 0; font_size: 0.8rem;">Data Persistent</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def render_chat_controls(self, bot_config: Dict):
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1])
        
        with col1:
            st.session_state.debug_mode = st.checkbox(
                "Debug Mode", 
                value=st.session_state.debug_mode,
                key=self.generate_unique_key("debug_mode", "controls")
            )
        
        with col2:
            st.session_state.streaming_enabled = st.checkbox(
                "Streaming", 
                value=st.session_state.streaming_enabled,
                key=self.generate_unique_key("streaming_enabled", "controls")
            )
        
        with col3:
            st.session_state.auto_save_enabled = st.checkbox(
                "Auto-Save", 
                value=st.session_state.auto_save_enabled,
                key=self.generate_unique_key("auto_save_enabled", "controls")
            )
        
        with col4:
            if self.rag_system and self.rag_system.vectorstore:
                try:
                    doc_count = self.rag_system.vectorstore._collection.count()
                    st.success(f"RAG Active ({doc_count} docs)")
                except:
                    st.error("RAG Error")
            else:
                st.error("RAG Inactive")
        
        with col5:
            if st.button(
                "Save Now", 
                key=self.generate_unique_key("manual_save", "controls")
            ):
                with st.spinner("Saving..."):
                    success = self.save_user_data(force_save=True)
                    if success:
                        st.success("Data saved!")
                    else:
                        st.error("Save failed!")
        
        with col6:
            if st.button(
                "Stats", 
                key=self.generate_unique_key("user_stats", "controls")
            ):
                self.show_user_statistics()
        
        with col7:
            if st.button(
                "Clear", 
                key=self.generate_unique_key("clear_chat", "controls"), 
                help="Clear current chat"
            ):
                self.clear_current_chat()

    def show_user_statistics(self):
        try:
            if self.data_manager and st.session_state.get('user_id'):
                user_id = st.session_state.user_id
                stats = self.data_manager.get_user_stats(user_id)
                storage = self.data_manager.get_storage_usage(user_id)
                
                with st.expander("User Statistics", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Bots", stats.get('total_bots', 0))
                        st.metric("Active Conversations", stats.get('total_conversations', 0))
                        st.metric("Total Messages", stats.get('total_messages', 0))
                    
                    with col2:
                        st.metric("KB-Enabled Bots", stats.get('kb_enabled_bots', 0))
                        st.metric("Uploaded Files", stats.get('total_uploaded_files', 0))
                        file_size_mb = stats.get('total_file_size', 0) / (1024 * 1024)
                        st.metric("Files Size", f"{file_size_mb:.1f} MB")
                    
                    with col3:
                        total_storage_mb = storage.get('total_size', 0) / (1024 * 1024)
                        st.metric("Total Storage", f"{total_storage_mb:.1f} MB")
                        st.metric("Last Activity", stats.get('last_activity', 'Unknown')[:10])
                        
                        if st.button(
                            "Cleanup Orphaned Files",
                            key=self.generate_unique_key("cleanup_files", "stats")
                        ):
                            self.data_manager.cleanup_orphaned_files(user_id)
                            st.success("Cleanup completed!")
            else:
                st.info("Statistics not available (not authenticated or DataManager unavailable)")
                
        except Exception as e:
            st.error(f"Error loading statistics: {e}")

    def clear_current_chat(self):
        chat_key = st.session_state.get('current_bot')
        if chat_key and chat_key in st.session_state.chat_history:
            st.session_state.chat_history[chat_key] = []
            
            self.save_user_data(force_save=True)
            
            st.success("Chat cleared!")
            st.rerun()

    def get_response_mode(self, bot_config: Dict) -> str:
        voice_config = bot_config.get('voice_config', {})
        
        if not voice_config.get('enabled', False):
            return 'text'
        
        return voice_config.get('response_mode', 'voice')

    def render_chat_messages(self, chat_key: str, bot_config: Dict):
        messages = st.session_state.chat_history.get(chat_key, [])
        response_mode = self.get_response_mode(bot_config)
        
        for i, message in enumerate(messages):
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
                    
                    if st.session_state.debug_mode:
                        timestamp = message.get('timestamp', 'Unknown')
                        st.caption(f"Time: {timestamp[:19].replace('T', ' ')}")
            else:
                with st.chat_message("assistant"):
                    if response_mode in ['text', 'both']:
                        st.write(message['content'])
                    
                    if message.get('audio_data') and isinstance(message['audio_data'], bytes):
                        audio_format = message.get('audio_format', 'audio/mp3')
                        
                        st.audio(message['audio_data'], format=audio_format)
                    elif message.get('audio_data_b64'):
                         try:
                             decoded_audio = base64.b64decode(message['audio_data_b64'])
                             audio_format = message.get('audio_format', 'audio/mp3')
                             st.audio(decoded_audio, format=audio_format)
                             message['audio_data'] = decoded_audio
                             del message['audio_data_b64']
                         except Exception as e:
                             st.warning(f"Could not decode audio data for playback: {e}")
                             st.caption("Audio playback unavailable.")
                    
                    if response_mode == 'voice' and message.get('audio_data'):
                        st.caption("🎵 Audio response (text hidden)")

                    if st.session_state.debug_mode and i > 0:
                        debug_info = message.get('debug_info', {})
                        timestamp = message.get('timestamp', 'Unknown')
                        
                        with st.expander(f"Debug Info - Message {i+1}", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Timestamp:** {timestamp[:19].replace('T', ' ')}")
                                st.write(f"**Role:** {message['role']}")
                                st.write(f"**Content Length:** {len(message['content'])}")
                                st.write(f"**Model Used:** {debug_info.get('model', 'N/A')}")
                                st.write(f"**Generation Time:** {debug_info.get('generation_time', 'N/A')}")
                                st.write(f"**User ID:** {message.get('user_id', 'N/A')[:12]}...")
                                st.write(f"**Has Audio:** {'Yes' if message.get('audio_data') or message.get('audio_data_b64') else 'No'}")
                            with col2:
                                rag_used = debug_info.get('rag_used', False)
                                st.write(f"**RAG Used:** {'Yes' if rag_used else 'No'}")
                                st.write(f"**Sources Found:** {debug_info.get('sources_count', 0)}")
                                st.write(f"**Response Mode:** {response_mode}")
                                if debug_info.get('sources'):
                                    st.write("**Sources:**")
                                    for j, source in enumerate(debug_info['sources'][:3]):
                                        st.write(f"  {j+1}. {source.get('source', 'Unknown')}")

    def handle_chat_input(self, bot_config: Dict, chat_key: str):
        if user_input := st.chat_input("Ask me anything about your documents..."):
            user_message = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "bot_id": chat_key,
                "user_id": st.session_state.get('user_id'),
                "message_id": f"msg_{int(time.time() * 1000)}"
            }
            st.session_state.chat_history[chat_key].append(user_message)

            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                response_mode = self.get_response_mode(bot_config)
                assistant_response_text = ""
                assistant_audio_data = None
                response_debug_info = {}
                audio_format = None

                if self.rag_system:
                    with st.spinner("Searching documents..."):
                        response_data = self.generate_rag_response_safe(user_input, bot_config, chat_key)
                        assistant_response_text = response_data.get('content', 'Error generating response')
                        response_debug_info = response_data.get('debug_info', {})
                else:
                    with st.spinner("Generating response..."):
                        llm = OpenRouterLLM(self.openrouter_client, model=bot_config.get('model'))
                        assistant_response_text = llm.invoke(user_input)
                        response_debug_info = {"rag_used": False, "model": bot_config.get('model')}

                if response_mode in ['text', 'both']:
                    st.write(assistant_response_text)
                
                voice_config = bot_config.get('voice_config', {})
                if voice_config.get('enabled', False) and response_mode in ['voice', 'both']:
                    assistant_audio_data, audio_format = self.generate_voice_response(voice_config, assistant_response_text)
                    
                    if assistant_audio_data:
                        st.audio(assistant_audio_data, format=audio_format)
                    elif response_mode == 'voice':
                        st.warning("Voice generation failed. Showing text response:")
                        st.write(assistant_response_text)
                
                if response_mode == 'voice' and assistant_audio_data:
                    st.caption("Audio response generated (text hidden in voice-only mode)")

                if st.session_state.debug_mode:
                    with st.expander("Debug Information", expanded=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Basic Info:**")
                            st.write(f"Success: {response_data.get('success', 'Unknown') if self.rag_system else 'N/A'}")
                            st.write(f"Model: {response_debug_info.get('model', 'Unknown')}")
                            st.write(f"Generation Time: {response_debug_info.get('generation_time', 'Unknown')}")
                            st.write(f"Total Documents: {response_debug_info.get('total_documents', 'Unknown')}")
                            st.write(f"Sources Found: {response_debug_info.get('sources_count', 0)}")
                            st.write(f"Response Mode: {response_mode}")
                            st.write(f"Voice Enabled: {voice_config.get('enabled', False)}")
                            st.write(f"Audio Format: {audio_format or 'N/A'}")
                        
                        with col2:
                            st.write("**Sources:**")
                            sources = response_debug_info.get('sources', [])
                            if sources:
                                for i, source in enumerate(sources):
                                    st.write(f"{i+1}. {source.get('source', 'N/A')} (Score: {source.get('score', 'N/A')})")
                            else:
                                st.write("No sources found")
                        
                        st.write("**Debug Steps:**")
                        debug_steps = response_debug_info.get('debug_steps', [])
                        for step in debug_steps:
                            st.text(step)
                        
                        if voice_config.get('enabled'):
                            st.write("**Voice Debug:**")
                            st.write(f"Provider: {voice_config.get('provider', 'N/A')}")
                            st.write(f"Voice ID: {voice_config.get('voice', 'N/A')}")
                            st.write(f"Audio Generated: {'Yes' if assistant_audio_data else 'No'}")

            assistant_message = {
                "role": "assistant",
                "content": assistant_response_text,
                "timestamp": datetime.now().isoformat(),
                "bot_id": chat_key,
                "user_id": st.session_state.get('user_id'),
                "message_id": f"msg_{int(time.time() * 1000)}",
                "debug_info": response_debug_info,
                "audio_data": assistant_audio_data,
                "response_mode": response_mode,
                "audio_format": audio_format
            }
            st.session_state.chat_history[chat_key].append(assistant_message)
            
            self.save_user_data(force_save=True)

    def generate_voice_response(self, voice_config: Dict, text: str) -> tuple[Optional[bytes], Optional[str]]:
        if not voice_config.get('enabled', False) or not text.strip():
            return None, None
        
        if not self.voice_config_manager:
            st.warning("Voice configuration manager not available")
            return None, None
        
        provider = voice_config.get('provider', 'murf')
        
        try:
            audio_data, audio_format = self.voice_config_manager.generate_voice_preview(voice_config, text)
            return audio_data, audio_format
                
        except Exception as e:
            st.error(f"Voice generation failed: {str(e)}")
            return None, None

    def generate_rag_response_safe(self, user_input: str, bot_config: Dict, chat_key: str) -> Dict:
        start_time = time.time()
        debug_info = {
            'model': bot_config.get('model', 'unknown'),
            'rag_used': True,
            'debug_steps': [],
            'errors': []
        }
        
        try:
            if not user_input or not user_input.strip():
                debug_info['debug_steps'].append("Input validation: Empty user input")
                return {
                    'success': False,
                    'content': 'Please enter a question.',
                    'debug_info': debug_info
                }
            
            if len(user_input) > 5000:
                user_input = user_input[:5000] + "..."
                debug_info['debug_steps'].append("Input truncated due to length")
            
            if not self.rag_system or not self.rag_system.vectorstore:
                debug_info['errors'].append("RAG system not available")
                debug_info['debug_steps'].append("RAG system check: Not initialized or vectorstore missing")
                return {
                    'success': False,
                    'content': 'Knowledge base system is not available. Please check your configuration.',
                    'debug_info': debug_info
                }
            
            search_results = []
            try:
                debug_info['debug_steps'].append("Attempting enhanced similarity search...")
                search_results = self.rag_system.improve_similarity_search(user_input, k=4)
                
                if search_results:
                    debug_info['debug_steps'].append(f"Enhanced search found {len(search_results)} documents.")
                else:
                    debug_info['debug_steps'].append("Enhanced search found no documents. Trying keyword fallback...")
                    keywords = user_input.lower().split()[:3]
                    for keyword in keywords:
                        if len(keyword) > 3:
                            try:
                                kw_results = self.rag_system.similarity_search(keyword, k=2)
                                if kw_results:
                                    search_results.extend([(doc, 0.5) for doc in kw_results])
                                    break
                            except Exception as kw_error:
                                debug_info['errors'].append(f"Keyword search for '{keyword}' failed: {kw_error}")
                                continue
            except Exception as search_error:
                debug_info['errors'].append(f"Overall search failed: {search_error}")
                debug_info['debug_steps'].append(f"Search failed: {search_error}")
            
            if not search_results:
                debug_info['errors'].append("No relevant documents found after all search attempts.")
                debug_info['debug_steps'].append("No documents found for RAG.")
                return {
                    'success': False,
                    'content': 'I could not find relevant information in your documents. Try rephrasing your question or check if documents were uploaded correctly.',
                    'debug_info': debug_info
                }
            
            context = ""
            try:
                docs_only = [doc for doc, score in search_results[:4]]
                context = self.rag_system.format_docs(docs_only)
                debug_info['debug_steps'].append(f"Context formatted from {len(docs_only)} documents.")
                
                if len(context) > 3000:
                    context = context[:3000] + "\n\n[Context truncated...]"
                    debug_info['debug_steps'].append("Context truncated due to length.")
                    
            except Exception as format_error:
                debug_info['errors'].append(f"Context formatting failed: {format_error}")
                debug_info['debug_steps'].append(f"Context formatting error: {format_error}")
                return {
                    'success': False,
                    'content': 'Error processing document content.',
                    'debug_info': debug_info
                }
            
            llm = None
            try:
                llm = OpenRouterLLM(
                    openrouter_client=self.openrouter_client,
                    model=bot_config.get('model', 'meta-llama/llama-3.2-3b-instruct:free'),
                    temperature=min(bot_config.get('temperature', 0.7), 1.0),
                )
                debug_info['debug_steps'].append(f"LLM instance created with model: {llm.model}")
            except Exception as llm_error:
                debug_info['errors'].append(f"LLM initialization failed: {llm_error}")
                debug_info['debug_steps'].append(f"LLM initialization error: {llm_error}")
                return {
                    'success': False,
                    'content': 'Error initializing AI model.',
                    'debug_info': debug_info
                }
            
            prompt = ""
            try:
                chat_history = st.session_state.chat_history.get(chat_key, [])
                recent_history = ""
                for msg in chat_history[-3:-1]:
                    if msg['role'] == 'user':
                        recent_history += f":User  {msg['content'][:200]}...\n" if len(msg['content']) > 200 else f":User  {msg['content']}\n"
                    elif msg['role'] == 'assistant':
                        recent_history += f"Assistant: {msg['content'][:200]}...\n" if len(msg['content']) > 200 else f"Assistant: {msg['content']}\n"
                
                if recent_history:
                    debug_info['debug_steps'].append("Including recent chat history in prompt.")
                    prompt = f"""You are a helpful AI assistant. Use the following context from documents and conversation history to answer the user's question.

    Previous conversation:
    {recent_history}

    Context from documents:
    {context}

    Current question: {user_input}

    Instructions:
    - Answer based primarily on the provided context.
    - If the information is not in the context, state clearly that you cannot find the answer in the provided documents. Do NOT make up information.
    - Maintain conversation continuity.
    - Be concise but comprehensive.

    Answer:"""
                else:
                    debug_info['debug_steps'].append("No recent chat history to include in prompt.")
                    prompt = f"""Use the following context to answer the question. If you cannot answer based on the context, say so clearly that the information is not in the provided documents. Do NOT make up information.

    Context:
    {context}

    Question: {user_input}

    Instructions:
    - Use the context to answer the question.
    - If information is missing, state clearly that you cannot find the answer in the provided documents.
    - Be concise and helpful.

    Answer:"""
                
                if len(prompt) > 6000:
                    prompt = prompt[:6000] + "\n\nAnswer based on available information:"
                    debug_info['debug_steps'].append("Final prompt truncated due to length.")
                    
            except Exception as prompt_error:
                debug_info['errors'].append(f"Prompt building failed: {prompt_error}")
                debug_info['debug_steps'].append(f"Prompt building error: {prompt_error}. Using fallback prompt.")
                prompt = f"Question: {user_input}\n\nAnswer based on the uploaded documents:\nContext:\n{context}\nAnswer:"
            
            response = ""
            try:
                response = llm.invoke_with_retry(prompt)
                debug_info['debug_steps'].append("Response generated successfully from LLM.")
            except Exception as response_error:
                debug_info['errors'].append(f"Response generation failed: {response_error}")
                debug_info['debug_steps'].append(f"LLM response generation error: {response_error}")
                return {
                    'success': False,
                    'content': 'I apologize, but I encountered an error generating a response. Please try again.',
                    'debug_info': debug_info
                }
            
            if not response or len(response.strip()) < 10:
                debug_info['debug_steps'].append("Response validation: Incomplete or empty response from LLM.")
                return {
                    'success': False,
                    'content': 'I received an incomplete response. Please try rephrasing your question.',
                    'debug_info': debug_info
                }
            
            generation_time = time.time() - start_time
            debug_info['generation_time'] = f"{generation_time:.2f}s"
            debug_info['sources_count'] = len(search_results)
            debug_info['sources'] = []
            
            return {
                'success': True,
                'content': response,
                'debug_info': debug_info
            }
        except Exception as critical_error:
            debug_info['errors'].append(f"Critical error: {critical_error}")
            generation_time = time.time() - start_time
            debug_info['generation_time'] = f"{generation_time:.2f}s"
            
    def export_chat_history(self, chat_key: str = None):
        try:
            if not st.session_state.get('authenticated'):
                st.error("Please authenticate first")
                return None
            
            user_id = st.session_state.get('user_id')
            username = st.session_state.get('username')
            
            exported_chat_history = {}
            for bot_id, messages in st.session_state.chat_history.items():
                exported_chat_history[bot_id] = []
                for message in messages:
                    msg_copy = message.copy()
                    if msg_copy.get('audio_data') and isinstance(msg_copy['audio_data'], bytes):
                        msg_copy['audio_data_b64'] = base64.b64encode(msg_copy.pop('audio_data')).decode('utf-8')
                    exported_chat_history[bot_id].append(msg_copy)

            if chat_key:
                messages_to_export = exported_chat_history.get(chat_key, [])
                export_data = {
                    'user_id': user_id,
                    'username': username,
                    'bot_id': chat_key,
                    'messages': messages_to_export,
                    'export_date': datetime.now().isoformat(),
                    'message_count': len(messages_to_export)
                }
                filename = f"chat_{chat_key[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            else:
                export_data = {
                    'user_id': user_id,
                    'username': username,
                    'all_chats': exported_chat_history,
                    'export_date': datetime.now().isoformat(),
                    'total_conversations': len(exported_chat_history),
                    'total_messages': sum(len(msgs) for msgs in exported_chat_history.values() if isinstance(msgs, list))
                }
                filename = f"all_chats_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                label="Download Chat Export",
                data=json_str,
                file_name=filename,
                mime="application/json",
                key=self.generate_unique_key(f"export_{chat_key or 'all'}", "export")
            )
            
            return export_data
            
        except Exception as e:
            st.error(f"Error exporting chat history: {e}")
            return None

    def show_data_management_interface(self):
        if not st.session_state.get('authenticated'):
            st.error("Please authenticate first to manage your data")
            return
        
        st.subheader("Data Management")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Backup & Export", "Statistics", "Cleanup", "Settings"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Export Data**")
                
                if st.button(
                    "Export All User Data",
                    key=self.generate_unique_key("export_all_data", "management")
                ):
                    if self.data_manager:
                        export_file = self.data_manager.export_user_data(st.session_state.user_id)
                        if export_file:
                            with open(export_file, 'r') as f:
                                data = f.read()
                            st.download_button(
                                "Download Full Backup",
                                data,
                                file_name=f"backup_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                key=self.generate_unique_key("download_full_backup", "management")
                            )
                
                current_chat = st.session_state.get('current_bot')
                if current_chat and st.button(
                    "Export Current Chat",
                    key=self.generate_unique_key("export_current_chat", "management")
                ):
                    self.export_chat_history(current_chat)
                
                if st.button(
                    "Export All Chats",
                    key=self.generate_unique_key("export_all_chats", "management")
                ):
                    self.export_chat_history()
            
            with col2:
                st.write("**Import Data**")
                
                uploaded_backup = st.file_uploader(
                    "Choose backup file", 
                    type=['json'],
                    key=self.generate_unique_key("file_uploader_backup", "management")
                )
                if uploaded_backup and st.button(
                    "Import Backup",
                    key=self.generate_unique_key("import_backup", "management")
                ):
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                            json.dump(json.load(uploaded_backup), tmp)
                            tmp_path = tmp.name
                        
                        if self.data_manager:
                            success = self.data_manager.import_user_data(st.session_state.user_id, tmp_path)
                            if success:
                                st.success("Data imported successfully! Please refresh the page.")
                        
                        import os
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                            st.error(f"Import failed: {e}")
        
        with tab2:
            self.show_user_statistics()
        
        with tab3:
            st.write("**Cleanup Tools**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(
                    "Clean Orphaned Files",
                    key=self.generate_unique_key("cleanup_orphaned", "management")
                ):
                    if self.data_manager:
                        self.data_manager.cleanup_orphaned_files(st.session_state.user_id)
                        st.success("Cleanup completed!")
                
                if st.button(
                    "Clear All Chat History",
                    key=self.generate_unique_key("clear_all_chats", "management")
                ):
                    if st.button(
                        "Confirm Delete All Chats",
                        key=self.generate_unique_key("confirm_delete_all", "management")
                    ):
                        st.session_state.chat_history = {}
                        self.save_user_data(force_save=True)
                        st.success("All chat history cleared!")
                        st.rerun()
            
            with col2:
                if self.data_manager:
                    storage = self.data_manager.get_storage_usage(st.session_state.user_id)
                    total_mb = storage.get('total_size', 0) / (1024 * 1024)
                    st.metric("Current Storage Usage", f"{total_mb:.2f} MB")
                    
                    st.write("**Storage Breakdown:**")
                    for key, size in storage.items():
                        if key != 'total_size':
                            size_mb = size / (1024 * 1024)
                            st.write(f"- {key.replace('_', ' ').title()}: {size_mb:.2f} MB")
        
        with tab4:
            st.write("**Data Settings**")
            
            auto_save = st.checkbox(
                "Enable Auto-Save", 
                value=st.session_state.get('auto_save_enabled', True),
                help="Automatically save data every 30 seconds",
                key=self.generate_unique_key("auto_save_settings", "management")
            )
            st.session_state.auto_save_enabled = auto_save
            
            if st.button(
                "Save Settings",
                key=self.generate_unique_key("save_settings", "management")
            ):
                self.save_user_data(force_save=True)
                st.success("Settings saved!")
            
            st.write("**Account Actions**")
            
            if st.button(
                "Logout",
                key=self.generate_unique_key("logout", "management")
            ):
                self.save_user_data(force_save=True)
                
                keys_to_clear = ['authenticated', 'user_id', 'username', 'user_config', 'chat_history', 'bots']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("Logged out successfully!")
                st.rerun()

    def on_session_end(self):
        try:
            if st.session_state.get('authenticated') and self.data_manager:
                self.save_user_data(force_save=True)
                st.info("Data saved before session end")
        except Exception as e:
            st.warning(f"Failed to save data on session end: {e}")

def ensure_authenticated_session():
    if not st.session_state.get('authenticated', False):
        return False
    return True

def auto_save_handler():
    try:
        if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
            chat_interface = ChatInterface()
            chat_interface.save_user_data(force_save=False)
    except Exception as e:
        pass

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = datetime.now()

st.session_state.last_activity = datetime.now()

if 'interaction_count' not in st.session_state:
    st.session_state.interaction_count = 0
st.session_state.interaction_count += 1

if st.session_state.interaction_count % 10 == 0:
    auto_save_handler()
