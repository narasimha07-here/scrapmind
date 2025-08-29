__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import json
import uuid
import os
from datetime import datetime
from typing import Dict, List, Optional
from components.data_manager import DataManager

try:
    from utils.openrouter_client import OpenRouterClient
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False

try:
    from utils.langchain_rag import ChromaRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from components.knowledge_processor import KnowledgeProcessor
    KNOWLEDGE_PROCESSOR_AVAILABLE = True
except ImportError:
    KNOWLEDGE_PROCESSOR_AVAILABLE = False

try:
    from components.voice_config import VoiceConfig
    VOICE_CONFIG_AVAILABLE = True
except ImportError:
    VOICE_CONFIG_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class BotCreator:
    def __init__(self):
        if KNOWLEDGE_PROCESSOR_AVAILABLE:
            self.knowledge_processor = None
        else:
            self.knowledge_processor = None
        
        self.openrouter_client = None
        
        if VOICE_CONFIG_AVAILABLE:
            self.voice_config_manager = None
        else:
            self.voice_config_manager = None
        
        self.initialize_session_state()
        
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        self.data_manager = st.session_state.get('data_manager')
        
        if OPENROUTER_AVAILABLE and st.session_state.get('user_config', {}).get('openrouter_api_key'):
            try:
                self.openrouter_client = OpenRouterClient(
                    st.session_state.user_config['openrouter_api_key']
                )
            except Exception as e:
                st.error(f"Error initializing OpenRouter client: {str(e)}")
        
        if VOICE_CONFIG_AVAILABLE:
            try:
                self.voice_config_manager = VoiceConfig(data_manager=self.data_manager)
            except Exception as e:
                st.warning(f"Voice config manager initialization failed: {str(e)}")

    def get_fallback_models(self) -> List[Dict]:
        return [
            {"id": "meta-llama/llama-3.2-3b-instruct:free", "name": "Llama 3.2 3B (Free)", "description": "Free fast model", "is_free": True},
            {"id": "meta-llama/llama-3.2-1b-instruct:free", "name": "Llama 3.2 1B (Free)", "description": "Free lightweight model", "is_free": True},
            {"id": "huggingfaceh4/zephyr-7b-beta:free", "name": "Zephyr 7B Beta (Free)", "description": "Free conversational model", "is_free": True},
            {"id": "openchat/openchat-7b:free", "name": "OpenChat 7B (Free)", "description": "Free chat model", "is_free": True},
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "description": "Fast and efficient", "is_free": False},
            {"id": "openai/gpt-4", "name": "GPT-4", "description": "Most capable model", "is_free": False},
        ]

    def format_file_size(self, size_bytes: int) -> str:
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        return f"{size_bytes:.1f} {size_names[i]}"

    def initialize_session_state(self):
        if 'user_config' not in st.session_state:
            st.session_state.user_config = {
                'openrouter_api_key': '',
                'openai_api_key': '',
                'default_model': 'meta-llama/llama-3.2-3b-instruct:free',
                'default_embedding': 'sentence-transformers/all-MiniLM-L6-v2',
                'default_chunking': 'recursive_character',
                'user_id': str(uuid.uuid4())
            }
        
        if 'bots' not in st.session_state:
            st.session_state.bots = {}
        
        if 'current_bot' not in st.session_state:
            st.session_state.current_bot = None
        
        if 'available_models' not in st.session_state:
            st.session_state.available_models = []

    def auto_save_data(self):
        if self.data_manager and st.session_state.get('user_id'):
            try:
                self.data_manager.save_all_user_data(st.session_state.user_id)
            except Exception as e:
                st.warning(f"Auto-save failed: {str(e)}")

    def start_new_bot_creation(self):
        new_bot_id = str(uuid.uuid4())
        st.session_state.current_bot = new_bot_id
        
        st.session_state.force_new_bot = True
        if 'is_editing_mode' in st.session_state:
            del st.session_state['is_editing_mode']
        if 'editing_bot_config' in st.session_state:
            del st.session_state['editing_bot_config']
        if 'editing_bot_id' in st.session_state:
            del st.session_state['editing_bot_id']
        
        keys_to_clear = [
            "bot_name_input",
            "bot_description_input",
            "bot_welcome_input",
            "bot_personality_select",
            "voice_enabled_checkbox",
            "response_mode_select",
            "voice_provider_select",
            "voice_api_key_input",
            "voice_select",
            "voice_test_btn",
            "bot_model_select",
            "bot_temperature_slider",
            "bot_max_tokens_input",
            "kb_enabled_checkbox",
            "kb_embedding_select",
            "kb_data_source_select",
            f"kb_file_uploader_{new_bot_id}",
            "kb_manual_text_area",
            "kb_urls_text_area",
            "kb_chunking_select",
            "kb_chunk_size_input",
            "kb_overlap_input",
            "kb_max_results_input",
            "kb_similarity_threshold",
            "adv_context_window_input",
            "adv_response_format_select",
            "adv_error_handling_select",
            "adv_system_message_input",
            "adv_custom_instructions_input",
            "faq_question_input",
            "faq_answer_input",
        ]
        
        for key in list(st.session_state.keys()):
            if any(pattern in key for pattern in [
                "kb_file_uploader_",
                "clear_files_",
                "edit_q_",
                "edit_a_",
                "del_faq_",
                "add_faq_btn",
                "save_draft_btn",
                "save_test_btn",
                "deploy_bot_btn",
                "generate_fastapi_btn"
            ]):
                del st.session_state[key]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.rerun()

    def show_bot_creation_wizard(self):
        st.title("🤖 Create Your AI Chatbot")
        st.markdown("Build a powerful AI chatbot in just a few steps - no coding required!")
        
        if st.session_state.get('create_new_bot_requested'):
            st.session_state.create_new_bot_requested = False
            self.start_new_bot_creation()
            return
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🆕 Start New Bot", key="start_new_bot_btn", type="secondary"):
                self.start_new_bot_creation()
                return
        
        with col2:
            if st.button("📂 Load Existing", key="load_existing_btn", type="secondary"):
                self.show_bot_selector()
                return
        
        self.show_progress_indicator()
        
        current_bot_id = st.session_state.get('current_bot')
        is_editing = (current_bot_id and
                     current_bot_id in st.session_state.get('bots', {}) and
                     not st.session_state.get('force_new_bot', False))
        
        if is_editing:
            bot_config = st.session_state.bots[current_bot_id].copy()
            st.info(f"📝 Editing: {bot_config.get('name', 'Unnamed Bot')}")
            
            if st.button("🔄 Start Fresh New Bot", key="start_fresh_from_edit", type="secondary"):
                st.session_state.force_new_bot = True
                st.session_state.create_new_bot_requested = True
                st.rerun()
        else:
            if not current_bot_id:
                current_bot_id = str(uuid.uuid4())
                st.session_state.current_bot = current_bot_id
            
            bot_config = self.create_default_bot_config()
            st.success(f"✨ Creating new chatbot (ID: {current_bot_id[:8]}...)")
        
        if 'force_new_bot' in st.session_state:
            del st.session_state['force_new_bot']
        
        with st.container():
            self.configure_basic_info(bot_config)
            st.markdown("---")
            
            self.configure_voice_settings(bot_config)
            st.markdown("---")
            
            self.configure_ai_model(bot_config)
            st.markdown("---")
            
            self.configure_knowledge_base(bot_config)
            st.markdown("---")
            
            self.configure_advanced_settings(bot_config)
            st.markdown("---")
            
            self.show_save_deploy_options(bot_config, is_editing)
            st.markdown("---")
            
            self.show_fastapi_generation(bot_config)

    def show_bot_selector(self):
        st.title("📂 Select Bot to Edit")
        
        if not st.session_state.get('bots'):
            st.info("No bots created yet. Start by creating a new bot!")
            if st.button("🆕 Create First Bot", type="primary"):
                self.start_new_bot_creation()
                return
        
        st.markdown("### Your Bots")
        for bot_id, bot_config in st.session_state.bots.items():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{bot_config.get('name', 'Unnamed Bot')}**")
                    st.caption(bot_config.get('description', 'No description')[:100] + "...")
                
                with col2:
                    status = bot_config.get('status', 'draft')
                    status_colors = {
                        'draft': '🟠 Draft',
                        'testing': '🔵 Testing',
                        'active': '🟢 Active'
                    }
                    st.markdown(status_colors.get(status, f'⚪ {status.title()}'))
                
                with col3:
                    updated = bot_config.get('updated_at', '')
                    if updated:
                        try:
                            dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                            st.caption(f"Updated: {dt.strftime('%m/%d/%Y')}")
                        except:
                            st.caption("Updated: Unknown")
                
                with col4:
                    if st.button("✏️ Edit", key=f"edit_bot_{bot_id}", type="primary"):
                        st.session_state.current_bot = bot_id
                        st.session_state.is_editing_mode = True
                        st.rerun()
        
        st.markdown("---")
        
        if st.button("⬅️ Back to Creator", key="back_to_creator"):
            if 'is_editing_mode' in st.session_state:
                del st.session_state['is_editing_mode']
            st.rerun()

    def show_progress_indicator(self):
        progress_steps = [
            "Basic Info",
            "Voice Config",
            "AI Model",
            "Knowledge Base",
            "Advanced Settings",
            "Deploy",
            "FastAPI"
        ]
        
        cols = st.columns(len(progress_steps))
        for i, (col, step) in enumerate(zip(cols, progress_steps)):
            with col:
                if i < 5:
                    st.markdown(f"✅ **{step}**")
                else:
                    st.markdown(f"⭐ {step}")

    def create_default_bot_config(self) -> Dict:
        return {
            'name': '',
            'description': '',
            'welcome_message': 'Hello! I am your AI assistant. How can I help you today?',
            'model': st.session_state.user_config.get('default_model', 'meta-llama/llama-3.2-3b-instruct:free'),
            'temperature': 0.7,
            'max_tokens': 1000,
            'personality': 'Professional',
            'voice_config': {
                'enabled': False,
                'response_mode': 'voice',
                'provider': 'murf',
                'api_key': '',
                'voice': '',
                'language': 'en',
                'speed': 1.0,
                'volume': 0.8,
                'pitch': 0
            },
            'knowledge_base': {
                'enabled': False,
                'data_source': 'Upload Documents',
                'chunking_strategy': st.session_state.user_config.get('default_chunking', 'recursive_character'),
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'embedding_model': st.session_state.user_config.get('default_embedding', 'sentence-transformers/all-MiniLM-L6-v2'),
                'similarity_threshold': 0.7,
                'max_results': 4,
                'uploaded_files': [],
                'file_names': [],
                'file_metadata': []
            },
            'advanced_settings': {
                'context_window': 5,
                'response_format': 'conversational',
                'error_handling': 'graceful',
                'system_message': '',
                'custom_instructions': ''
            },
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'draft',
            'user_id': st.session_state.get('user_id', 'anonymous')
        }

    def configure_basic_info(self, bot_config: Dict):
        st.markdown("### 📝 Step 1: Basic Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bot_config['name'] = st.text_input(
                "Bot Name *",
                value=bot_config.get('name', ''),
                placeholder="My AI Assistant",
                key="bot_name_input",
                help="Give your bot a memorable name"
            )
            
            bot_config['description'] = st.text_area(
                "Description",
                value=bot_config.get('description', ''),
                placeholder="Describe what your bot does and how it helps users...",
                key="bot_description_input",
                help="This helps users understand your bot's purpose"
            )
        
        with col2:
            bot_config['welcome_message'] = st.text_area(
                "Welcome Message",
                value=bot_config.get('welcome_message', 'Hello! I am your AI assistant. How can I help you today?'),
                key="bot_welcome_input",
                help="First message users will see when they start chatting"
            )
            
            personalities = [
                "Professional", "Friendly", "Casual", "Technical",
                "Creative", "Helpful", "Witty", "Empathetic", "Direct"
            ]
            
            current_personality = bot_config.get('personality', 'Professional')
            if current_personality not in personalities:
                current_personality = personalities[0]
            
            selected_personality = st.selectbox(
                "Personality",
                personalities,
                index=personalities.index(current_personality),
                key="bot_personality_select",
                help="Choose how your bot should communicate"
            )
            
            bot_config['personality'] = selected_personality

    def configure_voice_settings(self, bot_config: Dict):
        st.markdown("### 🎙️ Step 2: Voice Configuration")
        st.markdown("Configure voice responses to make your bot more engaging!")
        
        if 'voice_config' not in bot_config:
            bot_config['voice_config'] = {
                'enabled': False,
                'response_mode': 'voice',
                'provider': 'murf',
                'api_key': '',
                'voice': '',
                'language': 'en',
                'speed': 1.0,
                'volume': 0.8,
                'pitch': 0
            }
        
        voice_config = bot_config['voice_config']
        
        voice_enabled = st.checkbox(
            "🎙️ Enable Voice Responses",
            value=voice_config.get('enabled', False),
            key="voice_enabled_checkbox",
            help="Generate audio responses for bot messages"
        )
        
        voice_config['enabled'] = voice_enabled
        
        if not voice_enabled:
            st.info("💡 Voice responses are disabled. Your bot will use text-only responses.")
            return
        
        st.markdown("#### 🎯 Response Mode")
        response_modes = {
            'voice': '🎙️ Voice Only',
            'both': '🎙️📝 Voice + Text',
            'text': '📝 Text Only'
        }
        
        current_mode = voice_config.get('response_mode', 'voice')
        if current_mode not in response_modes:
            current_mode = 'voice'
        
        selected_mode = st.selectbox(
            "Choose Response Mode",
            list(response_modes.keys()),
            index=list(response_modes.keys()).index(current_mode),
            format_func=lambda x: response_modes[x],
            key="response_mode_select",
            help="Choose how your bot should respond to users"
        )
        
        voice_config['response_mode'] = selected_mode
        
        mode_descriptions = {
            'voice': "🎙️ **Voice Only**: Bot will respond with audio only. Users will hear responses but won't see text.",
            'both': "🎙️📝 **Voice + Text**: Bot will provide both audio and text responses. Best for accessibility.",
            'text': "📝 **Text Only**: Traditional text-based responses. Voice is enabled but not used for responses."
        }
        
        st.info(mode_descriptions[selected_mode])
        
        if selected_mode in ['voice', 'both']:
            st.markdown("#### 🎵 Voice Provider")
            voice_providers = {
                'murf': 'Murf AI (Recommended)',
                'openai': 'OpenAI TTS',
                'elevenlabs': 'ElevenLabs',
                'google': 'Google Text-to-Speech',
                'azure': 'Azure Speech Services',
                'amazon': 'Amazon Polly'
            }
            
            current_provider = voice_config.get('provider', 'murf')
            if current_provider not in voice_providers:
                current_provider = 'murf'
            
            selected_provider = st.selectbox(
                "Choose Voice Provider",
                list(voice_providers.keys()),
                index=list(voice_providers.keys()).index(current_provider),
                format_func=lambda x: voice_providers[x],
                key="voice_provider_select",
                help="Select the text-to-speech service provider"
            )
            
            voice_config['provider'] = selected_provider
            
            col1, col2 = st.columns(2)
            
            with col1:
                api_key_placeholder = f"Enter your {voice_providers[selected_provider]} API key"
                if selected_provider == 'google':
                    api_key_placeholder = "Google TTS is free - no API key required"
                
                api_key = st.text_input(
                    f"{voice_providers[selected_provider]} API Key",
                    value=voice_config.get('api_key', ''),
                    type="password" if selected_provider != 'google' else "default",
                    placeholder=api_key_placeholder,
                    key="voice_api_key_input",
                    disabled=selected_provider == 'google',
                    help=f"API key for {voice_providers[selected_provider]}"
                )
                
                voice_config['api_key'] = api_key
            
            with col2:
                if selected_provider != 'google':
                    test_key = f"test_api_key_{selected_provider}"
                    if st.button(f"🧪 Test {voice_providers[selected_provider]} API", key=test_key):
                        self.test_voice_api_key(selected_provider, api_key)
            
            if api_key or selected_provider == 'google':
                self.configure_voice_selection(voice_config, selected_provider, api_key)
            
            self.configure_voice_parameters(voice_config)

    def test_voice_api_key(self, provider: str, api_key: str):
        if not api_key:
            st.error("❌ Please enter an API key to test")
            return
        
        with st.spinner(f"🧪 Testing {provider} API key..."):
            try:
                if provider == 'murf' and REQUESTS_AVAILABLE:
                    success, message = self.test_murf_api_key(api_key)
                elif provider == 'openai':
                    success, message = self.test_openai_api_key(api_key)
                elif provider == 'elevenlabs':
                    success, message = self.test_elevenlabs_api_key(api_key)
                elif provider == 'azure':
                    success, message = self.test_azure_api_key(api_key)
                elif provider == 'amazon':
                    success, message = self.test_amazon_api_key(api_key)
                else:
                    success, message = False, "API testing not implemented for this provider"
                
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"❌ API test failed: {str(e)}")

    def test_murf_api_key(self, api_key: str) -> tuple[bool, str]:
        try:
            headers = {
                "accept": "application/json",
                "api-key": api_key
            }
            
            response = requests.get("https://api.murf.ai/v1/speech/voices", headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    voices_data = response.json()
                    if isinstance(voices_data, list):
                        voice_count = len(voices_data)
                        return True, f"API key valid! Found {voice_count} voices available"
                    elif isinstance(voices_data, dict) and 'voices' in voices_data:
                        voice_count = len(voices_data['voices'])
                        return True, f"API key valid! Found {voice_count} voices available"
                    else:
                        voice_count = len(voices_data) if hasattr(voices_data, '__len__') else 0
                        return True, f"API key valid! Found {voice_count} voices available"
                except (ValueError, TypeError) as e:
                    return False, f"Invalid response format from Murf API: {str(e)}"
            elif response.status_code == 401:
                return False, "Invalid API key or unauthorized access"
            elif response.status_code == 403:
                return False, "API key does not have permission to access voices"
            else:
                return False, f"API error: HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            return False, "Request timeout - please try again"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Murf API - check your internet connection"
        except requests.exceptions.RequestException as e:
            return False, f"Network error: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def test_openai_api_key(self, api_key: str) -> tuple[bool, str]:
        return True, "OpenAI API key format appears valid (actual test not implemented)"

    def test_elevenlabs_api_key(self, api_key: str) -> tuple[bool, str]:
        return True, "ElevenLabs API key format appears valid (actual test not implemented)"

    def test_azure_api_key(self, api_key: str) -> tuple[bool, str]:
        return True, "Azure API key format appears valid (actual test not implemented)"

    def test_amazon_api_key(self, api_key: str) -> tuple[bool, str]:
        return True, "Amazon API key format appears valid (actual test not implemented)"

    def configure_voice_selection(self, voice_config: Dict, provider: str, api_key: str):
        st.markdown("#### 🗣️ Voice Selection")
        
        if provider == 'murf':
            self.configure_murf_voices(voice_config, api_key)
        elif provider == 'openai':
            self.configure_openai_voices(voice_config)
        elif provider == 'elevenlabs':
            self.configure_elevenlabs_voices(voice_config)
        elif provider == 'google':
            self.configure_google_voices(voice_config)
        else:
            st.info(f"Voice selection for {provider} will be available soon")

    def configure_murf_voices(self, voice_config: Dict, api_key: str):
        if not api_key:
            st.warning("Please enter your Murf AI API key to see available voices")
            return
        
        try:
            with st.spinner("Loading Murf voices..."):
                voices = self.fetch_murf_voices(api_key)
                
                if not voices:
                    st.error("Failed to load Murf voices. Please check your API key.")
                    return
                
                if "Error" in voices:
                    st.error(f"Failed to load voices: {voices['Error']}")
                    return
                
                voice_options = list(voices.keys())
                current_voice = voice_config.get('voice', '')
                
                current_index = 0
                if current_voice:
                    voice_ids = list(voices.values())
                    if current_voice in voice_ids:
                        for display_name, voice_id in voices.items():
                            if voice_id == current_voice:
                                if display_name in voice_options:
                                    current_index = voice_options.index(display_name)
                                    break
                
                selected_voice_display = st.selectbox(
                    "Choose Voice",
                    voice_options,
                    index=current_index,
                    key="voice_select",
                    help="Select a voice from Murf AI"
                )
                
                voice_config['voice'] = voices[selected_voice_display]
                
        except Exception as e:
            st.error(f"Error loading Murf voices: {str(e)}")

    def fetch_murf_voices(self, api_key: str) -> Dict[str, str]:
        if not REQUESTS_AVAILABLE:
            return {"Error": "requests library not available"}
        
        try:
            headers = {
                "accept": "application/json",
                "api-key": api_key
            }
            
            response = requests.get("https://api.murf.ai/v1/speech/voices", headers=headers, timeout=10)
            response.raise_for_status()
            
            voices_data = response.json()
            voices = {}
            
            voice_list = []
            if isinstance(voices_data, list):
                voice_list = voices_data
            elif isinstance(voices_data, dict):
                for possible_key in ['voices', 'data', 'results', 'items']:
                    if possible_key in voices_data and isinstance(voices_data[possible_key], list):
                        voice_list = voices_data[possible_key]
                        break
                
                if not voice_list:
                    for key, value in voices_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            first_item = value[0]
                            if isinstance(first_item, dict) and any(k in first_item for k in ['voice_id', 'id', 'name', 'voiceId']):
                                voice_list = value
                                break
            
            if not voice_list:
                return {"Error": "No voice data found in API response"}
            
            for voice in voice_list:
                if not isinstance(voice, dict):
                    continue
                
                voice_id = None
                for id_key in ['voice_id', 'id', 'voiceId', 'Voice_ID', 'ID']:
                    if id_key in voice and voice[id_key]:
                        voice_id = str(voice[id_key])
                        break
                
                voice_name = None
                for name_key in ['name', 'voice_name', 'voiceName', 'display_name', 'displayName', 'Name']:
                    if name_key in voice and voice[name_key]:
                        voice_name = str(voice[name_key])
                        break
                
                if not voice_id or not voice_name:
                    continue
                
                display_parts = [voice_name]
                
                gender = None
                for gender_key in ['gender', 'sex', 'Gender', 'Sex']:
                    if gender_key in voice and voice[gender_key]:
                        gender = str(voice[gender_key])
                        break
                
                if gender:
                    display_parts.append(gender.title())
                
                locale = None
                for locale_key in ['locale', 'language', 'lang', 'Locale', 'Language', 'accent']:
                    if locale_key in voice and voice[locale_key]:
                        locale = str(voice[locale_key])
                        break
                
                if locale:
                    display_parts.append(locale.upper() if len(locale) <= 3 else locale.title())
                
                age = None
                for age_key in ['age', 'Age', 'age_group', 'ageGroup', 'voice_age']:
                    if age_key in voice and voice[age_key]:
                        age = str(voice[age_key])
                        break
                
                if age:
                    display_parts.append(age.title())
                
                if len(display_parts) > 1:
                    display_name = f"{display_parts[0]} ({', '.join(display_parts[1:])})"
                else:
                    display_name = display_parts[0]
                
                voices[display_name] = voice_id
            
            if not voices:
                return {"Error": f"No valid voices extracted from {len(voice_list)} voice entries"}
            
            return voices
            
        except requests.exceptions.RequestException as e:
            return {"Error": f"API request failed: {str(e)}"}
        except (ValueError, json.JSONDecodeError) as e:
            return {"Error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"Error": f"Unexpected error: {str(e)}"}

    def configure_openai_voices(self, voice_config: Dict):
        openai_voices = {
            'alloy': 'Alloy (Neutral)',
            'echo': 'Echo (Male)',
            'fable': 'Fable (British)',
            'onyx': 'Onyx (Deep)',
            'nova': 'Nova (Female)',
            'shimmer': 'Shimmer (Soft)'
        }
        
        current_voice = voice_config.get('voice', 'alloy')
        if current_voice not in openai_voices:
            current_voice = 'alloy'
        
        selected_voice = st.selectbox(
            "Choose OpenAI Voice",
            list(openai_voices.keys()),
            index=list(openai_voices.keys()).index(current_voice),
            format_func=lambda x: openai_voices[x],
            key="voice_select",
            help="Select an OpenAI TTS voice"
        )
        
        voice_config['voice'] = selected_voice

    def configure_elevenlabs_voices(self, voice_config: Dict):
        elevenlabs_voices = {
            'rachel': 'Rachel (Professional)',
            'drew': 'Drew (Casual)',
            'clyde': 'Clyde (Warm)',
            'paul': 'Paul (Narrator)',
            'domi': 'Domi (Strong)',
            'dave': 'Dave (British)',
            'fin': 'Fin (Irish)',
            'sarah': 'Sarah (Soft)'
        }
        
        current_voice = voice_config.get('voice', 'rachel')
        if current_voice not in elevenlabs_voices:
            current_voice = 'rachel'
        
        selected_voice = st.selectbox(
            "Choose ElevenLabs Voice",
            list(elevenlabs_voices.keys()),
            index=list(elevenlabs_voices.keys()).index(current_voice),
            format_func=lambda x: elevenlabs_voices[x],
            key="voice_select",
            help="Select an ElevenLabs voice"
        )
        
        voice_config['voice'] = selected_voice

    def configure_google_voices(self, voice_config: Dict):
        google_voices = {
            'female': 'Female Voice',
            'male': 'Male Voice'
        }
        
        current_voice = voice_config.get('voice', 'female')
        if current_voice not in google_voices:
            current_voice = 'female'
        
        selected_voice = st.selectbox(
            "Choose Google Voice",
            list(google_voices.keys()),
            index=list(google_voices.keys()).index(current_voice),
            format_func=lambda x: google_voices[x],
            key="voice_select",
            help="Select a Google TTS voice"
        )
        
        voice_config['voice'] = selected_voice

    def configure_voice_parameters(self, voice_config: Dict):
        st.markdown("#### 🎛️ Voice Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            languages = {
                'en': 'English',
                'es': 'Spanish',
                'fr': 'French',
                'de': 'German',
                'it': 'Italian',
                'pt': 'Portuguese',
                'ja': 'Japanese',
                'zh': 'Chinese',
                'hi': 'Hindi'
            }
            
            current_lang = voice_config.get('language', 'en')
            if current_lang not in languages:
                current_lang = 'en'
            
            selected_language = st.selectbox(
                "Language",
                list(languages.keys()),
                index=list(languages.keys()).index(current_lang),
                format_func=lambda x: languages[x],
                help="Primary language for text-to-speech"
            )
            
            voice_config['language'] = selected_language
        
        with col2:
            speed = st.slider(
                "Speaking Speed",
                0.25, 2.0,
                float(voice_config.get('speed', 1.0)),
                0.05,
                help="How fast the voice should speak"
            )
            
            voice_config['speed'] = speed
        
        with col3:
            volume = st.slider(
                "Volume",
                0.1, 1.0,
                float(voice_config.get('volume', 0.8)),
                0.1,
                help="Audio volume level"
            )
            
            voice_config['volume'] = volume
        
        if voice_config.get('voice') and (voice_config.get('api_key') or voice_config.get('provider') == 'google'):
            if st.button("🎵 Test Voice", key="voice_test_btn"):
                self.test_voice_generation(voice_config)

    def test_voice_generation(self, voice_config: Dict):
        provider = voice_config.get('provider')
        test_text = "Hello! This is a test of your voice configuration."
        
        with st.spinner(f"🎵 Generating test audio with {provider}..."):
            try:
                if self.voice_config_manager:
                    audio_data, audio_format = self.voice_config_manager.generate_voice_preview(voice_config, test_text)
                    if audio_data:
                        st.success("✅ Voice test completed successfully!")
                        st.audio(audio_data, format=audio_format)
                    else:
                        st.error("❌ Voice test failed: Could not generate audio.")
                else:
                    st.error("❌ Voice config manager not initialized.")
            except Exception as e:
                st.error(f"❌ Voice test failed: {str(e)}")

    def configure_ai_model(self, bot_config: Dict):
        st.markdown("### 🤖 Step 3: AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.get('available_models') and self.openrouter_client:
                with st.spinner("Loading available models..."):
                    try:
                        st.session_state.available_models = self.openrouter_client.get_available_models()
                    except Exception as e:
                        st.warning(f"Could not load models: {str(e)}")
                        st.session_state.available_models = self.get_fallback_models()
            
            available_models = st.session_state.get('available_models', self.get_fallback_models())
            
            model_options = [model['id'] for model in available_models]
            model_names = {model['id']: f"{model['name']} {'🆓' if model.get('is_free') else '💰'}" for model in available_models}
            
            current_model = bot_config.get('model', st.session_state.user_config.get('default_model'))
            if current_model not in model_options:
                current_model = model_options[0] if model_options else 'meta-llama/llama-3.2-3b-instruct:free'
            
            selected_model = st.selectbox(
                "AI Model",
                model_options,
                index=model_options.index(current_model) if current_model in model_options else 0,
                format_func=lambda x: model_names.get(x, x),
                key="bot_model_select",
                help="🆓 = Free models, 💰 = Paid models"
            )
            
            bot_config['model'] = selected_model
            
            selected_model_info = next((m for m in available_models if m['id'] == selected_model), None)
            if selected_model_info:
                st.info(f"ℹ️ {selected_model_info.get('description', 'No description available')}")
        
        with col2:
            st.markdown("**Model Parameters:**")
            
            bot_config['temperature'] = st.slider(
                "Creativity (Temperature)",
                0.0, 1.0,
                float(bot_config.get('temperature', 0.7)),
                0.1,
                key="bot_temperature_slider",
                help="Higher values make responses more creative but less focused"
            )
            
            bot_config['max_tokens'] = st.number_input(
                "Max Response Length",
                100, 4000,
                int(bot_config.get('max_tokens', 1000)),
                step=100,
                key="bot_max_tokens_input",
                help="Maximum number of tokens (roughly words) in responses"
            )

    def configure_knowledge_base(self, bot_config: Dict):
        st.markdown("### 📚 Step 4: Knowledge Base (Optional)")
        st.markdown("Upload documents or add content to make your bot smarter!")
        
        kb_enabled = st.checkbox(
            "Enable Knowledge Base",
            value=bot_config.get('knowledge_base', {}).get('enabled', False),
            key="kb_enabled_checkbox",
            help="Add custom knowledge to make your bot domain-specific"
        )
        
        if not kb_enabled:
            bot_config['knowledge_base'] = {'enabled': False}
            st.info("💡 Your bot will use only its base AI training without additional knowledge.")
            return
        
        if 'knowledge_base' not in bot_config:
            bot_config['knowledge_base'] = {}
        
        kb_config = bot_config['knowledge_base']
        kb_config['enabled'] = True
        
        st.markdown("#### 🤖 Embedding Model (No API Keys Required)")
        embedding_models = {
            'sentence-transformers/all-MiniLM-L6-v2': {
                'name': 'All MiniLM L6 v2 (Recommended)',
                'size': '80MB',
                'description': 'Fast, lightweight, good quality'
            },
            'sentence-transformers/all-mpnet-base-v2': {
                'name': 'All MPNet Base v2',
                'size': '420MB',
                'description': 'High quality, larger model'
            },
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1': {
                'name': 'Multi QA MiniLM L6',
                'size': '80MB',
                'description': 'Optimized for Q&A tasks'
            },
            'BAAI/bge-small-en-v1.5': {
                'name': 'BGE Small English',
                'size': '120MB',
                'description': 'Good performance, moderate size'
            },
            'BAAI/bge-base-en-v1.5': {
                'name': 'BGE Base English',
                'size': '420MB',
                'description': 'High performance, larger size'
            }
        }
        
        current_embedding = kb_config.get('embedding_model', list(embedding_models.keys())[0])
        if current_embedding not in embedding_models:
            current_embedding = list(embedding_models.keys())[0]
        
        selected_embedding = st.selectbox(
            "Embedding Model",
            list(embedding_models.keys()),
            index=list(embedding_models.keys()).index(current_embedding),
            format_func=lambda x: f"{embedding_models[x]['name']} ({embedding_models[x]['size']})",
            key="kb_embedding_select",
            help="Choose the Hugging Face model for document embeddings"
        )
        
        kb_config['embedding_model'] = selected_embedding
        
        model_info = embedding_models[selected_embedding]
        st.info(f"ℹ️ **{model_info['name']}**: {model_info['description']} | Size: {model_info['size']}")
        
        st.markdown("#### 📄 Choose Your Data Source")
        data_source_options = [
            "Upload Documents",
            "Enter Text Manually",
            "Web URLs",
            "FAQ Database"
        ]
        
        current_data_source = kb_config.get('data_source', data_source_options[0])
        if current_data_source not in data_source_options:
            current_data_source = data_source_options[0]
        
        data_source = st.selectbox(
            "Data Source",
            data_source_options,
            index=data_source_options.index(current_data_source),
            key="kb_data_source_select",
            help="Choose how you want to add knowledge to your bot"
        )
        
        kb_config['data_source'] = data_source
        
        if data_source == "Upload Documents":
            self.configure_document_upload_with_persistence(kb_config)
        elif data_source == "Enter Text Manually":
            self.configure_manual_text(kb_config)
        elif data_source == "Web URLs":
            self.configure_web_urls(kb_config)
        elif data_source == "FAQ Database":
            self.configure_faq_database(kb_config)
        
        self.configure_kb_advanced_settings(kb_config)

    def configure_document_upload_with_persistence(self, kb_config: Dict):
        st.markdown("📄 **Upload Your Documents**")
        
        bot_id = st.session_state.get('current_bot')
        user_id = st.session_state.get('user_id')
        
        if not bot_id:
            bot_id = str(uuid.uuid4())
            st.session_state.current_bot = bot_id
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'csv', 'md', 'docx'],
            accept_multiple_files=True,
            key=f"kb_file_uploader_{bot_id}",
            help="Supported formats: PDF, TXT, CSV, MD, DOCX"
        )
        
        if uploaded_files:
            file_metadata = []
            saved_files = []
            
            for i, file in enumerate(uploaded_files):
                if self.data_manager and user_id:
                    success = self.data_manager.save_uploaded_file(user_id, bot_id, i, file)
                    if success:
                        file_metadata.append({
                            'name': file.name,
                            'type': getattr(file, 'type', 'unknown'),
                            'size': len(file.getvalue()),
                            'index': i,
                            'upload_time': datetime.now().isoformat()
                        })
                        saved_files.append(file)
            
            if file_metadata:
                kb_config['file_metadata'] = file_metadata
                kb_config['uploaded_files'] = saved_files
                kb_config['file_names'] = [f.name for f in saved_files]
                
                self.auto_save_data()
                st.success(f"📄 Successfully uploaded {len(file_metadata)} file(s)")
                
                for meta in file_metadata:
                    st.caption(f"• {meta['name']} ({self.format_file_size(meta['size'])})")
            else:
                st.error("Failed to save uploaded files")
        
        elif user_id and bot_id and self.data_manager:
            existing_metadata = kb_config.get('file_metadata', [])
            if existing_metadata:
                restored_files = []
                for meta in existing_metadata:
                    file_content = self.data_manager.load_uploaded_file(user_id, bot_id, meta['index'])
                    if file_content:
                        class MockFile:
                            def __init__(self, name, content, file_type):
                                self.name = name
                                self.type = file_type
                                self._content = content
                            
                            def getvalue(self):
                                return self._content
                            
                            def __getstate__(self):
                                return {'name': self.name, 'type': self.type, '_content': self._content}
                            
                            def __setstate__(self, state):
                                self.name = state['name']
                                self.type = state['type']
                                self._content = state['_content']
                        
                        mock_file = MockFile(meta['name'], file_content, meta['type'])
                        restored_files.append(mock_file)
                
                if restored_files:
                    kb_config['uploaded_files'] = restored_files
                    kb_config['file_names'] = [f.name for f in restored_files]
                    st.info(f"📄 Loaded {len(restored_files)} previously uploaded file(s)")
                    
                    for meta in existing_metadata:
                        st.caption(f"• {meta['name']} ({self.format_file_size(meta['size'])})")
        
        if st.button("🗑️ Clear All Files", key=f"clear_files_{bot_id}"):
            kb_config['file_metadata'] = []
            kb_config['uploaded_files'] = []
            kb_config['file_names'] = []
            self.auto_save_data()
            st.rerun()
        
        if not kb_config.get('uploaded_files') and not kb_config.get('file_metadata'):
            st.info("No files uploaded yet. Upload documents to enhance your bot's knowledge.")

    def configure_kb_advanced_settings(self, kb_config: Dict):
        st.markdown("#### ⚙️ Knowledge Base Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunking_options = ["recursive_character", "character", "token", "semantic"]
            chunking_names = {
                "recursive_character": "Smart Chunking (Recommended)",
                "character": "Character-based",
                "token": "Token-based",
                "semantic": "Semantic Chunking"
            }
            
            current_chunking = kb_config.get('chunking_strategy', chunking_options[0])
            if current_chunking not in chunking_options:
                current_chunking = chunking_options[0]
            
            kb_config['chunking_strategy'] = st.selectbox(
                "Text Chunking Method",
                chunking_options,
                index=chunking_options.index(current_chunking),
                format_func=lambda x: chunking_names.get(x, x),
                key="kb_chunking_select",
                help="How to split large documents into smaller pieces"
            )
        
        with col2:
            kb_config['chunk_size'] = st.number_input(
                "Chunk Size",
                100, 2000,
                int(kb_config.get('chunk_size', 1000)),
                step=100,
                key="kb_chunk_size_input",
                help="Size of each text chunk"
            )
        
        with col3:
            kb_config['chunk_overlap'] = st.number_input(
                "Chunk Overlap",
                0, 500,
                int(kb_config.get('chunk_overlap', 200)),
                step=50,
                key="kb_overlap_input",
                help="Overlap between chunks to maintain context"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            kb_config['max_results'] = st.number_input(
                "Max Search Results",
                1, 10,
                int(kb_config.get('max_results', 4)),
                key="kb_max_results_input",
                help="Maximum relevant documents to use for each response"
            )
        
        with col2:
            kb_config['similarity_threshold'] = st.slider(
                "Similarity Threshold",
                0.0, 1.0,
                float(kb_config.get('similarity_threshold', 0.7)),
                0.05,
                key="kb_similarity_threshold",
                help="Minimum similarity score for including documents in responses"
            )

    def configure_manual_text(self, kb_config: Dict):
        st.markdown("✏️ **Enter Your Knowledge**")
        
        manual_text = st.text_area(
            "Knowledge Base Content",
            value=kb_config.get('manual_text', ''),
            height=300,
            key="kb_manual_text_area",
            placeholder="Enter the information you want your bot to know about...",
            help="Add any text content that your bot should reference"
        )
        
        kb_config['manual_text'] = manual_text
        
        if manual_text:
            word_count = len(manual_text.split())
            char_count = len(manual_text)
            st.caption(f"📊 Content: {word_count} words, {char_count} characters")

    def configure_web_urls(self, kb_config: Dict):
        st.markdown("🌐 **Add Web Content**")
        
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            value='\n'.join(kb_config.get('urls', [])),
            key="kb_urls_text_area",
            placeholder="https://example.com/page1\nhttps://example.com/page2",
            help="The bot will scrape and learn from these web pages"
        )
        
        if urls_text:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            kb_config['urls'] = urls
            st.success(f"🔗 Added {len(urls)} URL(s)")
        else:
            kb_config['urls'] = []

    def configure_faq_database(self, kb_config: Dict):
        st.markdown("❓ **Build FAQ Database**")
        
        if 'faqs' not in kb_config:
            kb_config['faqs'] = []
        
        with st.expander("➕ Add New FAQ", expanded=len(kb_config['faqs']) == 0):
            col1, col2 = st.columns(2)
            
            with col1:
                new_question = st.text_input("Question", key="faq_question_input")
            
            with col2:
                new_answer = st.text_area("Answer", key="faq_answer_input")
            
            if st.button("Add FAQ", key="add_faq_btn", type="primary"):
                if new_question.strip() and new_answer.strip():
                    kb_config['faqs'].append({
                        'id': str(uuid.uuid4()),
                        'question': new_question.strip(),
                        'answer': new_answer.strip(),
                        'created_at': datetime.now().isoformat()
                    })
                    self.auto_save_data()
                    st.success("FAQ added!")
                    st.rerun()
                else:
                    st.error("Please enter both question and answer")
        
        if kb_config['faqs']:
            st.markdown(f"**Current FAQs ({len(kb_config['faqs'])}):**")
            for i, faq in enumerate(kb_config['faqs']):
                with st.expander(f"FAQ {i+1}: {faq['question'][:50]}...", expanded=False):
                    col1, col2, col3 = st.columns([4, 4, 1])
                    
                    with col1:
                        updated_question = st.text_input(
                            "Question",
                            value=faq['question'],
                            key=f"edit_q_{i}"
                        )
                    
                    with col2:
                        updated_answer = st.text_area(
                            "Answer",
                            value=faq['answer'],
                            key=f"edit_a_{i}"
                        )
                    
                    with col3:
                        if st.button("🗑️", key=f"del_faq_{i}", help="Delete FAQ"):
                            kb_config['faqs'].pop(i)
                            self.auto_save_data()
                            st.rerun()
                    
                    if updated_question != faq['question'] or updated_answer != faq['answer']:
                        faq['question'] = updated_question
                        faq['answer'] = updated_answer
                        self.auto_save_data()
        else:
            st.info("No FAQs added yet. Click 'Add New FAQ' to get started.")

    def configure_advanced_settings(self, bot_config: Dict):
        st.markdown("### ⚙️ Step 5: Advanced Settings")
        
        if 'advanced_settings' not in bot_config:
            bot_config['advanced_settings'] = {}
        
        adv_config = bot_config['advanced_settings']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Conversation Settings**")
            
            adv_config['context_window'] = st.number_input(
                "Context Window (messages)",
                1, 20,
                int(adv_config.get('context_window', 5)),
                key="adv_context_window_input",
                help="Number of previous messages to remember in conversation"
            )
            
            response_formats = ["conversational", "structured", "bullet_points", "detailed"]
            format_names = {
                "conversational": "Natural conversation",
                "structured": "Well-organized responses",
                "bullet_points": "Bullet point format",
                "detailed": "Comprehensive details"
            }
            
            current_format = adv_config.get('response_format', response_formats[0])
            if current_format not in response_formats:
                current_format = response_formats[0]
            
            adv_config['response_format'] = st.selectbox(
                "Response Format",
                response_formats,
                index=response_formats.index(current_format),
                format_func=lambda x: format_names.get(x, x),
                key="adv_response_format_select",
                help="How responses should be formatted"
            )
            
            error_handling_options = ["graceful", "detailed", "silent"]
            error_names = {
                "graceful": "User-friendly error messages",
                "detailed": "Show technical details",
                "silent": "Minimal error information"
            }
            
            current_error = adv_config.get('error_handling', error_handling_options[0])
            if current_error not in error_handling_options:
                current_error = error_handling_options[0]
            
            adv_config['error_handling'] = st.selectbox(
                "Error Handling",
                error_handling_options,
                index=error_handling_options.index(current_error),
                format_func=lambda x: error_names.get(x, x),
                key="adv_error_handling_select",
                help="How to handle errors and failures"
            )
        
        with col2:
            st.markdown("**Custom Instructions**")
            
            adv_config['system_message'] = st.text_area(
                "System Message (Optional)",
                value=adv_config.get('system_message', ''),
                key="adv_system_message_input",
                placeholder="You are a helpful assistant that...",
                help="Custom system message to define bot behavior"
            )
            
            adv_config['custom_instructions'] = st.text_area(
                "Additional Instructions (Optional)",
                value=adv_config.get('custom_instructions', ''),
                key="adv_custom_instructions_input",
                placeholder="Always be polite and professional...",
                help="Specific guidelines for how the bot should behave"
            )

    def show_save_deploy_options(self, bot_config: Dict, is_editing: bool):
        st.markdown("### 🚀 Step 6: Save & Deploy")
        
        is_valid, validation_message = self.validate_bot_config(bot_config)
        
        if not is_valid:
            st.error(f"❌ {validation_message}")
        else:
            st.success("✅ Bot configuration is valid!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(
                "💾 Save Draft",
                key="save_draft_btn",
                type="secondary",
                use_container_width=True,
                disabled=not bot_config.get('name')
            ):
                if self.save_bot_config(bot_config, 'draft'):
                    st.success("Bot saved as draft!")
        
        with col2:
            if st.button(
                "🧪 Save & Test",
                key="save_test_btn",
                type="secondary",
                use_container_width=True,
                disabled=not is_valid
            ):
                if self.save_bot_config(bot_config, 'testing'):
                    st.success("Bot saved! You can now test it.")
        
        with col3:
            if st.button(
                "🚀 Deploy Bot",
                key="deploy_bot_btn",
                type="primary",
                use_container_width=True,
                disabled=not is_valid
            ):
                if self.save_bot_config(bot_config, 'active'):
                    st.success("🎉 Bot deployed successfully!")
                    st.balloons()

    def show_fastapi_generation(self, bot_config: Dict):
        st.markdown("### ⚡ Step 7: Generate FastAPI")
        st.markdown("Get a complete FastAPI implementation for your bot!")
        
        is_saved = bot_config.get('bot_id') in st.session_state.get('bots', {})
        
        if not is_saved:
            st.warning("⚠️ Please save your bot first before generating FastAPI code.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**FastAPI Features:**")
            st.markdown("✅ Complete REST API implementation")
            st.markdown("✅ Chat endpoint with streaming support")
            st.markdown("✅ Voice response integration")
            st.markdown("✅ Knowledge base RAG system")
            st.markdown("✅ CORS configuration")
            st.markdown("✅ Error handling and logging")
            st.markdown("✅ API documentation (Swagger)")
            st.markdown("✅ Production-ready code")
        
        with col2:
            st.markdown("**Deployment Options:**")
            st.markdown("🐳 Docker containerization")
            st.markdown("☁️ Cloud deployment ready")
            st.markdown("🔧 Environment configuration")
            st.markdown("📦 Requirements.txt included")
            st.markdown("🔐 API key management")
            st.markdown("📊 Health check endpoints")
            st.markdown("⚡ Async/await support")
            st.markdown("🚀 Auto-scaling compatible")
        
        if st.button(
            "⚡ Generate FastAPI Code",
            key="generate_fastapi_btn",
            type="primary",
            use_container_width=True,
            help="Generate complete FastAPI implementation for your bot"
        ):
            with st.spinner("🔄 Generating FastAPI code..."):
                fastapi_code = self.generate_fastapi_code(bot_config)
                
                if fastapi_code:
                    st.success("🎉 FastAPI code generated successfully!")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["🚀 main.py", "📋 requirements.txt", "🐳 Dockerfile", "⚙️ .env.example"])
                    
                    with tab1:
                        st.markdown("### FastAPI Main Application")
                        st.code(fastapi_code['main.py'], language="python", line_numbers=True)
                        
                        st.download_button(
                            label="📥 Download main.py",
                            data=fastapi_code['main.py'],
                            file_name="main.py",
                            mime="text/plain"
                        )
                    
                    with tab2:
                        st.markdown("### Python Dependencies")
                        st.code(fastapi_code['requirements.txt'], language="text")
                        
                        st.download_button(
                            label="📥 Download requirements.txt",
                            data=fastapi_code['requirements.txt'],
                            file_name="requirements.txt",
                            mime="text/plain"
                        )
                    
                    with tab3:
                        st.markdown("### Docker Configuration")
                        st.code(fastapi_code['Dockerfile'], language="dockerfile")
                        
                        st.download_button(
                            label="📥 Download Dockerfile",
                            data=fastapi_code['Dockerfile'],
                            file_name="Dockerfile",
                            mime="text/plain"
                        )
                    
                    with tab4:
                        st.markdown("### Environment Configuration")
                        st.code(fastapi_code['.env.example'], language="bash")
                        
                        st.download_button(
                            label="📥 Download .env.example",
                            data=fastapi_code['.env.example'],
                            file_name=".env.example",
                            mime="text/plain"
                        )
                    
                    with st.expander("🚀 Deployment Instructions", expanded=False):
                        st.markdown("""
### Quick Start Guide

1. **Download all files** using the download buttons above

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
- Copy `.env.example` to `.env`
- Fill in your API keys and configuration

4. **Run the FastAPI server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. **Access your API**:
- API Documentation: http://localhost:8000/docs
- Chat Endpoint: POST http://localhost:8000/chat
- Health Check: GET http://localhost:8000/health

### Docker Deployment

1. **Build Docker image**:
```bash
docker build -t my-chatbot-api .
```

2. **Run container**:
```bash
docker run -p 8000:8000 --env-file .env my-chatbot-api
```

### API Usage Example

```python
import requests

# Chat with your bot
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "message": "Hello, how can you help me?",
        "stream": False
    }
)
print(response.json())
```
""")
                else:
                    st.error("❌ Failed to generate FastAPI code. Please try again.")

    

    def generate_fastapi_code(self, bot_config: Dict) -> Dict[str, str]:
            try:
                bot_name = bot_config.get('name', 'AI Assistant')
                bot_description = bot_config.get('description', 'AI Assistant with FastAPI')
                model = bot_config.get('model', 'meta-llama/llama-3.2-3b-instruct:free')
                temperature = bot_config.get('temperature', 0.7)
                max_tokens = bot_config.get('max_tokens', 1000)

                voice_config = bot_config.get('voice_config', {})
                voice_enabled = voice_config.get('enabled', False)
                voice_provider = voice_config.get('provider', 'murf')
                response_mode = voice_config.get('response_mode', 'voice')

                kb_config = bot_config.get('knowledge_base', {})
                kb_enabled = kb_config.get('enabled', False)
                embedding_model = kb_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')

                adv_config = bot_config.get('advanced_settings', {})
                context_window = adv_config.get('context_window', 5)
                system_message = adv_config.get('system_message', '')
                custom_instructions = adv_config.get('custom_instructions', '')

                # --- Start of conditional code blocks for main.py ---

                # Knowledge Base Imports
                kb_imports_code = ""
                if kb_enabled:
                    kb_imports_code = f'''try:
        from sentence_transformers import SentenceTransformer
        from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
        from langchain_community.vectorstores import Chroma
        from langchain.schema import Document
        import chromadb
        KB_AVAILABLE = True
    except ImportError:
        KB_AVAILABLE = False
        print("Warning: Knowledge base dependencies not available")'''
                else:
                    kb_imports_code = "KB_AVAILABLE = False"

                # Voice Processing Imports
                voice_imports_code = ""
                if voice_enabled:
                    voice_imports_code = f'''try:
        import base64
        import io
        VOICE_AVAILABLE = True
    except ImportError:
        VOICE_AVAILABLE = False
        print("Warning: Voice processing dependencies not available")'''
                else:
                    voice_imports_code = "VOICE_AVAILABLE = False"

                # Knowledge Base Class Definition
                kb_class_code = ""
                if kb_enabled:
                    kb_class_code = f'''
    class KnowledgeBase:
        """Knowledge base for RAG functionality"""

        def __init__(self):
            self.embeddings = None
            self.vectorstore = None
            self.text_splitter = None
            self.initialized = False

        async def initialize(self):
            """Initialize knowledge base components"""
            try:
                if not KB_AVAILABLE:
                    logger.warning("Knowledge base dependencies not available")
                    return False

                self.embeddings = SentenceTransformer('{embedding_model}')

                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size={kb_config.get('chunk_size', 1000)},
                    chunk_overlap={kb_config.get('chunk_overlap', 200)}
                )

                self.vectorstore = None
                self.initialized = True
                logger.info("Knowledge base initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize knowledge base: {{str(e)}}")
                return False

        def embed_text(self, text: str) -> List[float]:
            """Generate embeddings for text"""
            if not self.embeddings:
                return []
            return self.embeddings.encode([text])[0].tolist()

        async def add_documents(self, documents: List[str]) -> bool:
            """Add documents to the knowledge base"""
            try:
                if not self.initialized:
                    await self.initialize()

                if not documents:
                    return True

                doc_chunks = []
                for doc in documents:
                    chunks = self.text_splitter.split_text(doc)
                    doc_chunks.extend([Document(page_content=chunk) for chunk in chunks])

                if self.vectorstore is None:
                    texts = [doc.page_content for doc in doc_chunks]
                    embeddings_list = [self.embed_text(text) for text in texts]

                    self.vectorstore = Chroma.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        persist_directory="./vectordb"
                    )
                else:
                    texts = [doc.page_content for doc in doc_chunks]
                    self.vectorstore.add_texts(texts)

                logger.info(f"Added {{len(doc_chunks)}} document chunks to knowledge base")
                return True

            except Exception as e:
                logger.error(f"Failed to add documents: {{str(e)}}")
                return False

        async def search(self, query: str, k: int = {kb_config.get('max_results', 4)}) -> List[str]:
            """Search for relevant documents"""
            try:
                if not self.vectorstore:
                    return []

                results = self.vectorstore.similarity_search(query, k=k)
                return [doc.page_content for doc in results]

            except Exception as e:
                logger.error(f"Search error: {{str(e)}}")
                return []'''

                # Voice Synthesizer Class Definition
                voice_synthesizer_class_code = ""
                if voice_enabled:
                    voice_synthesizer_class_code = f'''
    class VoiceSynthesizer:
        """Voice synthesis for text-to-speech"""

        def __init__(self):
            self.provider = "{voice_provider}"
            self.api_key = os.getenv("{voice_provider.upper()}_API_KEY", "")
            self.voice_id = "{voice_config.get('voice', '')}"
            self.language = "{voice_config.get('language', 'en')}"
            self.speed = {voice_config.get('speed', 1.0)}
            self.volume = {voice_config.get('volume', 0.8)}

        async def synthesize(self, text: str) -> tuple[Optional[bytes], Optional[str]]:
            """Convert text to speech"""
            try:
                if not VOICE_AVAILABLE or not self.api_key:
                    return None, None

                if self.provider == "murf":
                    return await self._synthesize_murf(text)
                elif self.provider == "openai":
                    return await self._synthesize_openai(text)
                elif self.provider == "elevenlabs":
                    return await self._synthesize_elevenlabs(text)
                else:
                    logger.warning(f"Unsupported voice provider: {{self.provider}}")
                    return None, None

            except Exception as e:
                logger.error(f"Voice synthesis error: {{str(e)}}")
                return None, None

        async def _synthesize_murf(self, text: str) -> tuple[Optional[bytes], Optional[str]]:
            """Synthesize speech using Murf AI"""
            try:
                if not REQUESTS_AVAILABLE:
                    return None, None

                headers = {{
                    "accept": "application/json",
                    "content-type": "application/json",
                    "api-key": self.api_key
                }}

                payload = {{
                    "text": text,
                    "voice_id": self.voice_id,
                    "speed": max(0.5, min(1.5, self.speed))
                }}

                response = requests.post(
                    "https://api.murf.ai/v1/speech/generate",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

                if "audioFile" in result:
                    audio_url = result["audioFile"].strip("[]\\'\"")
                    audio_response = requests.get(audio_url, timeout=30)
                    audio_response.raise_for_status()
                    return audio_response.content, "audio/wav"

                return None, None

            except Exception as e:
                logger.error(f"Murf synthesis error: {{str(e)}}")
                return None, None

        async def _synthesize_openai(self, text: str) -> tuple[Optional[bytes], Optional[str]]:
            """Synthesize speech using OpenAI TTS"""
            # Placeholder for OpenAI TTS implementation
            return None, None

        async def _synthesize_elevenlabs(self, text: str) -> tuple[Optional[bytes], Optional[str]]:
            """Synthesize speech using ElevenLabs"""
            # Placeholder for ElevenLabs implementation
            return None, None'''

                # Initialize Services Block
                initialize_services_kb_code = ""
                if kb_enabled:
                    initialize_services_kb_code = f'''    if KB_AVAILABLE:
            knowledge_base = KnowledgeBase()
            await knowledge_base.initialize()'''

                initialize_services_voice_code = ""
                if voice_enabled:
                    initialize_services_voice_code = f'''    if VOICE_AVAILABLE:
            voice_synthesizer = VoiceSynthesizer()
            logger.info("Voice synthesizer initialized")'''

                # Chat Endpoint Knowledge Base Logic
                chat_endpoint_kb_logic = ""
                if kb_enabled:
                    chat_endpoint_kb_logic = f'''        relevant_docs = []
            if knowledge_base and knowledge_base.initialized:
                relevant_docs = await knowledge_base.search(request.message)
                if relevant_docs:
                    context = "\\n".join(relevant_docs)
                    enhanced_message = f"Context from knowledge base:\\n{{context}}\\n\\nUser question: {{request.message}}"
                    messages.append({{"role": "user", "content": enhanced_message}})
                else:
                    messages.append({{"role": "user", "content": request.message}})
            else:
                messages.append({{"role": "user", "content": request.message}})'''
                else:
                    chat_endpoint_kb_logic = '''            messages.append({"role": "user", "content": request.message})'''

                # Chat Endpoint Voice Logic
                chat_endpoint_voice_logic = ""
                if voice_enabled:
                    chat_endpoint_voice_logic = f'''            if "{response_mode}" in ["voice", "both"] and voice_synthesizer:
                    voice_bytes, voice_fmt = await voice_synthesizer.synthesize(response_text)
                    if voice_bytes:
                        voice_data = base64.b64encode(voice_bytes).decode('utf-8')
                        voice_format = voice_fmt'''

                # Chat Endpoint Metadata Logic
                chat_endpoint_metadata_kb_logic = ""
                if kb_enabled:
                    chat_endpoint_metadata_kb_logic = f'''                    "relevant_docs_count": len(relevant_docs),'''

                # Knowledge Base Endpoints
                kb_endpoints_code = ""
                if kb_enabled:
                    kb_endpoints_code = f'''
    @app.post("/knowledge-base/add")
    async def add_knowledge(documents: List[str]):
        """Add documents to the knowledge base"""
        try:
            if not knowledge_base:
                raise HTTPException(status_code=503, detail="Knowledge base not available")

            success = await knowledge_base.add_documents(documents)
            if success:
                return {{"message": f"Successfully added {{len(documents)}} documents"}}
            else:
                raise HTTPException(status_code=500, detail="Failed to add documents")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Add knowledge error: {{str(e)}}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/knowledge-base/search")
    async def search_knowledge(query: str, limit: int = 4):
        """Search the knowledge base"""
        try:
            if not knowledge_base:
                raise HTTPException(status_code=503, detail="Knowledge base not available")

            results = await knowledge_base.search(query, k=limit)
            return {{
                "query": query,
                "results": results,
                "count": len(results)
            }}

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Search knowledge error: {{str(e)}}")
            raise HTTPException(status_code=500, detail=str(e))'''

                # --- End of conditional code blocks ---

                main_py = f'''"""
    {bot_name} - FastAPI Implementation
    Generated automatically by AI Chatbot Creator

    This is a complete FastAPI application for your AI chatbot with:
    - Chat endpoints with streaming support
    - Voice response integration (if enabled)
    - Response mode: {response_mode.upper()}
    {"- Voice response generation" if voice_enabled else ""}
    {"- Knowledge base integration with RAG" if kb_enabled else ""}
    - Streaming responses
    - Conversation context management
    - Health monitoring
    """

    import os
    import json
    import uuid
    import asyncio
    import logging
    from datetime import datetime
    from typing import Dict, List, Optional, AsyncGenerator, Any
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn

    try:
        import requests
        REQUESTS_AVAILABLE = True
    except ImportError:
        REQUESTS_AVAILABLE = False
        print("Warning: requests not available - some features may not work")

    {"# Knowledge base imports" if kb_enabled else ""}
    {kb_imports_code}

    {"# Voice processing imports" if voice_enabled else ""}
    {voice_imports_code}

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    openrouter_client = None
    knowledge_base = None
    voice_synthesizer = None

    class ChatRequest(BaseModel):
        message: str = Field(..., description="The user's message")
        conversation_id: Optional[str] = Field(None, description="Unique conversation identifier")
        stream: bool = Field(False, description="Whether to stream the response")
        include_voice: bool = Field({str(voice_enabled and response_mode in ['voice', 'both']).lower()}, description="Whether to include voice response")
        context: Optional[List[Dict]] = Field(None, description="Previous conversation context")

    class ChatResponse(BaseModel):
        response: str = Field(..., description="The bot's response")
        conversation_id: str = Field(..., description="Conversation identifier")
        timestamp: str = Field(..., description="Response timestamp")
        model: str = Field(..., description="Model used for response")
        response_mode: str = Field(..., description="Response mode (voice, text, both)")
        voice_data: Optional[str] = Field(None, description="Base64 encoded voice data")
        voice_format: Optional[str] = Field(None, description="Voice audio format")
        metadata: Optional[Dict] = Field(None, description="Additional response metadata")

    class HealthResponse(BaseModel):
        status: str = Field(..., description="Service status")
        timestamp: str = Field(..., description="Health check timestamp")
        version: str = Field(..., description="API version")
        services: Dict[str, str] = Field(..., description="Service component statuses")

    class OpenRouterClient:
        """OpenRouter API client for LLM interactions"""

        def __init__(self, api_key: str):
            self.api_key = api_key
            self.base_url = "https://openrouter.ai/api/v1"
            self.headers = {{
                "Authorization": f"Bearer {{api_key}}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "{bot_name} API"
            }}

        async def chat_completion(self, messages: List[Dict], model: str = "{model}", **kwargs) -> Dict:
            """Send chat completion request to OpenRouter"""
            try:
                payload = {{
                    "model": model,
                    "messages": messages,
                    "temperature": {temperature},
                    "max_tokens": {max_tokens},
                    **kwargs
                }}

                if not REQUESTS_AVAILABLE:
                    raise Exception("Requests library not available")

                response = requests.post(
                    f"{{self.base_url}}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                response.raise_for_status()
                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return {{"success": True, "content": content, "model": model}}
                else:
                    return {{"success": False, "error": "No response generated"}}

            except Exception as e:
                logger.error(f"OpenRouter API error: {{str(e)}}")
                return {{"success": False, "error": str(e)}}

        async def chat_completion_stream(self, messages: List[Dict], model: str = "{model}", **kwargs) -> AsyncGenerator[str, None]:
            """Stream chat completion from OpenRouter"""
            try:
                payload = {{
                    "model": model,
                    "messages": messages,
                    "temperature": {temperature},
                    "max_tokens": {max_tokens},
                    "stream": True,
                    **kwargs
                }}

                if not REQUESTS_AVAILABLE:
                    yield "Error: Requests library not available"
                    return

                response = requests.post(
                    f"{{self.base_url}}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    stream=True,
                    timeout=30
                )

                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {{}})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

            except Exception as e:
                logger.error(f"Streaming error: {{str(e)}}")
                yield f"Error: {{str(e)}}"

    {kb_class_code}

    {voice_synthesizer_class_code}

    async def initialize_services():
        """Initialize all required services"""
        global openrouter_client, knowledge_base, voice_synthesizer

        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            openrouter_client = OpenRouterClient(api_key)
            logger.info("OpenRouter client initialized")
        else:
            logger.warning("OPENROUTER_API_KEY not found - chat functionality will be limited")

    {initialize_services_kb_code}

    {initialize_services_voice_code}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager"""
        logger.info("Starting {bot_name} API...")
        await initialize_services()
        yield
        logger.info("Shutting down {bot_name} API...")

    app = FastAPI(
        title="{bot_name} API",
        description="""{bot_description}

    This API provides chat functionality with the following features:
    - Response Mode: {response_mode.upper()}
    {"- Voice response generation" if voice_enabled else ""}
    {"- Knowledge base integration with RAG" if kb_enabled else ""}
    - Streaming responses
    - Conversation context management
    - Health monitoring
    """,
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        services = {{
            "openrouter": "available" if openrouter_client else "unavailable",
            {"knowledge_base": "available" if knowledge_base and knowledge_base.initialized else "unavailable" if kb_enabled else ""},
            {"voice_synthesizer": "available" if voice_synthesizer else "unavailable" if voice_enabled else ""}
        }}

        # Remove trailing commas from services dict (if any were left by conditional logic)
        # This part of the original code was trying to remove trailing commas from string values,
        # but the dictionary syntax itself doesn't allow trailing commas in keys.
        # The conditional inclusion of dictionary items should handle this.
        # If a service is not enabled, its key-value pair simply won't be in the dictionary.
        # The original logic `services = {{k: v.rstrip(',') if isinstance(v, str) and v.endswith(',') else v for k, v in services.items()}}`
        # is likely unnecessary if the conditional string generation is correct.
        # Let's ensure the conditional parts generate valid dictionary entries.
        
        # Re-evaluating the services dictionary construction to avoid trailing commas
        # and ensure correct conditional inclusion.
        dynamic_services = {{}}
        dynamic_services["openrouter"] = "available" if openrouter_client else "unavailable"
        if kb_enabled:
            dynamic_services["knowledge_base"] = "available" if knowledge_base and knowledge_base.initialized else "unavailable"
        if voice_enabled:
            dynamic_services["voice_synthesizer"] = "available" if voice_synthesizer else "unavailable"

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            services=dynamic_services
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
        """Main chat endpoint with response mode support"""
        try:
            conversation_id = request.conversation_id or str(uuid.uuid4())

            if not openrouter_client:
                raise HTTPException(status_code=503, detail="OpenRouter client not available")

            messages = []
            system_msg = "{system_message or custom_instructions or f'You are {bot_name}, a helpful AI assistant.'}"
            if system_msg.strip():
                messages.append({{"role": "system", "content": system_msg}})

            if request.context:
                messages.extend(request.context[-{context_window}:])

    {chat_endpoint_kb_logic}

            if request.stream:
                async def generate_stream():
                    full_response = ""
                    async for chunk in openrouter_client.chat_completion_stream(messages):
                        full_response += chunk
                        yield f"data: {{json.dumps({{'chunk': chunk}})}}\\n\\n"

                    final_data = {{
                        "response": full_response,
                        "conversation_id": conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "model": "{model}",
                        "response_mode": "{response_mode}",
                        "done": True
                    }}
                    yield f"data: {{json.dumps(final_data)}}\\n\\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={{"Cache-Control": "no-cache", "Connection": "keep-alive"}}
                )
            else:
                result = await openrouter_client.chat_completion(messages)

                if not result.get("success"):
                    raise HTTPException(status_code=500, detail=f"AI response error: {{result.get('error')}}")

                response_text = result["content"]

                voice_data = None
                voice_format = None
    {chat_endpoint_voice_logic}

                final_response_text = response_text if "{response_mode}" in ["text", "both"] else ""

                metadata = {{
    {chat_endpoint_metadata_kb_logic}
                    "response_length": len(response_text),
                    "model_used": result.get("model", "{model}"),
                    "response_mode": "{response_mode}",
                    "voice_generated": voice_data is not None
                }}

                return ChatResponse(
                    response=final_response_text,
                    conversation_id=conversation_id,
                    timestamp=datetime.now().isoformat(),
                    model=result.get("model", "{model}"),
                    response_mode="{response_mode}",
                    voice_data=voice_data,
                    voice_format=voice_format,
                    metadata=metadata
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Chat endpoint error: {{str(e)}}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {{str(e)}}")

    {kb_endpoints_code}

    @app.get("/config")
    async def get_config():
        """Get bot configuration"""
        return {{
            "name": "{bot_name}",
            "description": "{bot_description}",
            "model": "{model}",
            "voice_enabled": {str(voice_enabled).lower()},
            "response_mode": "{response_mode}",
            "knowledge_base_enabled": {str(kb_enabled).lower()},
            "features": {{
                "streaming": True,
                "voice_responses": {str(voice_enabled).lower()},
                "knowledge_base": {str(kb_enabled).lower()},
                "conversation_context": True,
                "response_modes": ["{response_mode}"]
            }}
        }}

    if __name__ == "__main__":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 8000)),
            reload=os.getenv("ENVIRONMENT") == "development"
        )
    '''

                requirements_txt = f'''fastapi>=0.104.1
    uvicorn[standard]>=0.24.0
    pydantic>=2.0.0
    python-multipart>=0.0.6
    requests>=2.31.0
    python-dotenv>=1.0.0
    structlog>=23.0.0
    {"sentence-transformers>=4.0.0" if kb_enabled else ""}
    {"langchain>=0.0.350" if kb_enabled else ""}
    {"langchain-community>=0.0.10" if kb_enabled else ""}
    {"chromadb>=0.4.15" if kb_enabled else ""}
    {"numpy>=1.24.0" if kb_enabled else ""}
    {"scikit-learn>=1.3.0" if kb_enabled else ""}
    gunicorn>=21.2.0
    '''

                dockerfile = f'''FROM python:3.11-slim

    WORKDIR /app

    ENV PYTHONUNBUFFERED=1
    ENV PYTHONDONTWRITEBYTECODE=1

    RUN apt-get update \\
        && apt-get install -y --no-install-recommends \\
            gcc \\
            g++ \\
        && rm -rf /var/lib/apt/lists/*

    COPY requirements.txt .

    RUN pip install --no-cache-dir --upgrade pip \\
        && pip install --no-cache-dir -r requirements.txt

    COPY main.py .
    COPY .env.example .env

    RUN mkdir -p vectordb logs

    EXPOSE 8000

    HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
        CMD curl -f http://localhost:8000/health || exit 1

    CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    '''

                env_example = f'''ENVIRONMENT=production
    PORT=8000
    APP_NAME="{bot_name}"

    OPENROUTER_API_KEY=your_openrouter_api_key_here

    DEFAULT_MODEL="{model}"
    DEFAULT_TEMPERATURE={temperature}
    DEFAULT_MAX_TOKENS={max_tokens}

    {"VOICE_ENABLED=" + str(voice_enabled).lower() if voice_enabled else ""}
    {"RESPONSE_MODE=\"" + response_mode + "\"" if voice_enabled else ""}
    {"VOICE_PROVIDER=\"" + voice_provider + "\"" if voice_enabled else ""}
    {voice_provider.upper() + "_API_KEY=your_" + voice_provider + "_api_key_here" if voice_enabled else ""}
    {"VOICE_ID=\"" + voice_config.get('voice', '') + "\"" if voice_enabled else ""}
    {"VOICE_LANGUAGE=\"" + voice_config.get('language', 'en') + "\"" if voice_enabled else ""}
    {"VOICE_SPEED=" + str(voice_config.get('speed', 1.0)) if voice_enabled else ""}
    {"VOICE_VOLUME=" + str(voice_config.get('volume', 0.8)) if voice_enabled else ""}

    {"KNOWLEDGE_BASE_ENABLED=" + str(kb_enabled).lower() if kb_enabled else ""}
    {"EMBEDDING_MODEL=\"" + embedding_model + "\"" if kb_enabled else ""}
    {"CHUNK_SIZE=" + str(kb_config.get('chunk_size', 1000)) if kb_enabled else ""}
    {"CHUNK_OVERLAP=" + str(kb_config.get('chunk_overlap', 200)) if kb_enabled else ""}
    {"MAX_SEARCH_RESULTS=" + str(kb_config.get('max_results', 4)) if kb_enabled else ""}
    {"SIMILARITY_THRESHOLD=" + str(kb_config.get('similarity_threshold', 0.7)) if kb_enabled else ""}

    CONTEXT_WINDOW={context_window}
    RESPONSE_FORMAT="{adv_config.get('response_format', 'conversational')}"
    ERROR_HANDLING="{adv_config.get('error_handling', 'graceful')}"

    SYSTEM_MESSAGE="{system_message.replace('"', '\\"') if system_message else ''}"
    CUSTOM_INSTRUCTIONS="{custom_instructions.replace('"', '\\"') if custom_instructions else ''}"

    ALLOWED_ORIGINS=*
    ALLOWED_METHODS=GET,POST,PUT,DELETE
    ALLOWED_HEADERS=*

    LOG_LEVEL=INFO
    LOG_FORMAT=json "'''
                
                return {
                    'main.py': main_py,
                    'requirements.txt': requirements_txt,
                    'Dockerfile': dockerfile,
                    '.env.example': env_example
                }

            except Exception as e:
                return {"error": f"Error generating FastAPI code: {str(e)}"}

    def validate_bot_config(self, bot_config: Dict) -> tuple[bool, str]:
        if not bot_config.get('name', '').strip():
            return False, "Bot name is required"
        
        if not bot_config.get('model'):
            return False, "AI model must be selected"
        
        voice_config = bot_config.get('voice_config', {})
        if voice_config.get('enabled', False):
            provider = voice_config.get('provider')
            if not provider:
                return False, "Voice provider must be selected"
            
            if provider != 'google' and not voice_config.get('api_key'):
                return False, f"{provider} API key is required for voice responses"
            
            if not voice_config.get('voice'):
                return False, "Voice type must be selected"
        
        if bot_config.get('knowledge_base', {}).get('enabled'):
            if KNOWLEDGE_PROCESSOR_AVAILABLE:
                temp_knowledge_processor = KnowledgeProcessor()
                kb_valid, kb_message = temp_knowledge_processor.validate_knowledge_base(
                    bot_config['knowledge_base']
                )
                
                if not kb_valid:
                    return False, f"Knowledge base error: {kb_message}"
            else:
                st.warning("Knowledge base validation skipped - processor not available")
        
        return True, "Configuration is valid"

    def save_bot_config(self, bot_config: Dict, status: str) -> bool:
        try:
            bot_id = st.session_state.get('current_bot')
            user_id = st.session_state.get('user_id')
            
            if not bot_id:
                bot_id = str(uuid.uuid4())
                st.session_state.current_bot = bot_id
            
            if not user_id:
                st.error("User not authenticated")
                return False
            
            bot_config['bot_id'] = bot_id
            bot_config['status'] = status
            bot_config['updated_at'] = datetime.now().isoformat()
            bot_config['user_id'] = user_id
            bot_config['user_config'] = st.session_state.get('user_config', {}).copy()
            
            kb_success = True
            if bot_config.get('knowledge_base', {}).get('enabled'):
                st.info("🔄 Setting up knowledge base...")
                kb_success = self.initialize_knowledge_base(bot_id, bot_config)
                
                if not kb_success:
                    bot_config['knowledge_base']['_initialization_failed'] = True
                    st.warning("⚠️ Bot saved but knowledge base setup failed")
            
            if '_knowledge_processor' in bot_config:
                del bot_config['_knowledge_processor']
            
            st.session_state.bots[bot_id] = bot_config
            st.session_state.last_saved_bot_id = bot_id
            
            if self.data_manager:
                self.data_manager.save_user_bots(user_id, st.session_state.bots)
                self.data_manager.save_user_config(user_id, st.session_state.user_config)
            
            if kb_success and bot_config.get('knowledge_base', {}).get('enabled'):
                file_metadata = bot_config.get('knowledge_base', {}).get('file_metadata', [])
                if file_metadata:
                    st.success(f"🎉 Bot deployed with {len(file_metadata)} documents!")
                else:
                    st.success("✅ Bot deployed with empty knowledge base!")
            else:
                st.success("✅ Bot saved successfully!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving bot: {str(e)}")
            return False

    def load_existing_bot(self, bot_config: Dict):
        try:
            if 'bot_id' in bot_config:
                st.session_state.current_bot = bot_config['bot_id']
            
            st.session_state.editing_bot_config = bot_config
            st.session_state.is_editing_mode = True
            
            if bot_config.get('knowledge_base', {}).get('enabled'):
                st.info("🔄 Re-initializing knowledge base for editing...")
                self.initialize_knowledge_base(bot_config['bot_id'], bot_config)
                
        except Exception as e:
            st.error(f"Error loading existing bot: {str(e)}")

    def initialize_knowledge_base(self, bot_id: str, bot_config: Dict):
        try:
            kb_config = bot_config.get('knowledge_base', {})
            
            if not kb_config.get('enabled'):
                self.knowledge_processor = None
                return True
            
            if not KNOWLEDGE_PROCESSOR_AVAILABLE:
                st.error("⚠️ Knowledge processor not available")
                st.info("Install required packages: pip install sentence-transformers langchain langchain-community chromadb")
                self.knowledge_processor = None
                return False
            
            st.info("🤖 Using Hugging Face embeddings (no API keys required)")
            
            with st.spinner("🔄 Initializing knowledge processor..."):
                self.knowledge_processor = KnowledgeProcessor()
                
                embedding_model = kb_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                embeddings_success = self.knowledge_processor.initialize_embeddings(
                    model_name=embedding_model
                )
                
                if not embeddings_success:
                    st.error("❌ Failed to initialize Hugging Face embeddings")
                    st.info("💡 Trying with fallback model...")
                    
                    embeddings_success = self.knowledge_processor.initialize_embeddings(
                        model_name='sentence-transformers/all-MiniLM-L6-v2'
                    )
                    
                    if not embeddings_success:
                        self.knowledge_processor = None
                        return False
                
                vectorstore = self.knowledge_processor.setup_vectorstore_with_persistence(bot_id)
                if not vectorstore:
                    st.error("❌ Failed to setup vectorstore")
                    self.knowledge_processor = None
                    return False
                
                chunks_processed = 0
                user_id = st.session_state.get('user_id')
                
                if user_id and self.data_manager:
                    file_metadata = kb_config.get('file_metadata', [])
                    if file_metadata:
                        with st.spinner(f"🔄 Processing {len(file_metadata)} stored documents..."):
                            restored_files = []
                            for meta in file_metadata:
                                file_content = self.data_manager.load_uploaded_file(user_id, bot_id, meta['index'])
                                if file_content:
                                    class MockFile:
                                        def __init__(self, name, content, file_type):
                                            self.name = name
                                            self.type = file_type
                                            self._content = content
                                        
                                        def getvalue(self):
                                            return self._content
                                        
                                        def __getstate__(self):
                                            return {'name': self.name, 'type': self.type, '_content': self._content}
                                        
                                        def __setstate__(self, state):
                                            self.name = state['name']
                                            self.type = state['type']
                                            self._content = state['_content']
                                    
                                    mock_file = MockFile(meta['name'], file_content, meta['type'])
                                    restored_files.append(mock_file)
                            
                            if restored_files:
                                chunks_processed = self.knowledge_processor.process_uploaded_files(
                                    uploaded_files=restored_files,
                                    bot_id=bot_id,
                                    chunk_size=kb_config.get('chunk_size', 1000),
                                    chunk_overlap=kb_config.get('chunk_overlap', 200),
                                    chunking_strategy=kb_config.get('chunking_strategy', 'recursive_character')
                                )
                                
                                if chunks_processed > 0:
                                    st.success(f"✅ Successfully processed {chunks_processed} document chunks")
                                else:
                                    st.warning("⚠️ No documents were processed")
                            else:
                                st.info("ℹ️ No documents found. Knowledge base is ready but empty.")
                
                manual_text = kb_config.get('manual_text', '').strip()
                if manual_text:
                    st.info("📝 Processing manual text...")
                    text_chunks = self.knowledge_processor.process_manual_text(
                        manual_text=manual_text,
                        bot_id=bot_id,
                        chunk_size=kb_config.get('chunk_size', 1000),
                        chunk_overlap=kb_config.get('chunk_overlap', 200)
                    )
                    
                    chunks_processed += text_chunks
                    if text_chunks > 0:
                        st.success(f"✅ Processed {text_chunks} text chunks from manual input")
                
                test_success, test_message = self.knowledge_processor.test_vectorstore()
                if test_success:
                    st.success(f"🧪 Vectorstore test: {test_message}")
                else:
                    st.warning(f"🧪 Vectorstore test: {test_message}")
                
                bot_config['_kb_status'] = {
                    'initialized': True,
                    'embedding_model': embedding_model,
                    'document_count': len(kb_config.get('file_metadata', [])),
                    'chunks_processed': chunks_processed,
                    'manual_text_processed': bool(manual_text),
                    'vectorstore_type': 'Chroma',
                    'embedding_type': 'HuggingFace'
                }
                
                file_count = len(kb_config.get('file_metadata', []))
                total_content = file_count + (1 if manual_text else 0)
                
                if total_content > 0:
                    st.success(f"🎉 Knowledge base initialized successfully!")
                    st.info(f"📊 Summary: {chunks_processed} chunks from {total_content} content source(s)")
                else:
                    st.success("✅ Knowledge base initialized (empty - ready for content)")
                
                return True
                
        except Exception as e:
            st.error(f"❌ Error initializing knowledge base: {str(e)}")
            
            st.markdown("**Debug Information:**")
            st.code(f"""
Error Type: {type(e).__name__}
Error Message: {str(e)}
Bot ID: {bot_id}
KB Config Keys: {list(kb_config.keys()) if kb_config else []}
File Metadata: {len(kb_config.get('file_metadata', []))}
""")
            
            self.knowledge_processor = None
            return False