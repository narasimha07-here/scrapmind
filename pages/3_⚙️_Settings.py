import streamlit as st
import sys
import os
import json
from datetime import datetime
import hashlib
from components.data_manager import DataManager

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from utils.openrouter_client import OpenRouterClient
    OPENROUTER_AVAILABLE = True
except ImportError:
    OPENROUTER_AVAILABLE = False
    st.warning("OpenRouterClient not available")

st.set_page_config(
    page_title="Settings - Chatbot Builder",
    page_icon="⚙️",
    layout="wide"
)

def load_settings_css():
    st.markdown("""
    <style>
    .settings-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .settings-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .status-good {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .status-warning {
        background: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }

    .metric-display {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def get_data_manager_instance():
    if 'data_manager_instance' not in st.session_state:
        st.session_state.data_manager_instance = DataManager()
    return st.session_state.data_manager_instance

def initialize_user_config():
    if 'user_config' not in st.session_state:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id', 'default_user')
        default_config = dm.get_default_user_config(user_id)
        st.session_state.user_config = default_config

def get_data_manager():
    try:
        return get_data_manager_instance()
    except Exception as e:
        st.error(f"Failed to initialize data manager: {e}")
        return None

def save_user_config():
    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id')
        if user_id and dm:
            success = dm.save_user_config(user_id, st.session_state.user_config)
            if success:
                st.success("✅ Settings saved successfully!")
            return success
        return False
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return False

def rehydrate_session():
    if st.session_state.get('authenticated') and st.session_state.get('user_id'):
        dm = get_data_manager_instance()

        if not st.session_state.get('user_config'):
            st.session_state.user_config = dm.load_user_config(st.session_state.user_id)

        if not st.session_state.get('bots'):
            st.session_state.bots = dm.load_user_bots(st.session_state.user_id)
        if not st.session_state.get('chat_history'):
            st.session_state.chat_history = dm.load_chat_history(st.session_state.user_id)


def main():
    rehydrate_session()
    load_settings_css()
    initialize_user_config()

    if not st.session_state.get('authenticated', False):
        st.error("🔒 Please login first")
        if st.button("Go to Login", key="settings_login_redirect"):
            st.switch_page("app.py")
        return

    render_settings_page()

def render_settings_page():
    render_sidebar()

    st.markdown("""
    <div class="settings-header">
        <h1>⚙️ Settings & Configuration</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
            Manage your API keys, preferences, and account settings
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔑 API Keys",
        "🤖 Bot Defaults",
        "👤 Profile",
        "🎨 Appearance",
        "🔧 Advanced"
    ])

    with tab1:
        show_api_settings()

    with tab2:
        show_bot_defaults()

    with tab3:
        show_profile_settings()

    with tab4:
        show_appearance_settings()

    with tab5:
        show_advanced_settings()

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1.5rem;
        ">
            <h3 style="margin: 0;">👤 {st.session_state.get('username', 'User')}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Settings Panel</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Navigation")

        nav_buttons = [
            ("🏠 Home", "app.py"),
            ("🤖 Create Bot", "pages/1_🤖_Create_Bot.py"),
            ("📊 Dashboard", "pages/2_📊_Dashboard.py"),
            ("💬 Test Chat", "pages/4_🎯_Chat.py")
        ]

        for i, (label, page) in enumerate(nav_buttons):
            if st.button(label, key=f"settings_nav_{i}", use_container_width=True):
                st.switch_page(page)

        st.markdown("---")

        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh Models", key="settings_refresh_models", use_container_width=True):
            refresh_available_models()

        if st.button("💾 Save Settings", key="settings_save", use_container_width=True):
            if save_user_config():
                st.rerun()

        if st.button("📥 Import Settings", key="settings_import", use_container_width=True):
            st.session_state.show_import = True

        st.markdown("---")

        show_config_status_sidebar()

def show_config_status_sidebar():
    st.markdown("### 📊 Configuration Status")

    openrouter_status = "✅ Connected" if st.session_state.user_config.get('openrouter_api_key') else "❌ Not Set"
    openai_status = "✅ Connected" if st.session_state.user_config.get('openai_api_key') else "⚠️ Optional"

    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px;">
        <div><strong>OpenRouter:</strong> {openrouter_status}</div>
        <div><strong>OpenAI:</strong> {openai_status}</div>
        <div><strong>Bots:</strong> {len(st.session_state.get('bots', {}))}</div>
        <div><strong>Models:</strong> {len(st.session_state.get('available_models', []))}</div>
    </div>
    """, unsafe_allow_html=True)

def test_openai_key(api_key: str):
    try:
        if not api_key or not api_key.startswith('sk-'):
            st.error("❌ Invalid API key format. OpenAI keys should start with 'sk-'")
            return

        import requests

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        with st.spinner("🔄 Testing OpenAI connection..."):
            response = requests.get(
                'https://api.openai.com/v1/models',
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                models_data = response.json()
                model_count = len(models_data.get('data', []))
                st.success(f"✅ Valid OpenAI API key! Found {model_count} models")

                test_chat_payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Say 'API test successful'"}],
                    "max_tokens": 10
                }

                chat_response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=test_chat_payload,
                    timeout=10
                )

                if chat_response.status_code == 200:
                    st.success("✅ Chat completion test successful!")
                else:
                    st.warning(f"⚠️ Models accessible, but chat test failed: {chat_response.status_code}")

            elif response.status_code == 401:
                error_data = response.json()
                error_message = error_data.get('error', {}).get('message', 'Invalid API key')
                st.error(f"❌ Invalid API key: {error_message}")

            elif response.status_code == 403:
                st.error("❌ API key is valid but lacks necessary permissions")

            elif response.status_code == 429:
                st.error("❌ Rate limit exceeded. Please try again later.")

            else:
                st.error(f"❌ API test failed with status {response.status_code}")

    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. Please check your internet connection.")
    except requests.exceptions.ConnectionError:
        st.error("❌ Connection error. Please check your internet connection.")
    except Exception as e:
        st.error(f"❌ Test failed: {str(e)}")

def show_api_settings():
    st.markdown("## 🔑 API Configuration")
    st.markdown("Configure your API keys for AI models and services")

    show_api_status_overview()

    st.markdown("---")

    show_openrouter_config()

    st.markdown("---")

    show_openai_config()

    st.markdown("---")

    show_api_guidelines()

def show_api_status_overview():
    st.markdown("### 📊 Current API Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        openrouter_key = st.session_state.user_config.get('openrouter_api_key', '')
        if openrouter_key:
            st.markdown('<div class="status-good">✅ OpenRouter Connected</div>', unsafe_allow_html=True)
            st.caption(f"Key: ...{openrouter_key[-8:] if len(openrouter_key) > 8 else openrouter_key}")
        else:
            st.markdown('<div class="status-error">❌ OpenRouter Not Set</div>', unsafe_allow_html=True)

    with col2:
        openai_key = st.session_state.user_config.get('openai_api_key', '')
        if openai_key:
            st.markdown('<div class="status-good">✅ OpenAI Connected</div>', unsafe_allow_html=True)
            st.caption(f"Key: ...{openai_key[-8:] if len(openai_key) > 8 else openai_key}")
        else:
            st.markdown('<div class="status-warning">⚠️ OpenAI Optional</div>', unsafe_allow_html=True)
            st.caption("Will use OpenRouter")

    with col3:
        available_models = len(st.session_state.get('available_models', []))
        if available_models > 0:
            st.markdown(f'<div class="status-good">📊 {available_models} Models</div>', unsafe_allow_html=True)
            st.caption("Models loaded successfully")
        else:
            st.markdown('<div class="status-warning">📊 No Models</div>', unsafe_allow_html=True)
            st.caption("Load models from API")

def show_openai_config():
    st.markdown("### 🧠 OpenAI API Configuration (Optional)")

    with st.container():
        st.markdown("""
        <div class="settings-card">
        """, unsafe_allow_html=True)

        st.markdown("**Optional: Use direct OpenAI API for embeddings (can also use OpenRouter)**")

        col1, col2 = st.columns([3, 1])

        with col1:
            current_openai_key = st.session_state.user_config.get('openai_api_key', '')

            if current_openai_key:
                masked_display = 'sk-' + '*' * (len(current_openai_key) - 10) + current_openai_key[-8:] if len(current_openai_key) > 10 else current_openai_key
                st.text_input(
                    "Current OpenAI API Key",
                    value=masked_display,
                    disabled=True,
                    help="Current key (masked for security)"
                )

            new_openai_key = st.text_input(
                "Enter New OpenAI API Key (Optional)",
                value='',
                type="password",
                help="Enter your OpenAI API key (starts with sk-)",
                key="new_openai_key_input",
                placeholder="sk-..."
            )

        with col2:
            if st.button("🧪 Test Key", key="test_openai_btn"):
                key_to_test = new_openai_key if new_openai_key else current_openai_key
                if key_to_test:
                    test_openai_key(key_to_test)
                else:
                    st.warning("Please enter an API key to test")

        if new_openai_key:
            col_save1, col_save2 = st.columns([1, 1])
            with col_save1:
                if st.button("💾 Save New Key", key="save_new_openai", type="primary"):
                    st.session_state.user_config['openai_api_key'] = new_openai_key
                    if save_user_config():
                        st.rerun()

            with col_save2:
                if st.button("🗑️ Remove Key", key="remove_openai"):
                    st.session_state.user_config['openai_api_key'] = ''
                    if save_user_config():
                        st.rerun()

        elif current_openai_key:
            if st.button("🗑️ Remove Current Key", key="remove_current_openai"):
                st.session_state.user_config['openai_api_key'] = ''
                if save_user_config():
                    st.rerun()

        with st.expander("💡 How to get OpenAI API Key"):
            st.markdown("""
            **Steps to get your OpenAI API Key:**

            1. **Visit** [platform.openai.com](https://platform.openai.com)
            2. **Sign up** or log in to your account
            3. **Go to API Keys** in your dashboard
            4. **Create new secret key**
            5. **Copy the key** (starts with sk-) and paste it here

            **Important Notes:**
            - Keep your API key secure and never share it
            - OpenAI charges for API usage based on tokens
            - You can set usage limits in your OpenAI dashboard
            - Free tier has limited credits
            """)

        st.markdown("</div>", unsafe_allow_html=True)

def show_openrouter_config():
    st.markdown("### 🌐 OpenRouter API Configuration")

    with st.container():
        st.markdown("""
        <div class="settings-card">
        """, unsafe_allow_html=True)

        st.markdown("**OpenRouter provides access to 100+ AI models including free options**")

        col1, col2 = st.columns([3, 1])

        with col1:
            current_key = st.session_state.user_config.get('openrouter_api_key', '')
            masked_key = ('*' * (len(current_key) - 8) + current_key[-8:]
                         if len(current_key) > 8 else current_key)

            new_openrouter_key = st.text_input(
                "OpenRouter API Key *",
                value=masked_key if current_key else '',
                type="password",
                help="Get your API key from openrouter.ai",
                key="openrouter_key_input"
            )

        with col2:
            if st.button("🧪 Test Key", key="test_openrouter_btn"):
                test_key = new_openrouter_key if new_openrouter_key != masked_key else current_key
                if test_key:
                    test_openrouter_key(test_key)
                else:
                    st.error("Please enter an API key to test")

        if new_openrouter_key and new_openrouter_key != masked_key:
            if st.button("💾 Save OpenRouter Key", key="save_openrouter", type="primary"):
                st.session_state.user_config['openrouter_api_key'] = new_openrouter_key
                if save_user_config():
                    with st.spinner("🔄 Loading available models..."):
                        refresh_available_models()
                    st.rerun()

        with st.expander("💡 How to get OpenRouter API Key"):
            st.markdown("""
            **Steps to get your OpenRouter API Key:**

            1. **Visit** [openrouter.ai](https://openrouter.ai)
            2. **Sign up** for a free account
            3. **Verify** your email address
            4. **Add credits** (some models are free)
            5. **Go to API Keys** in your dashboard
            6. **Create new key** and copy it here

            **Free Models Available:**
            - Llama 3.2 3B (Free)
            - Zephyr 7B Beta (Free)
            - OpenChat 7B (Free)
            - And many more!
            """)

        st.markdown("</div>", unsafe_allow_html=True)

def show_api_guidelines():
    with st.expander("📋 API Usage Guidelines & Best Practices", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 🌟 OpenRouter Best Practices:
            - **Free Models**: Look for `:free` suffix
            - **Rate Limits**: Respect API rate limits
            - **Cost Management**: Monitor usage for paid models
            - **Security**: Never share API keys publicly
            - **Testing**: Use free models for development

            ### 🆓 Recommended Free Models:
            - `meta-llama/llama-3.2-3b-instruct:free`
            - `huggingfaceh4/zephyr-7b-beta:free`
            - `openchat/openchat-7b:free`
            - `gryphe/mythomist-7b:free`
            """)

        with col2:
            st.markdown("""
            ### 🔐 Security Guidelines:
            - Store keys securely
            - Use environment variables in production
            - Rotate keys regularly
            - Monitor usage patterns
            - Report suspicious activity

            ### 💰 Cost Optimization:
            - Start with free models
            - Set usage limits
            - Monitor spending
            - Use caching when possible
            - Choose appropriate models for tasks
            """)

def test_openrouter_key(api_key: str):
    if not OPENROUTER_AVAILABLE:
        st.error("❌ OpenRouterClient not available")
        return

    try:
        with st.spinner("🔄 Testing OpenRouter connection..."):
            client = OpenRouterClient(api_key)
            models = client.get_available_models()

            if models and len(models) > 0:
                free_models = len([m for m in models if m.get('is_free')])
                st.success(f"✅ Connection successful! Found {len(models)} models ({free_models} free)")

                test_result = client.test_connection()
                if test_result['success']:
                    st.success("✅ Chat completion test successful!")
                else:
                    st.warning(f"⚠️ Chat test warning: {test_result['message']}")

            else:
                st.error("❌ Invalid API key or no models available")

    except Exception as e:
        st.error(f"❌ Connection test failed: {str(e)}")


def refresh_available_models():
    if not OPENROUTER_AVAILABLE:
        st.error("OpenRouterClient not available")
        return

    api_key = st.session_state.user_config.get('openrouter_api_key')

    if api_key:
        try:
            client = OpenRouterClient(api_key)
            models = client.get_available_models()
            st.session_state.available_models = models
            st.success(f"✅ Refreshed {len(models)} models successfully!")
        except Exception as e:
            st.error(f"❌ Failed to refresh models: {str(e)}")
    else:
        st.error("❌ No OpenRouter API key configured")

def show_bot_defaults():
    st.markdown("## 🤖 Default Bot Settings")
    st.markdown("Set default values that will be pre-filled when creating new bots")

    with st.container():
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)

        st.markdown("### 🧠 AI Model Defaults")

        col1, col2 = st.columns(2)

        with col1:
            available_models = st.session_state.get('available_models', [])

            if available_models:
                model_options = [model['id'] for model in available_models]
                model_names = {
                    model['id']: f"{model['name']} {'🆓' if model.get('is_free') else '💰'}"
                    for model in available_models
                }

                current_default = st.session_state.user_config.get('default_model', model_options[0])
                default_index = model_options.index(current_default) if current_default in model_options else 0

                new_default_model = st.selectbox(
                    "Default AI Model",
                    model_options,
                    index=default_index,
                    format_func=lambda x: model_names.get(x, x),
                    help="This model will be pre-selected when creating new bots",
                    key="default_model_select"
                )

                if new_default_model != current_default:
                    st.session_state.user_config['default_model'] = new_default_model
                    if save_user_config():
                        st.rerun()
            else:
                st.warning("⚠️ No models available. Please configure OpenRouter API key.")

            new_temperature = st.slider(
                "Default Creativity (Temperature)",
                0.0, 1.0,
                float(st.session_state.user_config.get('default_temperature', 0.7)),
                0.1,
                help="Higher values make responses more creative",
                key="default_temperature_slider"
            )

            if new_temperature != st.session_state.user_config.get('default_temperature'):
                st.session_state.user_config['default_temperature'] = new_temperature
                save_user_config()

        with col2:
            new_max_tokens = st.number_input(
                "Default Max Response Length",
                100, 4000,
                int(st.session_state.user_config.get('default_max_tokens', 1000)),
                step=100,
                help="Maximum length of bot responses",
                key="default_max_tokens_input"
            )

            if new_max_tokens != st.session_state.user_config.get('default_max_tokens'):
                st.session_state.user_config['default_max_tokens'] = new_max_tokens
                save_user_config()

            new_context_window = st.number_input(
                "Default Context Window",
                1, 20,
                int(st.session_state.user_config.get('default_context_window', 5)),
                help="Number of previous messages to remember",
                key="default_context_input"
            )

            if new_context_window != st.session_state.user_config.get('default_context_window'):
                st.session_state.user_config['default_context_window'] = new_context_window
                save_user_config()

        st.markdown("### 🎭 Bot Behavior Defaults")

        col1, col2 = st.columns(2)

        with col1:
            personalities = [
                "Professional", "Friendly", "Casual", "Technical",
                "Creative", "Helpful", "Witty", "Empathetic", "Direct"
            ]

            new_personality = st.selectbox(
                "Default Personality",
                personalities,
                index=personalities.index(st.session_state.user_config.get('default_personality', 'Professional')),
                help="Default personality for new bots",
                key="default_personality_select"
            )

            if new_personality != st.session_state.user_config.get('default_personality'):
                st.session_state.user_config['default_personality'] = new_personality
                save_user_config()

        with col2:
            new_welcome = st.text_area(
                "Default Welcome Message",
                value=st.session_state.user_config.get('default_welcome', 'Hello! I am your AI assistant. How can I help you today?'),
                help="Default welcome message for new bots",
                key="default_welcome_input"
            )

            if new_welcome != st.session_state.user_config.get('default_welcome'):
                st.session_state.user_config['default_welcome'] = new_welcome
                save_user_config()

        st.markdown("### 📚 Knowledge Base Defaults")

        col1, col2, col3 = st.columns(3)

        with col1:
            chunking_options = ["recursive_character", "character", "token", "semantic"]
            chunking_names = {
                "recursive_character": "Smart Chunking (Recommended)",
                "character": "Character-based",
                "token": "Token-based",
                "semantic": "Semantic"
            }

            new_chunking = st.selectbox(
                "Default Chunking Strategy",
                chunking_options,
                index=chunking_options.index(st.session_state.user_config.get('default_chunking', 'recursive_character')),
                format_func=lambda x: chunking_names.get(x, x),
                help="How to split documents into chunks",
                key="default_chunking_select"
            )

            if new_chunking != st.session_state.user_config.get('default_chunking'):
                st.session_state.user_config['default_chunking'] = new_chunking
                save_user_config()

        with col2:
            new_chunk_size = st.number_input(
                "Default Chunk Size",
                100, 2000,
                int(st.session_state.user_config.get('default_chunk_size', 1000)),
                step=100,
                help="Default size for text chunks",
                key="default_chunk_size_input"
            )

            if new_chunk_size != st.session_state.user_config.get('default_chunk_size'):
                st.session_state.user_config['default_chunk_size'] = new_chunk_size
                save_user_config()

        with col3:
            new_chunk_overlap = st.number_input(
                "Default Chunk Overlap",
                0, 500,
                int(st.session_state.user_config.get('default_chunk_overlap', 200)),
                step=50,
                help="Overlap between chunks",
                key="default_overlap_input"
            )

            if new_chunk_overlap != st.session_state.user_config.get('default_chunk_overlap'):
                st.session_state.user_config['default_chunk_overlap'] = new_chunk_overlap
                save_user_config()

        embedding_options = [
            "openai/text-embedding-ada-002",
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large"
            "sentence-transformers/all-MiniLM-L6-v2"
        ]

        current_embedding = st.session_state.user_config.get('default_embedding')

        if current_embedding and current_embedding not in embedding_options:
            embedding_options.insert(0, current_embedding)
            default_index = 0
        elif current_embedding:
            default_index = embedding_options.index(current_embedding)
        else:
            current_embedding = embedding_options[0]
            default_index = 0

        new_embedding = st.selectbox(
            "Default Embedding Model",
            embedding_options,
            index=default_index,
            help="Model used to create document embeddings",
            key="default_embedding_select"
        )

        if new_embedding != st.session_state.user_config.get('default_embedding'):
            st.session_state.user_config['default_embedding'] = new_embedding
            save_user_config()


        st.markdown("</div>", unsafe_allow_html=True)

def show_profile_settings():
    st.markdown("## 👤 Profile & Account Settings")

    with st.container():
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)

        st.markdown("### 📝 Personal Information")

        col1, col2 = st.columns(2)

        with col1:
            current_username = st.session_state.get('username', '')
            new_username = st.text_input(
                "Display Name",
                value=current_username,
                help="Your display name in the app",
                key="profile_username_input"
            )

            user_email = st.text_input(
                "Email Address",
                value=st.session_state.get('user_email', ''),
                help="Your email address for notifications",
                key="profile_email_input"
            )

        with col2:
            timezone_options = [
                "UTC", "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
                "Europe/London", "Europe/Paris", "Europe/Berlin", "Asia/Tokyo",
                "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney"
            ]

            user_timezone = st.selectbox(
                "Timezone",
                timezone_options,
                index=timezone_options.index(st.session_state.get('user_timezone', 'UTC')),
                help="Your local timezone",
                key="profile_timezone_select"
            )

            language_options = [
                "English", "Spanish", "French", "German", "Italian",
                "Portuguese", "Russian", "Japanese", "Korean", "Chinese (Simplified)"
            ]

            user_language = st.selectbox(
                "Language",
                language_options,
                index=language_options.index(st.session_state.get('user_language', 'English')),
                help="Preferred language for the interface",
                key="profile_language_select"
            )

        if st.button("💾 Save Profile Changes", key="save_profile", type="primary"):
            st.session_state.username = new_username
            st.session_state.user_email = user_email
            st.session_state.user_timezone = user_timezone
            st.session_state.user_language = user_language

            st.session_state.user_config['username'] = new_username

            if save_user_config():
                st.success("✅ Profile updated successfully!")

        st.markdown("### 📊 Account Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_bots = len(st.session_state.get('bots', {}))
            st.markdown(f'<div class="metric-display"><strong>{total_bots}</strong><br>Total Bots</div>', unsafe_allow_html=True)

        with col2:
            total_chats = sum(len(chats) for chats in st.session_state.get('chat_history', {}).values())
            st.markdown(f'<div class="metric-display"><strong>{total_chats}</strong><br>Total Messages</div>', unsafe_allow_html=True)

        with col3:
            member_since = st.session_state.get('member_since', datetime.now().strftime('%Y-%m-%d'))
            st.markdown(f'<div class="metric-display"><strong>{member_since}</strong><br>Member Since</div>', unsafe_allow_html=True)

        with col4:
            last_active = datetime.now().strftime('%Y-%m-%d')
            st.markdown(f'<div class="metric-display"><strong>{last_active}</strong><br>Last Active</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

def show_appearance_settings():
    st.markdown("## 🎨 Appearance & Interface")

    with st.container():
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)

        st.markdown("### 🎨 Theme & Display")

        col1, col2 = st.columns(2)

        with col1:
            theme_options = ["Auto", "Light", "Dark"]
            theme_preference = st.selectbox(
                "Color Theme",
                theme_options,
                index=theme_options.index(st.session_state.get('theme_preference', 'Auto')),
                help="Choose your preferred color theme",
                key="theme_preference_select"
            )
            st.session_state.theme_preference = theme_preference

            enable_animations = st.checkbox(
                "Enable Animations",
                value=st.session_state.get('enable_animations', True),
                help="Show smooth transitions and loading animations",
                key="animations_checkbox"
            )
            st.session_state.enable_animations = enable_animations

            default_debug = st.checkbox(
                "Enable Debug Mode by Default",
                value=st.session_state.get('default_debug_mode', False),
                help="Show debug information in chat interfaces",
                key="debug_mode_checkbox"
            )
            st.session_state.default_debug_mode = default_debug

        with col2:
            compact_mode = st.checkbox(
                "Compact Mode",
                value=st.session_state.get('compact_mode', False),
                help="Use smaller spacing and compact layouts",
                key="compact_mode_checkbox"
            )
            st.session_state.compact_mode = compact_mode

            auto_save = st.checkbox(
                "Auto-save Bot Drafts",
                value=st.session_state.get('auto_save_drafts', True),
                help="Automatically save bot creation progress",
                key="auto_save_checkbox"
            )
            st.session_state.auto_save_drafts = auto_save

            show_tips = st.checkbox(
                "Show Helpful Tips",
                value=st.session_state.get('show_tips', True),
                help="Display helpful tips and guidance throughout the app",
                key="show_tips_checkbox"
            )
            st.session_state.show_tips = show_tips

        st.markdown("### 📱 Chat Interface")

        col1, col2 = st.columns(2)

        with col1:
            bubble_styles = ["Modern", "Classic", "Minimal", "Bubble", "Card"]
            message_style = st.selectbox(
                "Message Style",
                bubble_styles,
                index=bubble_styles.index(st.session_state.get('message_style', 'Modern')),
                help="Visual style for chat messages",
                key="message_style_select"
            )
            st.session_state.message_style = message_style

            typing_indicators = st.checkbox(
                "Show Typing Indicators",
                value=st.session_state.get('typing_indicators', True),
                help="Show typing animations when AI is responding",
                key="typing_indicators_checkbox"
            )
            st.session_state.typing_indicators = typing_indicators

        with col2:
            show_timestamps = st.checkbox(
                "Show Message Timestamps",
                value=st.session_state.get('show_timestamps', True),
                help="Display timestamps on chat messages",
                key="timestamps_checkbox"
            )
            st.session_state.show_timestamps = show_timestamps

            show_avatars = st.checkbox(
                "Show User/AI Avatars",
                value=st.session_state.get('show_avatars', True),
                help="Display profile pictures in chat",
                key="avatars_checkbox"
            )
            st.session_state.show_avatars = show_avatars

        if st.button("💾 Save Appearance Settings", key="save_appearance", type="primary"):
            appearance_settings = {
                'theme_preference': theme_preference,
                'enable_animations': enable_animations,
                'default_debug_mode': default_debug,
                'compact_mode': compact_mode,
                'auto_save_drafts': auto_save,
                'show_tips': show_tips,
                'message_style': message_style,
                'typing_indicators': typing_indicators,
                'show_timestamps': show_timestamps,
                'show_avatars': show_avatars
            }

            for key, value in appearance_settings.items():
                st.session_state.user_config[key] = value

            if save_user_config():
                st.success("✅ Appearance settings saved successfully!")

        st.markdown("</div>", unsafe_allow_html=True)

def show_advanced_settings():
    st.markdown("## 🔧 Advanced Settings")

    with st.container():
        st.markdown('<div class="settings-card">', unsafe_allow_html=True)

        st.markdown("### ⚙️ System Configuration")

        col1, col2 = st.columns(2)

        with col1:
            cache_enabled = st.checkbox(
                "Enable Response Caching",
                value=st.session_state.get('cache_enabled', True),
                help="Cache AI responses to reduce API calls",
                key="cache_checkbox"
            )
            st.session_state.cache_enabled = cache_enabled

            cache_duration = st.slider(
                "Cache Duration (hours)",
                1, 168,
                int(st.session_state.get('cache_duration', 24)),
                help="How long to cache responses before refreshing",
                key="cache_duration_slider"
            )
            st.session_state.cache_duration = cache_duration

        with col2:
            api_timeout = st.number_input(
                "API Timeout (seconds)",
                10, 300,
                int(st.session_state.get('api_timeout', 30)),
                help="Timeout for API requests",
                key="api_timeout_input"
            )
            st.session_state.api_timeout = api_timeout

            retry_attempts = st.number_input(
                "API Retry Attempts",
                0, 5,
                int(st.session_state.get('retry_attempts', 2)),
                help="Number of retry attempts for failed API calls",
                key="retry_attempts_input"
            )
            st.session_state.retry_attempts = retry_attempts

        st.markdown("### 🔍 Developer Options")

        col1, col2 = st.columns(2)

        with col1:
            debug_logging = st.checkbox(
                "Enable Debug Logging",
                value=st.session_state.get('debug_logging', False),
                help="Enable detailed debug logging for troubleshooting",
                key="debug_logging_checkbox"
            )
            st.session_state.debug_logging = debug_logging

            api_logging = st.checkbox(
                "Log API Responses",
                value=st.session_state.get('api_logging', False),
                help="Log full API responses (may contain sensitive data)",
                key="api_logging_checkbox"
            )
            st.session_state.api_logging = api_logging

        with col2:
            performance_metrics = st.checkbox(
                "Collect Performance Metrics",
                value=st.session_state.get('performance_metrics', True),
                help="Collect anonymous usage metrics to improve the app",
                key="performance_metrics_checkbox"
            )
            st.session_state.performance_metrics = performance_metrics

            error_reporting = st.checkbox(
                "Enable Error Reporting",
                value=st.session_state.get('error_reporting', True),
                help="Automatically report errors to help improve the app",
                key="error_reporting_checkbox"
            )
            st.session_state.error_reporting = error_reporting

        st.markdown("### 🗑️ Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📤 Export All Data", key="export_data_btn", use_container_width=True):
                export_user_data()

            if st.button("📥 Import Data", key="import_data_btn", use_container_width=True):
                st.session_state.show_import = True

        with col2:
            if st.button("🗑️ Clear Cache", key="clear_cache_btn", use_container_width=True):
                clear_cache()

            if st.button("🔄 Reset to Defaults", key="reset_settings_btn", use_container_width=True):
                reset_to_defaults()

        with st.expander("⚠️ Dangerous Zone", expanded=False):
            st.warning("⚠️ These actions cannot be undone!")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🗑️ Delete All Bots", key="delete_bots_btn", type="secondary"):
                    confirm_delete_bots()

            with col2:
                if st.button("🔥 Delete All Data", key="delete_all_btn", type="primary"):
                    confirm_delete_all_data()

        st.markdown("</div>", unsafe_allow_html=True)

def export_user_data():
    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id')

        if user_id and dm:
            export_data = dm.export_user_data(user_id)

            st.download_button(
                label="📥 Download Data Export",
                data=json.dumps(export_data, indent=2),
                file_name=f"chatbot_builder_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_export"
            )

            st.success("✅ Data export ready for download!")
        else:
            st.error("❌ User not authenticated or data manager not available")
    except Exception as e:
        st.error(f"❌ Failed to export data: {str(e)}")

def clear_cache():
    try:
        cache_keys = [k for k in st.session_state.keys() if k.startswith('cache_')]
        for key in cache_keys:
            del st.session_state[key]

        dm = get_data_manager_instance()
        if dm and hasattr(dm, 'clear_cache'):
            dm.clear_cache(st.session_state.user_id)

        st.success("✅ Cache cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Failed to clear cache: {str(e)}")

def reset_to_defaults():
    if st.button("⚠️ Confirm Reset All Settings", key="confirm_reset", type="primary"):
        try:
            dm = get_data_manager_instance()
            user_id = st.session_state.get('user_id')

            if user_id and dm:
                default_config = dm.get_default_user_config(user_id)
                st.session_state.user_config = default_config

                dm.save_user_config(user_id, default_config)

                st.success("✅ Settings reset to defaults!")
                st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to reset settings: {str(e)}")

def confirm_delete_bots():
    if st.button("⚠️ Confirm Delete All Bots", key="confirm_delete_bots", type="primary"):
        try:
            dm = get_data_manager_instance()
            user_id = st.session_state.get('user_id')

            if user_id and dm:
                success = dm.delete_all_bots(user_id)
                if success:
                    st.session_state.bots = {}
                    st.success("✅ All bots deleted successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete bots")
        except Exception as e:
            st.error(f"❌ Failed to delete bots: {str(e)}")

def confirm_delete_all_data():
    if st.button("⚠️ Confirm Delete ALL Data", key="confirm_delete_all", type="primary"):
        try:
            dm = get_data_manager_instance()
            user_id = st.session_state.get('user_id')

            if user_id and dm:
                success = dm.delete_all_user_data(user_id)
                if success:
                    for key in list(st.session_state.keys()):
                        if key not in ['authenticated', 'user_id', 'username', 'user_email']:
                            del st.session_state[key]

                    initialize_user_config()

                    st.success("✅ All data deleted successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete data")
        except Exception as e:
            st.error(f"❌ Failed to delete data: {str(e)}")

if __name__ == "__main__":
    main()

