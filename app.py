__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(__file__))

st.set_page_config(
    page_title="No-Code Chatbot Builder",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'data_manager' not in st.session_state:
        try:
            from components.data_manager import DataManager
            st.session_state.data_manager = DataManager()
        except ImportError:
            st.error("DataManager not found. Please ensure components/data_manager.py exists.")
            st.stop()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if st.session_state.authenticated and st.session_state.user_id:
        load_user_data()
    else:
        initialize_default_session_state()
    
    if 'current_bot' not in st.session_state:
        st.session_state.current_bot = None
    if 'available_models' not in st.session_state:
        st.session_state.available_models = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"

def load_user_data():
    try:
        user_id = st.session_state.user_id
        data_manager = st.session_state.data_manager
        
        st.session_state.user_config = data_manager.load_user_config(user_id)
        
        st.session_state.bots = data_manager.load_user_bots(user_id)
        
        st.session_state.chat_history = data_manager.load_chat_history(user_id)
        
        if st.session_state.user_config.get('username') and not st.session_state.username:
            st.session_state.username = st.session_state.user_config['username']
            
    except Exception as e:
        st.error(f"Error loading user data: {e}")
        initialize_default_session_state()

def initialize_default_session_state():
    if 'user_config' not in st.session_state:
        st.session_state.user_config = get_default_user_config()
    if 'bots' not in st.session_state:
        st.session_state.bots = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

def get_default_user_config():
    return {
        'openai_api_key': '', 
        'default_model': 'meta-llama/llama-3.2-3b-instruct:free',
        'default_embedding': 'sentence-transformers/all-MiniLM-L6-v2',
        'default_chunking': 'recursive_character',
        'default_chunk_size': 1000,
        'default_chunk_overlap': 200,
        'default_personality': 'Professional',
        'default_temperature': 0.7,
        'default_max_tokens': 1000,
        'default_context_window': 5,
        'created_at': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'version': '1.2'
    }

def save_user_data():
    if not st.session_state.get('authenticated') or not st.session_state.get('user_id'):
        return False
        
    try:
        data_manager = st.session_state.data_manager
        user_id = st.session_state.user_id
        
        success = True
        success &= data_manager.save_user_config(user_id, st.session_state.get('user_config', {}))
        success &= data_manager.save_user_bots(user_id, st.session_state.get('bots', {}))
        success &= data_manager.save_chat_history(user_id, st.session_state.get('chat_history', {}))
        
        if success:
            st.success("✅ Data saved successfully!", icon="💾")
        else:
            st.error("❌ Failed to save some data")
            
        return success
        
    except Exception as e:
        st.error(f"Error saving user data: {e}")
        return False

def trigger_save():
    if st.session_state.get('authenticated'):
        save_user_data()

def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .nav-item {
        padding: 0.75rem 1.5rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    
    .nav-item:hover {
        background: #e9ecef;
        transform: translateX(5px);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
    }
    
    .custom-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .status-active {
        background: #d4edda;
        color: #155724;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-draft {
        background: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-testing {
        background: #cce5f0;
        color: #055160;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .save-indicator {
        position: fixed;
        top: 80px;
        right: 20px;
        background: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        z-index: 1000;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(100px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    initialize_session_state()
    load_custom_css()

    if not st.session_state.authenticated:
        show_auth_page()
        return

    show_main_dashboard()

def show_auth_page():
    st.markdown("""
    <div class="main-header">
        <h1>🤖 No-Code Chatbot Builder</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0;">
            Create intelligent AI chatbots without writing any code
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from components.auth import show_authentication
        show_authentication()
    except ImportError:
        show_simple_auth()

def show_simple_auth():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.subheader("Welcome Back!")
            username = st.text_input("Username", key="login_user")
            email = st.text_input("Email (optional)", key="login_email")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", type="primary", use_container_width=True):
                if username and password:
                    authenticate_user(username, email, password, is_new_user=False)
                else:
                    st.error("Please enter both username and password")
        
        with tab2:
            st.subheader("Join Us!")
            new_username = st.text_input("Choose Username", key="signup_user")
            email = st.text_input("Email Address", key="signup_email") 
            new_password = st.text_input("Password", type="password", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            
            if st.button("Create Account", type="primary", use_container_width=True):
                if new_username and email and new_password and confirm_password:
                    if new_password == confirm_password:
                        authenticate_user(new_username, email, new_password, is_new_user=True)
                    else:
                        st.error("Passwords don't match")
                else:
                    st.error("Please fill in all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)

def authenticate_user(username: str, email: str, password: str, is_new_user: bool = False):
    try:
        data_manager = st.session_state.data_manager
        
        user_id = data_manager.get_user_id(username, email)
        
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = user_id
        
        if is_new_user:
            st.success(f"Account created successfully! Welcome, {username}!")
            st.session_state.user_config = get_default_user_config()
            st.session_state.user_config['username'] = username
            st.session_state.bots = {}
            st.session_state.chat_history = {}
            
            save_user_data()
        else:
            load_user_data()
            st.success(f"Welcome back, {username}!")
        
        import time
        time.sleep(1)
        st.rerun()
        
    except Exception as e:
        st.error(f"Authentication failed: {e}")

def show_main_dashboard():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-bottom: 2rem;">
            <h3 style="margin: 0;">Welcome, {st.session_state.username}!</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">ID: {st.session_state.user_id[:12]}...</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Navigation")
        
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
        
        if st.button("🤖 Create Bot", use_container_width=True):
            st.switch_page("pages/1_🤖_Create_Bot.py")
        
        if st.button("📊 My Bots", use_container_width=True):
            st.switch_page("pages/2_📊_Dashboard.py")
        
        if st.button("⚙️ Settings", use_container_width=True):
            st.switch_page("pages/3_⚙️_Settings.py")
        
        if st.button("💬 Test Chat", use_container_width=True):
            st.switch_page("pages/4_🎯_Chat.py")
        
        st.markdown("---")
        
        st.markdown("### 📈 Quick Stats")
        display_user_stats()
        
        st.markdown("---")
        
        st.markdown("### 💾 Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save", help="Manually save all data"):
                with st.spinner("Saving..."):
                    if save_user_data():
                        st.success("Saved!")
                    else:
                        st.error("Save failed!")
        
        with col2:
            if st.button("🔄 Reload", help="Reload data from storage"):
                with st.spinner("Reloading..."):
                    load_user_data()
                    st.success("Reloaded!")
                    st.rerun()
        
        if st.session_state.get('data_manager'):
            st.success("🔄 Auto-save enabled", icon="✅")
        else:
            st.error("❌ Data manager not loaded", icon="⚠️")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", type="secondary"):
            save_user_data()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Chatbot Builder Dashboard</h1>
        <p>Build, deploy, and manage your AI chatbots with persistent data storage</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🚀 Quick Start</h3>
            <p>Create your first chatbot in minutes with our no-code builder</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Create New Bot", type="primary", use_container_width=True, key="create_new"):
            st.switch_page("pages/1_🤖_Create_Bot.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📚 Knowledge Base</h3>
            <p>Upload documents and create intelligent, context-aware responses</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Manage Knowledge", use_container_width=True, key="manage_knowledge"):
            st.switch_page("pages/2_📊_Dashboard.py")
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Test & Deploy</h3>
            <p>Test your bots and deploy them with shareable links</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Test Bots", use_container_width=True, key="test_bots"):
            st.switch_page("pages/4_🎯_Chat.py")
    
    display_recent_bots()
    
    display_storage_info()

def display_user_stats():
    try:
        data_manager = st.session_state.data_manager
        stats = data_manager.get_user_stats(st.session_state.user_id)
        
        total_bots = stats.get('total_bots', 0)
        active_bots = stats.get('active_bots', 0)
        total_files = stats.get('total_uploaded_files', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_bots}</h3>
            <p>Total Bots</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{active_bots}</h3>
            <p>Active Bots</p>
        </div>
        """, unsafe_allow_html=True)
        
        if total_files > 0:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                <h3>{total_files}</h3>
                <p>Uploaded Files</p>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error loading stats: {e}")

def display_recent_bots():
    if st.session_state.bots:
        st.markdown("## 🤖 Your Recent Bots")
        
        sorted_bots = sorted(
            st.session_state.bots.items(),
            key=lambda x: x[1].get('updated_at', x[1].get('created_at', '')),
            reverse=True
        )
        
        for bot_id, bot_config in sorted_bots[:3]:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    status = bot_config.get('status', 'draft')
                    status_class = f"status-{status}"
                    
                    file_count = ""
                    if bot_config.get('knowledge_base', {}).get('enabled'):
                        files = len(bot_config.get('knowledge_base', {}).get('file_metadata', []))
                        file_count = f" • {files} files"
                    
                    st.markdown(f"""
                    <div style="padding: 1rem; background: white; border-radius: 10px; border: 1px solid #e1e5e9; margin: 0.5rem 0;">
                        <h4 style="margin: 0 0 0.5rem 0;">🤖 {bot_config['name']}</h4>
                        <p style="margin: 0 0 0.5rem 0; color: #666;">{bot_config.get('description', 'No description')[:100]}...{file_count}</p>
                        <span class="{status_class}">{status.title()}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{bot_id}"):
                        st.session_state.current_bot = bot_id
                        st.switch_page("pages/1_🤖_Create_Bot.py")
                
                with col3:
                    if st.button("💬 Chat", key=f"chat_{bot_id}"):
                        st.session_state.current_bot = bot_id
                        st.switch_page("pages/4_🎯_Chat.py")
                
                with col4:
                    if st.button("📊 Stats", key=f"stats_{bot_id}"):
                        st.session_state.current_bot = bot_id
                        st.switch_page("pages/2_📊_Dashboard.py")
    else:
        st.markdown("""
        <div class="feature-card" style="text-align: center; padding: 3rem;">
            <h2>🌟 Ready to get started?</h2>
            <p>You haven't created any chatbots yet. Let's build your first one!</p>
            <p><small>💾 All your data will be automatically saved and persist across sessions</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Create Your First Bot", type="primary", use_container_width=True, key="first_bot"):
            st.switch_page("pages/1_🤖_Create_Bot.py")

def display_storage_info():
    try:
        data_manager = st.session_state.data_manager
        usage = data_manager.get_storage_usage(st.session_state.user_id)
        
        if usage:
            from components.data_manager import format_file_size
            
            with st.expander("💾 Storage Usage", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Storage", format_file_size(usage.get('total_size', 0)))
                
                with col2:
                    st.metric("Bot Configs", format_file_size(usage.get('bots_size', 0)))
                
                with col3:
                    st.metric("Uploaded Files", format_file_size(usage.get('files_size', 0)))
                
                st.markdown("**Storage Breakdown:**")
                st.progress(0.3)
                st.caption(f"Config: {format_file_size(usage.get('config_size', 0))} | "
                          f"Chats: {format_file_size(usage.get('chat_size', 0))} | "
                          f"Files: {format_file_size(usage.get('files_size', 0))}")
                
    except Exception as e:
        st.error(f"Error loading storage info: {e}")

def on_data_change():
    if st.session_state.get('authenticated'):
        save_user_data()

if __name__ == "__main__":
    main()
