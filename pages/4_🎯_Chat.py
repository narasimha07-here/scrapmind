import streamlit as st
import sys
import os
import json
from datetime import datetime
from typing import Dict
from components.data_manager import DataManager, auto_save_user_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from components.chat_interface import ChatInterface
    CHAT_INTERFACE_AVAILABLE = True
except ImportError as e:
    st.warning(f"ChatInterface not available: {str(e)}")
    CHAT_INTERFACE_AVAILABLE = False

st.set_page_config(
    page_title="Chat - Chatbot Builder",
    page_icon="💬",
    layout="wide"
)

def load_chat_css():
    st.markdown("""
    <style>
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .bot-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .bot-selection-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e1e5e9;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .bot-selection-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .chat-stats {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }

    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

def get_data_manager_instance():
    if 'data_manager_instance' not in st.session_state:
        st.session_state.data_manager_instance = DataManager()
    return st.session_state.data_manager_instance

def initialize_chat_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if 'user_id' not in st.session_state:
        st.session_state.user_id = None

    if 'username' not in st.session_state:
        st.session_state.username = None

    if 'bots' not in st.session_state:
        st.session_state.bots = {}

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

    if 'user_config' not in st.session_state:
        st.session_state.user_config = {}

    if 'current_bot' not in st.session_state:
        st.session_state.current_bot = None

def restore_user_session():
    try:
        data_manager = get_data_manager_instance()
        if not data_manager:
            st.error("Failed to initialize data manager")
            return False

        stored_user_id = st.session_state.get('user_id')
        stored_username = st.session_state.get('username')

        if stored_user_id and stored_username and not st.session_state.get('authenticated'):
            st.session_state.authenticated = True

        if st.session_state.get('authenticated') and stored_user_id:
            user_config = data_manager.load_user_config(stored_user_id)
            user_bots = data_manager.load_user_bots(stored_user_id)
            chat_history = data_manager.load_chat_history(stored_user_id)

            st.session_state.user_config = user_config
            st.session_state.bots = user_bots
            st.session_state.chat_history = chat_history

            if user_config.get('username') and not st.session_state.get('username'):
                st.session_state.username = user_config['username']

            return True

        return False

    except Exception as e:
        st.error(f"Error restoring user session: {e}")
        return False

def ensure_user_authentication():
    if not st.session_state.get('authenticated') or not st.session_state.get('user_id'):
        st.markdown("""
        <div class="chat-header">
            <h1>🔒 Authentication Required</h1>
            <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
                Please authenticate to access your chat data
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🔑 Quick Login")

        with st.form("quick_login"):
            username = st.text_input("Username", value="demo_user", help="Enter your username")
            email = st.text_input("Email (optional)", help="Optional email for better user identification")

            if st.form_submit_button("🚀 Login & Restore Data", type="primary", use_container_width=True):
                if username:
                    try:
                        data_manager = get_data_manager_instance()
                        if data_manager:
                            user_id = data_manager.initialize_user_session(username, email)
                            st.success(f"✅ Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Failed to initialize data manager")
                    except Exception as e:
                        st.error(f"Login failed: {e}")
                else:
                    st.error("Please enter a username")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Home", use_container_width=True):
                st.switch_page("app.py")
        with col2:
            if st.button("🤖 Create Bot", use_container_width=True):
                st.switch_page("pages/1_🤖_Create_Bot.py")

        return False

    return True

def save_chat_data():
    try:
        if st.session_state.get('authenticated') and st.session_state.get('user_id'):
            dm = get_data_manager_instance()
            if dm:
                dm.save_all_user_data(st.session_state.user_id)
                return True
            return False
        return False
    except Exception as e:
        st.error(f"Error saving chat data: {e}")
        return False

def main():
    load_chat_css()
    initialize_chat_state()

    restore_user_session()

    if not ensure_user_authentication():
        return

    handle_bot_selection()

    if st.session_state.current_bot:
        render_chat_interface()
    else:
        render_bot_selection()

def handle_bot_selection():
    query_params = st.query_params
    bot_id_from_url = query_params.get("bot_id")

    if bot_id_from_url and bot_id_from_url in st.session_state.get('bots', {}):
        st.session_state.current_bot = bot_id_from_url
    elif bot_id_from_url:
        st.error(f"❌ Bot with ID '{bot_id_from_url}' not found")

def render_chat_interface():
    bot_config = st.session_state.bots.get(st.session_state.current_bot)

    if not bot_config:
        st.error("❌ Selected bot not found")
        st.session_state.current_bot = None
        st.rerun()
        return

    show_chat_sidebar(bot_config)

    if CHAT_INTERFACE_AVAILABLE:
        chat_interface = ChatInterface()
        chat_interface.show_chat_interface(bot_config)
    else:
        show_fallback_chat_interface(bot_config)

def render_bot_selection():
    st.markdown("""
    <div class="chat-header">
        <h1>💬 Chat with Your AI Bots</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
            Select a bot below to start an intelligent conversation
        </p>
    </div>
    """, unsafe_allow_html=True)

    show_bot_selection_interface()

def show_chat_sidebar(bot_config: Dict):
    with st.sidebar:
        if st.session_state.get('username'):
            st.markdown(f"👤 **Logged in as:** {st.session_state.username}")
            st.markdown("---")

        st.markdown(f"""
        <div class="bot-info-card">
            <h3 style="margin: 0 0 0.5rem 0;">🤖 {bot_config['name']}</h3>
            <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">
                {bot_config.get('description', 'AI Assistant')[:100]}{'...' if len(bot_config.get('description', '')) > 100 else ''}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Bot Configuration")

        config_info = f"""
        **Model:** {bot_config.get('model', 'N/A')}
        **Personality:** {bot_config.get('personality', 'Professional')}
        **Status:** {bot_config.get('status', 'active').title()}
        **Temperature:** {bot_config.get('temperature', 0.7)}
        **Max Tokens:** {bot_config.get('max_tokens', 1000)}
        """

        st.markdown(config_info)

        kb_enabled = bot_config.get('knowledge_base', {}).get('enabled', False)
        kb_status = "✅ Enabled" if kb_enabled else "❌ Disabled"
        st.markdown(f"**Knowledge Base:** {kb_status}")

        if kb_enabled:
            kb_source = bot_config.get('knowledge_base', {}).get('data_source', 'N/A')
            st.caption(f"Source: {kb_source}")

        st.markdown("---")

        show_data_status()

        st.markdown("---")

        st.markdown("### 🎯 Navigation")

        nav_buttons = [
            ("🏠 Home", "app.py"),
            ("🤖 Create Bot", "pages/1_🤖_Create_Bot.py"),
            ("📊 Dashboard", "pages/2_📊_Dashboard.py"),
            ("⚙️ Settings", "pages/3_⚙️_Settings.py")
        ]

        for i, (label, page) in enumerate(nav_buttons):
            if st.button(label, key=f"chat_nav_{i}", use_container_width=True):
                save_chat_data()
                st.switch_page(page)

        st.markdown("---")

        show_chat_statistics()

        st.markdown("---")

        show_bot_actions()

def show_data_status():
    st.markdown("### 💾 Data Status")

    try:
        data_manager = get_data_manager_instance()
        if data_manager and st.session_state.get('user_id'):
            user_id = st.session_state.user_id

            bots_exist = len(st.session_state.get('bots', {})) > 0
            chat_exist = len(st.session_state.get('chat_history', {})) > 0
            config_exist = len(st.session_state.get('user_config', {})) > 0

            st.success("✅ Data Manager Active")
            st.info(f"👤 User: {user_id[:12]}...")

            status_items = [
                ("Bots", bots_exist),
                ("Chats", chat_exist),
                ("Config", config_exist)
            ]

            for item, exists in status_items:
                icon = "✅" if exists else "⚪"
                st.caption(f"{icon} {item}: {'Saved' if exists else 'Empty'}")

            if st.button("💾 Save Now", key="manual_save", use_container_width=True):
                if save_chat_data():
                    st.success("Data saved!")
                    st.rerun()
                else:
                    st.error("Save failed!")
        else:
            st.error("❌ Data Manager Inactive")
            if st.button("🔄 Reconnect", key="reconnect_dm", use_container_width=True):
                st.rerun()

    except Exception as e:
        st.error(f"❌ Data Status Error: {str(e)[:50]}")

def show_chat_statistics():
    if st.session_state.current_bot in st.session_state.chat_history:
        messages = st.session_state.chat_history[st.session_state.current_bot]
        user_messages = len([m for m in messages if m.get('role') == 'user'])
        bot_messages = len([m for m in messages if m.get('role') == 'assistant'])

        st.markdown("### 📈 Chat Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f'<div class="chat-stats"><strong>{user_messages}</strong><br>Your Messages</div>', unsafe_allow_html=True)

        with col2:
            st.markdown(f'<div class="chat-stats"><strong>{bot_messages}</strong><br>Bot Responses</div>', unsafe_allow_html=True)

        likes = len([m for m in messages if m.get('rating') == 'like'])
        dislikes = len([m for m in messages if m.get('rating') == 'dislike'])

        if likes > 0 or dislikes > 0:
            st.markdown("### 👍 Feedback")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("👍 Likes", likes)
            with col2:
                st.metric("👎 Dislikes", dislikes)

            total_ratings = likes + dislikes
            if total_ratings > 0:
                satisfaction_rate = (likes / total_ratings) * 100
                st.metric("😊 Satisfaction", f"{satisfaction_rate:.0f}%")
    else:
        st.markdown("### 💬 Start Chatting")
        st.info("No conversation history yet. Start chatting to see statistics!")

def show_bot_actions():
    st.markdown("### ⚡ Quick Actions")

    action_buttons = [
        ("✏️ Edit Bot", "edit_bot"),
        ("🔗 Share Link", "share_link"),
        ("📥 Export Chat", "export_chat"),
        ("🔄 Clear Chat", "clear_chat"),
        ("🔙 Back to Bots", "back_to_bots")
    ]

    for label, action in action_buttons:
        if st.button(label, key=f"action_{action}", use_container_width=True):
            handle_bot_action(action)

def handle_bot_action(action: str):
    if action == "edit_bot":
        save_chat_data()
        st.switch_page("pages/1_🤖_Create_Bot.py")

    elif action == "share_link":
        show_share_link()

    elif action == "export_chat":
        export_current_chat()

    elif action == "clear_chat":
        clear_current_chat()

    elif action == "back_to_bots":
        st.session_state.current_bot = None
        st.rerun()

def show_bot_selection_interface():
    available_bots = {
        bot_id: bot_config for bot_id, bot_config in st.session_state.get('bots', {}).items()
        if bot_config.get('status') in ['active', 'testing']
    }

    if not available_bots:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
            <h2>🤖 No Active Bots Available</h2>
            <p style="color: #6c757d; margin: 1rem 0;">Create your first bot to start having AI conversations!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🚀 Create Your First Bot", key="create_first_bot", type="primary", use_container_width=True):
                save_chat_data()
                st.switch_page("pages/1_🤖_Create_Bot.py")
        return

    st.markdown("### 🔍 Find Your Bot")

    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("🔍 Search bots", key="bot_search", placeholder="Enter bot name...")
    with col2:
        status_filter = st.selectbox("Filter by status", ["All", "active", "testing"], key="status_filter")

    filtered_bots = []
    for bot_id, bot_config in available_bots.items():
        if search_term and search_term.lower() not in bot_config.get('name', '').lower():
            continue

        if status_filter != "All" and bot_config.get('status') != status_filter:
            continue

        filtered_bots.append((bot_id, bot_config))

    if filtered_bots:
        st.markdown(f"### 🤖 Available Bots ({len(filtered_bots)})")

        cols_per_row = 2
        for i in range(0, len(filtered_bots), cols_per_row):
            cols = st.columns(cols_per_row)

            for j, col in enumerate(cols):
                if i + j < len(filtered_bots):
                    bot_id, bot_config = filtered_bots[i + j]

                    with col:
                        show_bot_selection_card(bot_id, bot_config)
    else:
        st.info("🔍 No bots found matching your criteria. Try adjusting your search or filters.")

def show_bot_selection_card(bot_id: str, bot_config: Dict):
    messages = st.session_state.chat_history.get(bot_id, [])
    message_count = len([m for m in messages if m.get('role') == 'user'])

    status_colors = {
        'active': '#28a745',
        'testing': '#ffc107',
        'draft': '#6c757d'
    }

    status = bot_config.get('status', 'draft')
    status_color = status_colors.get(status, '#6c757d')

    st.markdown(f"""
    <div class="bot-selection-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: #333; font-size: 1.1rem;">🤖 {bot_config.get('name', 'Unnamed Bot')}</h4>
            <span class="status-badge" style="background: {status_color}; color: white;">{status}</span>
        </div>

        <p style="margin: 0 0 1rem 0; color: #666; font-size: 0.9rem; line-height: 1.4;">
            {bot_config.get('description', 'No description available')[:120]}{'...' if len(bot_config.get('description', '')) > 120 else ''}
        </p>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: #666; margin-bottom: 1rem;">
            <div><strong>Model:</strong> {bot_config.get('model', 'N/A')[:20]}{'...' if len(bot_config.get('model', '')) > 20 else ''}</div>
            <div><strong>Chats:</strong> {message_count}</div>
            <div><strong>Personality:</strong> {bot_config.get('personality', 'Professional')}</div>
            <div><strong>KB:</strong> {'Yes' if bot_config.get('knowledge_base', {}).get('enabled') else 'No'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬 Start Chat", key=f"chat_{bot_id}", type="primary", use_container_width=True):
            st.session_state.current_bot = bot_id
            initialize_bot_chat(bot_id, bot_config)
            st.rerun()

    with col2:
        if st.button("ℹ️ Details", key=f"details_{bot_id}", use_container_width=True):
            show_bot_details_modal(bot_id, bot_config)

def initialize_bot_chat(bot_id: str, bot_config: Dict):
    if bot_id not in st.session_state.chat_history:
        welcome_msg = bot_config.get('welcome_message', 'Hello! How can I help you today?')
        st.session_state.chat_history[bot_id] = [
            {
                "role": "assistant",
                "content": welcome_msg,
                "timestamp": datetime.now().isoformat(),
                "bot_id": bot_id
            }
        ]
        save_chat_data()

def show_bot_details_modal(bot_id: str, bot_config: Dict):
    with st.expander(f"📋 {bot_config.get('name', 'Unnamed Bot')} - Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔧 Configuration")
            st.markdown(f"**Name:** {bot_config.get('name', 'N/A')}")
            st.markdown(f"**Model:** {bot_config.get('model', 'N/A')}")
            st.markdown(f"**Personality:** {bot_config.get('personality', 'N/A')}")
            st.markdown(f"**Status:** {bot_config.get('status', 'N/A').title()}")
            st.markdown(f"**Created:** {bot_config.get('created_at', 'N/A')[:10]}")
            st.markdown(f"**Updated:** {bot_config.get('updated_at', 'N/A')[:10]}")

        with col2:
            st.markdown("#### ⚙️ Settings")
            st.markdown(f"**Temperature:** {bot_config.get('temperature', 'N/A')}")
            st.markdown(f"**Max Tokens:** {bot_config.get('max_tokens', 'N/A')}")
            st.markdown(f"**Context Window:** {bot_config.get('advanced_settings', {}).get('context_window', 'N/A')}")

            kb_config = bot_config.get('knowledge_base', {})
            if kb_config.get('enabled'):
                st.markdown(f"**Knowledge Base:** Enabled")
                st.markdown(f"**Data Source:** {kb_config.get('data_source', 'N/A')}")
                st.markdown(f"**Chunking:** {kb_config.get('chunking_strategy', 'N/A')}")
            else:
                st.markdown(f"**Knowledge Base:** Disabled")

        if bot_config.get('description'):
            st.markdown("#### 📝 Description")
            st.markdown(bot_config['description'])

        if bot_config.get('welcome_message'):
            st.markdown("#### 👋 Welcome Message")
            st.info(bot_config['welcome_message'])

        messages = st.session_state.chat_history.get(bot_id, [])
        if messages:
            st.markdown("#### 📊 Chat Statistics")

            user_msgs = len([m for m in messages if m.get('role') == 'user'])
            bot_msgs = len([m for m in messages if m.get('role') == 'assistant'])
            likes = len([m for m in messages if m.get('rating') == 'like'])
            dislikes = len([m for m in messages if m.get('rating') == 'dislike'])

            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

            with stats_col1:
                st.metric("User Messages", user_msgs)
            with stats_col2:
                st.metric("Bot Responses", bot_msgs)
            with stats_col3:
                st.metric("👍 Likes", likes)
            with stats_col4:
                st.metric("👎 Dislikes", dislikes)

def show_fallback_chat_interface(bot_config: Dict):
    st.markdown("### 💬 Chat Interface")

    bot_id = st.session_state.current_bot
    initialize_bot_chat(bot_id, bot_config)

    messages = st.session_state.chat_history[bot_id]

    for i, message in enumerate(messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and i > 0:
                col1, col2, col3 = st.columns([1, 1, 6])

                with col1:
                    if st.button("👍", key=f"like_{bot_id}_{i}", help="Like this response"):
                        message["rating"] = "like"
                        save_chat_data()
                        st.rerun()

                with col2:
                    if st.button("👎", key=f"dislike_{bot_id}_{i}", help="Dislike this response"):
                        message["rating"] = "dislike"
                        save_chat_data()
                        st.rerun()

    if prompt := st.chat_input("Type your message here..."):
        user_message = {
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat(),
            "bot_id": bot_id
        }
        st.session_state.chat_history[bot_id].append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                personality = bot_config.get('personality', 'Professional')
                bot_name = bot_config.get('name', 'Assistant')

                if personality.lower() == 'friendly':
                    response = f"Hi there! I'm {bot_name}, and I appreciate you asking: '{prompt}'. While I'd love to give you a full AI response, I'm currently using a basic interface. Your message has been saved and I'm here to help however I can! 😊"
                elif personality.lower() == 'professional':
                    response = f"Thank you for your inquiry: '{prompt}'. I am {bot_name}, and while operating in basic mode, I can confirm your message has been properly logged. For advanced AI capabilities, please ensure the full chat interface is available."
                elif personality.lower() == 'creative':
                    response = f"🎨 Oh, what an interesting thought: '{prompt}'! I'm {bot_name}, and even in this simplified mode, I can sense the creativity in your message. Your words have been captured and stored safely!"
                else:
                    response = f"I understand you said: '{prompt}'. I'm {bot_name}, and while this is a basic response interface, your message has been properly saved and logged."

                st.markdown(response)

                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat(),
                    "bot_id": bot_id,
                    "interface_type": "fallback",
                    "model_used": bot_config.get('model', 'fallback'),
                    "personality": personality
                }
                st.session_state.chat_history[bot_id].append(assistant_message)

                save_chat_data()

def show_share_link():
    if not st.session_state.current_bot:
        return

    base_url = "https://your-chatbot-app.streamlit.app"
    share_link = f"{base_url}/pages/4_🎯_Chat.py?bot_id={st.session_state.current_bot}"

    with st.expander("🔗 Share This Chat", expanded=True):
        st.markdown("**Direct Chat Link:**")
        st.code(share_link, language="text")

        st.markdown("**Sharing Options:**")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📋 Copy Link", key="copy_share_link"):
                st.success("Link copied to clipboard!")

        with col2:
            bot_name = st.session_state.bots[st.session_state.current_bot].get('name', 'My Bot')
            email_subject = f"Check out my AI chatbot: {bot_name}"
            email_body = f"Try chatting with my AI bot: {share_link}"
            st.markdown(f'<a href="mailto:?subject={email_subject}&body={email_body}" target="_blank">📧 Open Email</a>', unsafe_allow_html=True)

        with col3:
            whatsapp_text = f"Check out this AI chatbot: {share_link}"
            st.markdown(f'<a href="https://wa.me/?text={whatsapp_text}" target="_blank">💬 Share on WhatsApp</a>', unsafe_allow_html=True)

        st.markdown("**Instructions:**")
        st.markdown("• Share this link with anyone")
        st.markdown("• They can chat with your bot directly")
        st.markdown("• No account required for users")

def export_current_chat():
    if not st.session_state.current_bot:
        st.warning("No bot selected")
        return

    bot_id = st.session_state.current_bot
    chat_data = st.session_state.chat_history.get(bot_id, [])

    if not chat_data:
        st.warning("No chat history to export")
        return

    bot_config = st.session_state.bots.get(bot_id, {})
    user_config = st.session_state.get('user_config', {})

    export_data = {
        "export_info": {
            "type": "chat_history",
            "version": "1.2",
            "bot_id": bot_id,
            "bot_name": bot_config.get('name', 'Unknown Bot'),
            "user_id": st.session_state.get('user_id', 'unknown'),
            "username": st.session_state.get('username', 'unknown'),
            "exported_at": datetime.now().isoformat(),
            "total_messages": len(chat_data),
            "interface_version": "enhanced_with_persistence"
        },
        "bot_config": {
            "name": bot_config.get('name', 'Unknown'),
            "description": bot_config.get('description', ''),
            "model": bot_config.get('model', 'N/A'),
            "personality": bot_config.get('personality', 'N/A'),
            "status": bot_config.get('status', 'N/A'),
            "temperature": bot_config.get('temperature', 0.7),
            "max_tokens": bot_config.get('max_tokens', 1000),
            "knowledge_base_enabled": bot_config.get('knowledge_base', {}).get('enabled', False),
            "created_at": bot_config.get('created_at', ''),
            "updated_at": bot_config.get('updated_at', '')
        },
        "chat_history": chat_data,
        "statistics": {
            "user_messages": len([m for m in chat_data if m.get('role') == 'user']),
            "bot_messages": len([m for m in chat_data if m.get('role') == 'assistant']),
            "likes": len([m for m in chat_data if m.get('rating') == 'like']),
            "dislikes": len([m for m in chat_data if m.get('rating') == 'dislike']),
            "conversation_duration": calculate_conversation_duration(chat_data),
            "average_response_length": calculate_average_response_length(chat_data)
        }
    }

    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

    st.download_button(
        label="📥 Download Chat History",
        data=json_str,
        file_name=f"chat_history_{bot_config.get('name', 'bot').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="export_chat_btn"
    )

    st.success("✅ Chat history prepared for download!")

    with st.expander("📊 Export Summary", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", export_data["statistics"]["user_messages"] + export_data["statistics"]["bot_messages"])
        with col2:
            st.metric("User Messages", export_data["statistics"]["user_messages"])
        with col3:
            st.metric("Bot Responses", export_data["statistics"]["bot_messages"])

def calculate_conversation_duration(chat_data):
    try:
        if len(chat_data) < 2:
            return "0 minutes"

        first_msg = chat_data[0].get('timestamp')
        last_msg = chat_data[-1].get('timestamp')

        if first_msg and last_msg:
            from datetime import datetime
            start = datetime.fromisoformat(first_msg)
            end = datetime.fromisoformat(last_msg)
            duration = end - start

            if duration.days > 0:
                return f"{duration.days} days"
            elif duration.seconds > 3600:
                hours = duration.seconds // 3600
                return f"{hours} hours"
            elif duration.seconds > 60:
                minutes = duration.seconds // 60
                return f"{minutes} minutes"
            else:
                return f"{duration.seconds} seconds"

        return "Unknown"
    except Exception:
        return "Unknown"

def calculate_average_response_length(chat_data):
    try:
        bot_messages = [m for m in chat_data if m.get('role') == 'assistant']
        if not bot_messages:
            return 0

        total_length = sum(len(m.get('content', '')) for m in bot_messages)
        return round(total_length / len(bot_messages))
    except Exception:
        return 0

def clear_current_chat():
    if not st.session_state.current_bot:
        return

    bot_id = st.session_state.current_bot
    bot_config = st.session_state.bots.get(bot_id, {})

    st.markdown("### ⚠️ Clear Chat History")
    st.warning("This will permanently delete all messages in this conversation. This action cannot be undone.")

    messages = st.session_state.chat_history.get(bot_id, [])
    if messages:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", len(messages))
        with col2:
            st.metric("User Messages", len([m for m in messages if m.get('role') == 'user']))
        with col3:
            st.metric("Bot Responses", len([m for m in messages if m.get('role') == 'assistant']))

    col1, col2 = st.columns(2)

    with col1:
        if st.button("❌ Yes, Clear All Messages", key="confirm_clear_chat", type="primary"):
            welcome_msg = bot_config.get('welcome_message', 'Hello! How can I help you today?')
            st.session_state.chat_history[bot_id] = [
                {
                    "role": "assistant",
                    "content": welcome_msg,
                    "timestamp": datetime.now().isoformat(),
                    "bot_id": bot_id,
                    "is_welcome": True
                }
            ]

            save_chat_data()

            st.success("✅ Chat history cleared!")
            st.rerun()

    with col2:
        if st.button("🔙 Cancel", key="cancel_clear_chat"):
            st.info("Chat history preserved.")

def add_chat_message_rating(bot_id: str, message_index: int, rating: str):
    try:
        if bot_id in st.session_state.chat_history:
            messages = st.session_state.chat_history[bot_id]
            if 0 <= message_index < len(messages):
                messages[message_index]['rating'] = rating
                messages[message_index]['rated_at'] = datetime.now().isoformat()

                save_chat_data()
                return True
        return False
    except Exception as e:
        st.error(f"Error rating message: {e}")
        return False

def on_session_state_change():
    if st.session_state.get('authenticated'):
        save_chat_data()

if 'chat_auto_save_initialized' not in st.session_state:
    st.session_state.chat_auto_save_initialized = True

    if st.session_state.get('authenticated'):
        if 'previous_chat_history_hash' not in st.session_state:
            st.session_state.previous_chat_history_hash = hash(str(st.session_state.get('chat_history', {})))

if __name__ == "__main__":
    main()

