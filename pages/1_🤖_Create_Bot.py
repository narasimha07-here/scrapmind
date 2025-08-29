import streamlit as st
import sys
import os
from components.data_manager import DataManager, auto_save_user_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from components.bot_creator import BotCreator

st.set_page_config(
    page_title="Create Bot - Chatbot Builder",
    page_icon="🤖",
    layout="wide"
)

def get_data_manager_instance():
    if 'data_manager_instance' not in st.session_state:
        st.session_state.data_manager_instance = DataManager()
    return st.session_state.data_manager_instance

def initialize_data_manager():
    if (st.session_state.get('authenticated') and
        st.session_state.get('user_id') and
        not st.session_state.get('user_config')):

        dm = get_data_manager_instance()
        user_id = st.session_state.user_id

        st.session_state.user_config = dm.load_user_config(user_id)
        st.session_state.bots = dm.load_user_bots(user_id)
        st.session_state.chat_history = dm.load_chat_history(user_id)

def rehydrate_session():
    initialize_data_manager()

    if st.session_state.get('authenticated') and st.session_state.get('user_id'):
        dm = get_data_manager_instance()
        user_id = st.session_state.user_id

        if not st.session_state.get('bots'):
            st.session_state.bots = dm.load_user_bots(user_id)
        if not st.session_state.get('user_config'):
            st.session_state.user_config = dm.load_user_config(user_id)
        if not st.session_state.get('chat_history'):
            st.session_state.chat_history = dm.load_chat_history(user_id)

def save_bot_data(bot_id, bot_config):
    if not st.session_state.get('authenticated') or not st.session_state.get('user_id'):
        return False

    try:
        if 'bots' not in st.session_state:
            st.session_state.bots = {}

        st.session_state.bots[bot_id] = bot_config

        dm = get_data_manager_instance()
        success = dm.save_user_bots(st.session_state.user_id, st.session_state.bots)

        if success:
            st.success("✅ Bot saved successfully!")
            if 'knowledge_base' in bot_config and 'uploaded_files' in bot_config['knowledge_base']:
                save_bot_files(bot_id, bot_config['knowledge_base']['uploaded_files'])
        else:
            st.error("❌ Failed to save bot")

        return success
    except Exception as e:
        st.error(f"Error saving bot: {e}")
        return False

def save_bot_files(bot_id, uploaded_files):
    if not uploaded_files or not st.session_state.get('user_id'):
        return True

    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.user_id

        for i, file in enumerate(uploaded_files):
            success = dm.save_uploaded_file(user_id, bot_id, i, file)
            if not success:
                st.warning(f"Failed to save file: {file.name}")

        return True
    except Exception as e:
        st.error(f"Error saving files: {e}")
        return False

def load_bot_draft(bot_id):
    if bot_id in st.session_state.get('bots', {}):
        return st.session_state.bots[bot_id]
    return None

def main():
    rehydrate_session()

    if not st.session_state.get('authenticated', False):
        st.error("🔒 Please login first")
        if st.button("Go to Login"):
            st.switch_page("app.py")
        return

    if not st.secrets.get("OPENROUTER_API_KEY", ""):
        st.warning("⚠️ OpenRouter API key not configured")
        if st.button("Go to Settings"):
            st.switch_page("pages/3_⚙️_Settings.py")
        return

    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        if st.button("🏠 Dashboard"):
            st.switch_page("app.py")
        if st.button("📊 My Bots"):
            st.switch_page("pages/2_📊_Dashboard.py")
        if st.button("💬 Test Chat"):
            st.switch_page("pages/4_🎯_Chat.py")
        if st.button("⚙️ Settings"):
            st.switch_page("pages/3_⚙️_Settings.py")

        st.markdown("---")

        if st.session_state.get('bots'):
            st.markdown("### 📝 Your Bots")
            for bot_id, bot_config in st.session_state.bots.items():
                status_icon = {"active": "🟢", "draft": "🟡", "testing": "🔵"}.get(
                    bot_config.get('status', 'draft'), "⚪"
                )
                if st.button(f"{status_icon} {bot_config.get('name', 'Unnamed')[:15]}..."):
                    st.session_state.editing_bot_id = bot_id
                    st.rerun()

        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - **Auto-Save**: Your progress is automatically saved
        - **Free Models**: Look for 🆓 in model selection
        - **Knowledge Base**: Upload documents to make your bot smarter
        - **Test First**: Always test before deploying
        """)

    editing_bot_id = st.session_state.get('editing_bot_id')
    if editing_bot_id:
        st.info(f"✏️ Editing bot: {st.session_state.bots[editing_bot_id].get('name', 'Unnamed')}")

    try:
        bot_creator = BotCreator()

        bot_creator.save_bot_callback = save_bot_data
        bot_creator.load_bot_callback = load_bot_draft

        if editing_bot_id and editing_bot_id in st.session_state.get('bots', {}):
            bot_creator.load_existing_bot(st.session_state.bots[editing_bot_id])

        bot_creator.show_bot_creation_wizard()

    except Exception as e:
        st.error(f"Error loading bot creator: {str(e)}")
        st.info("Please check your configuration and try again.")

        with st.expander("Debug Information"):
            st.write("Session State Keys:", list(st.session_state.keys()))
            st.write("User ID:", st.session_state.get('user_id', 'Not set'))
            st.write("Authenticated:", st.session_state.get('authenticated', False))

if __name__ == "__main__":
    main()

