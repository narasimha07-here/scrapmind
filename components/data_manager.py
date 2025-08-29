import streamlit as st
import json
import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import pickle
import shutil
import base64

class MockFile:
    """
    A mock class to make Streamlit's UploadedFile objects pickle-serializable.
    It stores the file content as bytes.
    """
    def __init__(self, name, content, file_type="application/octet-stream"):
        self.name = name
        self.type = file_type
        # Ensure content is bytes
        self._content = content if isinstance(content, bytes) else content.encode('utf-8') if isinstance(content, str) else b''
        
    def getvalue(self):
        return self._content
        
    def read(self):
        return self._content
        
    def __getstate__(self):
        # This method is called when pickling
        return {
            'name': self.name,
            'type': self.type,
            '_content': self._content
        }
        
    def __setstate__(self, state):
        # This method is called when unpickling
        self.name = state['name']
        self.type = state['type']
        self._content = state['_content']
        
    def __repr__(self):
        return f"MockFile(name='{self.name}', type='{self.type}', size={len(self._content) if self._content else 0})"

class DataManager:
    def __init__(self):
        self.data_dir = "user_data"
        self.ensure_data_directory()
        self.auto_save_enabled = True
        # Expose MockFile class for use in other modules
        self.MockFile = MockFile

    def ensure_data_directory(self):
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "users"),
            os.path.join(self.data_dir, "bots"),
            os.path.join(self.data_dir, "vectordb"),
            os.path.join(self.data_dir, "files"),
            os.path.join(self.data_dir, "backups")
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def make_bot_config_pickle_safe(self, bot_config):
        """
        Recursively cleans a bot_config dictionary to ensure all its contents
        are pickle-serializable. Specifically handles Streamlit's UploadedFile
        objects by converting them to MockFile instances.
        """
        if not isinstance(bot_config, dict):
            return bot_config
            
        cleaned_config = {}
        
        for key, value in bot_config.items():
            # Exclude non-pickleable internal objects
            if key in ['_knowledge_processor', '_openai_client', '_vector_store']:
                continue
                
            # Exclude callable objects that might not be pickleable (e.g., lambda functions)
            if callable(value) and not hasattr(value, '__name__'):
                continue
                
            if hasattr(value, '__name__') and value.__name__ == '<lambda>':
                continue
                
            if isinstance(value, dict):
                cleaned_config[key] = self.make_bot_config_pickle_safe(value)
            elif isinstance(value, list):
                cleaned_list = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_list.append(self.make_bot_config_pickle_safe(item))
                    # Check if it's an UploadedFile or similar file-like object
                    elif hasattr(item, 'getvalue') and hasattr(item, 'name'):
                        # Convert to MockFile if it's not already one
                        if not isinstance(item, MockFile):
                            mock_file = MockFile(
                                getattr(item, 'name', 'unknown'),
                                item.getvalue() if hasattr(item, 'getvalue') else b'',
                                getattr(item, 'type', 'application/octet-stream')
                            )
                            cleaned_list.append(mock_file)
                        else:
                            cleaned_list.append(item) # Already a MockFile
                    else:
                        cleaned_list.append(item)
                cleaned_config[key] = cleaned_list
            else:
                cleaned_config[key] = value
                
        return cleaned_config

    def get_user_id(self, username: str, email: str = None) -> str:
        identifier = email if email else username
        user_hash = hashlib.md5(identifier.encode()).hexdigest()
        return f"user_{user_hash[:12]}"

    def save_user_config(self, user_id: str, config: Dict) -> bool:
        try:
            user_file = os.path.join(self.data_dir, "users", f"{user_id}.json")
            
            config['user_id'] = user_id
            config['last_updated'] = datetime.now().isoformat()
            config['version'] = '1.2'
            
            if 'username' not in config and hasattr(st.session_state, 'username'):
                config['username'] = st.session_state.username
            
            if os.path.exists(user_file):
                backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_config_backup.json")
                shutil.copy2(user_file, backup_file)
            
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            st.session_state.user_config = config.copy()
            return True
            
        except Exception as e:
            st.error(f"Error saving user config: {e}")
            return False

    def load_user_config(self, user_id: str) -> Dict:
        try:
            user_file = os.path.join(self.data_dir, "users", f"{user_id}.json")
            if os.path.exists(user_file):
                try:
                    with open(user_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    st.session_state.user_config = config.copy()
                    return config
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Config file corrupted, trying backup: {e}")
                    backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_config_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        st.session_state.user_config = config.copy()
                        st.info("Restored from backup")
                        return config
            
            default_config = self.get_default_user_config(user_id)
            st.session_state.user_config = default_config.copy()
            return default_config
            
        except Exception as e:
            st.error(f"Error loading user config: {e}")
            return self.get_default_user_config(user_id)

    def get_default_user_config(self, user_id: str) -> Dict:
        return {
            'user_id': user_id,
            'username': 'User',
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

    def save_user_bots(self, user_id: str, bots: Dict) -> bool:
        try:
            bots_file = os.path.join(self.data_dir, "bots", f"{user_id}_bots.json")
            
            if os.path.exists(bots_file):
                backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_bots_backup.json")
                shutil.copy2(bots_file, backup_file)

            # Prepare bots for JSON serialization (remove uploaded_files as they are saved separately)
            json_safe_bots = {}
            for bot_id, bot_config in bots.items():
                cleaned_bot = bot_config.copy() # Start with a copy
                
                # Remove 'uploaded_files' from knowledge_base for JSON serialization
                if 'knowledge_base' in cleaned_bot and 'uploaded_files' in cleaned_bot['knowledge_base']:
                    cleaned_bot['knowledge_base']['uploaded_files'] = [] # Clear the list for JSON
                
                cleaned_bot['user_id'] = user_id
                cleaned_bot['last_saved'] = datetime.now().isoformat()
                
                json_safe_bots[bot_id] = cleaned_bot

            with open(bots_file, 'w', encoding='utf-8') as f:
                json.dump(json_safe_bots, f, indent=2, ensure_ascii=False, default=str)

            # The st.session_state.bots should already contain pickle-safe objects (MockFile)
            # from the BotCreator's save_bot_config method.
            # No need to re-assign st.session_state.bots here, as it's managed by BotCreator.
            return True

        except Exception as e:
            st.error(f"Error saving bots: {e}")
            return False

    def load_user_bots(self, user_id: str) -> Dict:
        try:
            bots_file = os.path.join(self.data_dir, "bots", f"{user_id}_bots.json")
            if os.path.exists(bots_file):
                try:
                    with open(bots_file, 'r', encoding='utf-8') as f:
                        bots = json.load(f)

                    # Reconstruct MockFile objects from file_metadata and loaded content
                    for bot_id, bot_config in bots.items():
                        if 'knowledge_base' in bot_config:
                            kb = bot_config['knowledge_base']
                            if 'file_metadata' in kb and kb['file_metadata']:
                                restored_files = []
                                for file_meta in kb['file_metadata']:
                                    file_content = self.load_uploaded_file(
                                        user_id, bot_id, file_meta['index']
                                    )
                                    if file_content:
                                        mock_file = MockFile( # Use the MockFile class
                                            file_meta['name'],
                                            file_content,
                                            file_meta.get('type', 'unknown')
                                        )
                                        restored_files.append(mock_file)
                                
                                kb['uploaded_files'] = restored_files # Populate with MockFile objects
                                kb['file_names'] = [f.name for f in restored_files]

                    st.session_state.bots = bots.copy() # Update session state with loaded bots
                    return bots

                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Bots file corrupted, trying backup: {e}")
                    backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_bots_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            bots = json.load(f)
                        
                        # Reconstruct MockFile objects from backup
                        for bot_id, bot_config in bots.items():
                            if 'knowledge_base' in bot_config:
                                kb = bot_config['knowledge_base']
                                if 'file_metadata' in kb and kb['file_metadata']:
                                    restored_files = []
                                    for file_meta in kb['file_metadata']:
                                        file_content = self.load_uploaded_file(
                                            user_id, bot_id, file_meta['index']
                                        )
                                        if file_content:
                                            mock_file = MockFile( # Use the MockFile class
                                                file_meta['name'],
                                                file_content,
                                                file_meta.get('type', 'unknown')
                                            )
                                            restored_files.append(mock_file)
                                    kb['uploaded_files'] = restored_files
                                    kb['file_names'] = [f.name for f in restored_files]
                        
                        st.session_state.bots = bots.copy()
                        st.info("Restored bots from backup")
                        return bots

            return {}

        except Exception as e:
            st.error(f"Error loading bots: {e}")
            return {}

    def save_uploaded_file(self, user_id: str, bot_id: str, file_index: int, file) -> bool:
        try:
            file_dir = os.path.join(self.data_dir, "files", user_id, bot_id)
            os.makedirs(file_dir, exist_ok=True)

            file_path = os.path.join(file_dir, f"file_{file_index}.dat")
            
            if hasattr(file, 'getvalue'):
                content = file.getvalue()
            elif hasattr(file, 'read'):
                content = file.read()
            else:
                content = file # Assume content is already bytes or string
                
            with open(file_path, 'wb') as f:
                if isinstance(content, str):
                    f.write(content.encode('utf-8'))
                else:
                    f.write(content)

            meta_path = os.path.join(file_dir, f"file_{file_index}_meta.json")
            metadata = {
                'name': getattr(file, 'name', f'file_{file_index}'),
                'type': getattr(file, 'type', 'unknown'),
                'size': len(content) if content else 0,
                'index': file_index,
                'upload_time': datetime.now().isoformat(),
                'user_id': user_id,
                'bot_id': bot_id
            }

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")
            return False

    def load_uploaded_file(self, user_id: str, bot_id: str, file_index: int) -> Optional[bytes]:
        try:
            file_path = os.path.join(self.data_dir, "files", user_id, bot_id, f"file_{file_index}.dat")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return f.read()
            return None
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return None

    def get_uploaded_file_metadata(self, user_id: str, bot_id: str, file_index: int) -> Optional[Dict]:
        try:
            meta_path = os.path.join(self.data_dir, "files", user_id, bot_id, f"file_{file_index}_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error loading file metadata: {e}")
            return None

    def delete_bot_files(self, user_id: str, bot_id: str) -> bool:
        try:
            bot_file_dir = os.path.join(self.data_dir, "files", user_id, bot_id)
            if os.path.exists(bot_file_dir):
                shutil.rmtree(bot_file_dir)
                st.success(f"Successfully deleted files for bot {bot_id[:8]}...")
            return True
        except Exception as e:
            st.error(f"Error deleting bot files: {e}")
            return False

    def save_chat_history(self, user_id: str, chat_history: Dict) -> bool:
        try:
            chat_file = os.path.join(self.data_dir, "users", f"{user_id}_chats.json")

            if os.path.exists(chat_file):
                backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_chats_backup.json")
                shutil.copy2(chat_file, backup_file)

            serializable_chat_history = {}
            for bot_id, messages in chat_history.items():
                serializable_messages = []
                for message in messages:
                    msg_copy = message.copy()
                    if 'audio_data' in msg_copy and isinstance(msg_copy['audio_data'], bytes):
                        msg_copy['audio_data_b64'] = base64.b64encode(msg_copy['audio_data']).decode('utf-8')
                        del msg_copy['audio_data']
                    serializable_messages.append(msg_copy)
                serializable_chat_history[bot_id] = serializable_messages

            chat_data = {
                'user_id': user_id,
                'last_updated': datetime.now().isoformat(),
                'version': '1.2',
                'chat_history': serializable_chat_history
            }

            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False, default=str)

            st.session_state.chat_history = chat_history.copy()
            return True

        except Exception as e:
            st.error(f"Error saving chat history: {e}")
            return False

    def load_chat_history(self, user_id: str) -> Dict:
        try:
            chat_file = os.path.join(self.data_dir, "users", f"{user_id}_chats.json")
            if os.path.exists(chat_file):
                try:
                    with open(chat_file, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)

                    if isinstance(chat_data, dict) and 'chat_history' in chat_data:
                        loaded_chat_history = chat_data['chat_history']
                    else:
                        loaded_chat_history = chat_data

                    restored_chat_history = {}
                    for bot_id, messages in loaded_chat_history.items():
                        restored_messages = []
                        for message in messages:
                            msg_copy = message.copy()
                            if 'audio_data_b64' in msg_copy and isinstance(msg_copy['audio_data_b64'], str):
                                try:
                                    msg_copy['audio_data'] = base64.b64decode(msg_copy['audio_data_b64'])
                                    del msg_copy['audio_data_b64']
                                except Exception as decode_e:
                                    st.warning(f"Failed to decode audio_data_b64 for message: {decode_e}")
                                    msg_copy['audio_data'] = None
                            restored_messages.append(msg_copy)
                        restored_chat_history[bot_id] = restored_messages

                    st.session_state.chat_history = restored_chat_history.copy()
                    return restored_chat_history

                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Chat history corrupted, trying backup: {e}")
                    backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_chats_backup.json")
                    if os.path.exists(backup_file):
                        pass # Logic to load from backup could go here if needed

            return {}

        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            return {}

    def initialize_user_session(self, username: str, email: str = None):
        user_id = self.get_user_id(username, email)

        st.session_state.user_id = user_id
        st.session_state.username = username
        st.session_state.authenticated = True

        user_config = self.load_user_config(user_id)
        user_bots = self.load_user_bots(user_id)
        chat_history = self.load_chat_history(user_id)

        st.session_state.user_config = user_config
        st.session_state.bots = user_bots
        st.session_state.chat_history = chat_history

        if user_config.get('username') != username:
            user_config['username'] = username
            self.save_user_config(user_id, user_config)

        st.success(f"Welcome back, {username}! Your data has been restored.")
        return user_id

    def save_all_user_data(self, user_id: str = None):
        if not user_id:
            user_id = st.session_state.get('user_id')
        if not user_id:
            return False

        try:
            success = True
            success &= self.save_user_config(user_id, st.session_state.get('user_config', {}))
            success &= self.save_user_bots(user_id, st.session_state.get('bots', {}))
            success &= self.save_chat_history(user_id, st.session_state.get('chat_history', {}))
            return success
        except Exception as e:
            st.error(f"Error saving user data: {e}")
            return False

    def get_user_stats(self, user_id: str) -> Dict:
        try:
            bots = self.load_user_bots(user_id)
            chat_history = self.load_chat_history(user_id)

            total_files = 0
            total_file_size = 0
            for bot in bots.values():
                kb = bot.get('knowledge_base', {})
                file_metadata = kb.get('file_metadata', [])
                total_files += len(file_metadata)
                total_file_size += sum(meta.get('size', 0) for meta in file_metadata)

            stats = {
                'total_bots': len(bots),
                'active_bots': sum(1 for bot in bots.values() if bot.get('status') == 'active'),
                'draft_bots': sum(1 for bot in bots.values() if bot.get('status') == 'draft'),
                'testing_bots': sum(1 for bot in bots.values() if bot.get('status') == 'testing'),
                'total_conversations': sum(1 for conv_id in chat_history.keys()), # Count unique conversation IDs
                'total_messages': sum(len(messages) for messages in chat_history.values() if isinstance(messages, list)),
                'kb_enabled_bots': sum(1 for bot in bots.values() if bot.get('knowledge_base', {}).get('enabled')),
                'total_uploaded_files': total_files,
                'total_file_size': total_file_size,
                'last_activity': max(
                    [bot.get('updated_at', bot.get('created_at', '2024-01-01')) for bot in bots.values()] + ['2024-01-01']
                ) if bots else '2024-01-01'
            }
            return stats

        except Exception as e:
            st.error(f"Error getting user stats: {e}")
            return {}

    def cleanup_orphaned_files(self, user_id: str):
        try:
            user_file_dir = os.path.join(self.data_dir, "files", user_id)
            if not os.path.exists(user_file_dir):
                return True

            bots = self.load_user_bots(user_id)
            active_bot_ids = set(bots.keys())

            for bot_dir in os.listdir(user_file_dir):
                bot_path = os.path.join(user_file_dir, bot_dir)
                if os.path.isdir(bot_path) and bot_dir not in active_bot_ids:
                    st.info(f"Cleaning up orphaned files for bot {bot_dir[:8]}...")
                    shutil.rmtree(bot_path)

            return True

        except Exception as e:
            st.error(f"Error cleaning up orphaned files: {e}")
            return False

    def get_storage_usage(self, user_id: str) -> Dict:
        try:
            usage = {
                'config_size': 0,
                'bots_size': 0,
                'chat_size': 0,
                'files_size': 0,
                'total_size': 0
            }

            config_file = os.path.join(self.data_dir, "users", f"{user_id}.json")
            if os.path.exists(config_file):
                usage['config_size'] = os.path.getsize(config_file)

            bots_file = os.path.join(self.data_dir, "bots", f"{user_id}_bots.json")
            if os.path.exists(bots_file):
                usage['bots_size'] = os.path.getsize(bots_file)

            chat_file = os.path.join(self.data_dir, "users", f"{user_id}_chats.json")
            if os.path.exists(chat_file):
                usage['chat_size'] = os.path.getsize(chat_file)

            files_dir = os.path.join(self.data_dir, "files", user_id)
            if os.path.exists(files_dir):
                for root, dirs, files in os.walk(files_dir):
                    usage['files_size'] += sum(
                        os.path.getsize(os.path.join(root, file)) for file in files
                    )

            usage['total_size'] = sum(usage.values())
            return usage

        except Exception as e:
            st.error(f"Error calculating storage usage: {e}")
            return {}

def get_data_manager_instance():
    if 'data_manager_instance' not in st.session_state:
        st.session_state.data_manager_instance = DataManager()
    return st.session_state.data_manager_instance

def auto_save_user_data():
    try:
        data_manager = get_data_manager_instance()
        if data_manager and data_manager.auto_save_enabled:
            data_manager.save_all_user_data()
    except Exception as e:
        st.warning(f"Auto-save failed: {str(e)}")

def get_data_manager():
    try:
        return get_data_manager_instance()
    except Exception as e:
        st.error(f"Failed to initialize data manager: {e}")
        return None

def format_file_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"