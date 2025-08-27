import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import sqlite3
import streamlit as st
import hashlib
import pickle
import shutil

class DataManager:
    def __init__(self):
        self.data_dir = "user_data"
        self.ensure_data_directory()
        self.auto_save_enabled = True
    
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
            'openrouter_api_key': '',
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
            
            serializable_bots = {}
            for bot_id, bot_config in bots.items():
                serializable_bot = bot_config.copy()
                
                serializable_bot['user_id'] = user_id
                serializable_bot['last_saved'] = datetime.now().isoformat()
                
                if '_knowledge_processor' in serializable_bot:
                    del serializable_bot['_knowledge_processor']
                
                if 'knowledge_base' in serializable_bot:
                    kb = serializable_bot['knowledge_base']
                    
                    if 'uploaded_files' in kb:
                        kb['uploaded_files'] = []
                
                serializable_bots[bot_id] = serializable_bot
            
            with open(bots_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_bots, f, indent=2, ensure_ascii=False, default=str)
            
            st.session_state.bots = bots.copy()
            
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
                                
                                kb['uploaded_files'] = restored_files
                                kb['file_names'] = [f.name for f in restored_files]
                    
                    st.session_state.bots = bots.copy()
                    return bots
                    
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Bots file corrupted, trying backup: {e}")
                    
                    backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_bots_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            bots = json.load(f)
                        
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
            
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
            
            meta_path = os.path.join(file_dir, f"file_{file_index}_meta.json")
            metadata = {
                'name': file.name,
                'type': getattr(file, 'type', 'unknown'),
                'size': len(file.getvalue()),
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
                return True
            
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
            
            chat_data = {
                'user_id': user_id,
                'last_updated': datetime.now().isoformat(),
                'version': '1.2',
                'chat_history': chat_history
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
                        chat_history = chat_data['chat_history']
                    else:
                        chat_history = chat_data
                    
                    st.session_state.chat_history = chat_history.copy()
                    return chat_history
                    
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Chat history corrupted, trying backup: {e}")
                    
                    backup_file = os.path.join(self.data_dir, "backups", f"{user_id}_chats_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            chat_data = json.load(f)
                        
                        if isinstance(chat_data, dict) and 'chat_history' in chat_data:
                            chat_history = chat_data['chat_history']
                        else:
                            chat_history = chat_data
                        
                        st.session_state.chat_history = chat_history.copy()
                        st.info("Restored chat history from backup")
                        return chat_history
            
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
        
        st.session_state.data_manager = self
        
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
                'total_conversations': len(chat_history),
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
    
    def export_user_data(self, user_id: str) -> Optional[str]:
        try:
            export_data = {
                'user_config': self.load_user_config(user_id),
                'bots': self.load_user_bots(user_id),
                'chat_history': self.load_chat_history(user_id),
                'stats': self.get_user_stats(user_id),
                'export_date': datetime.now().isoformat(),
                'version': '1.2'
            }
            
            export_file = os.path.join(self.data_dir, "backups", f"{user_id}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return export_file
            
        except Exception as e:
            st.error(f"Error exporting user data: {e}")
            return None
    
    def import_user_data(self, user_id: str, import_file_path: str) -> bool:
        try:
            with open(import_file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            required_keys = ['user_config', 'bots', 'chat_history']
            if not all(key in import_data for key in required_keys):
                st.error("Invalid import file format")
                return False
            
            self.save_all_user_data(user_id)
            
            success = True
            success &= self.save_user_config(user_id, import_data['user_config'])
            success &= self.save_user_bots(user_id, import_data['bots'])
            success &= self.save_chat_history(user_id, import_data['chat_history'])
            
            if success:
                self.initialize_user_session(import_data['user_config'].get('username', 'User'))
                st.success("User data imported successfully!")
            
            return success
            
        except Exception as e:
            st.error(f"Error importing user data: {e}")
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
            
            usage['total_size'] = sum(usage.values()) - usage['total_size']
            
            return usage
            
        except Exception as e:
            st.error(f"Error calculating storage usage: {e}")
            return {}

class Database:
    def __init__(self):
        self.data_manager = DataManager()
    
    def save_bot(self, bot_id: str, config: Dict) -> bool:
        user_id = config.get('user_id')
        if not user_id:
            st.error("Bot config missing user_id")
            return False
        
        bots = self.data_manager.load_user_bots(user_id)
        
        bots[bot_id] = config
        
        return self.data_manager.save_user_bots(user_id, bots)
    
    def load_bot(self, bot_id: str) -> Optional[Dict]:
        users_dir = os.path.join(self.data_manager.data_dir, "users")
        
        for user_file in os.listdir(users_dir):
            if user_file.endswith('.json') and not user_file.endswith('_chats.json'):
                user_id = user_file[:-5]
                bots = self.data_manager.load_user_bots(user_id)
                
                if bot_id in bots:
                    return bots[bot_id]
        
        return None
    
    def delete_bot(self, bot_id: str) -> bool:
        users_dir = os.path.join(self.data_manager.data_dir, "users")
        
        for user_file in os.listdir(users_dir):
            if user_file.endswith('.json') and not user_file.endswith('_chats.json'):
                user_id = user_file[:-5]
                bots = self.data_manager.load_user_bots(user_id)
                
                if bot_id in bots:
                    del bots[bot_id]
                    
                    self.data_manager.delete_bot_files(user_id, bot_id)
                    
                    return self.data_manager.save_user_bots(user_id, bots)
        
        return False
    
    def get_user_bots(self, user_id: str) -> List[Dict]:
        bots = self.data_manager.load_user_bots(user_id)
        
        bot_list = []
        for bot_id, config in bots.items():
            config['id'] = bot_id
            bot_list.append(config)
        
        bot_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return bot_list
    
    def save_chat_history(self, bot_id: str, messages: List[Dict]) -> bool:
        users_dir = os.path.join(self.data_manager.data_dir, "users")
        
        for user_file in os.listdir(users_dir):
            if user_file.endswith('.json') and not user_file.endswith('_chats.json'):
                user_id = user_file[:-5]
                bots = self.data_manager.load_user_bots(user_id)
                
                if bot_id in bots:
                    chat_history = self.data_manager.load_chat_history(user_id)
                    
                    chat_history[bot_id] = messages
                    
                    return self.data_manager.save_chat_history(user_id, chat_history)
        
        return False
    
    def load_chat_history(self, bot_id: str) -> List[Dict]:
        users_dir = os.path.join(self.data_manager.data_dir, "users")
        
        for user_file in os.listdir(users_dir):
            if user_file.endswith('.json') and not user_file.endswith('_chats.json'):
                user_id = user_file[:-5]
                bots = self.data_manager.load_user_bots(user_id)
                
                if bot_id in bots:
                    chat_history = self.data_manager.load_chat_history(user_id)
                    
                    return chat_history.get(bot_id, [])
        
        return []
    
    def get_user_statistics(self, user_id: str) -> Dict:
        return self.data_manager.get_user_stats(user_id)
    
    def backup_user_data(self, user_id: str) -> Dict:
        backup = {
            'user_id': user_id,
            'backup_created': datetime.now().isoformat(),
            'bots': {},
            'chat_histories': {}
        }
        
        user_bots = self.get_user_bots(user_id)
        
        for bot in user_bots:
            bot_id = bot['id']
            backup['bots'][bot_id] = bot
            
            chat_history = self.load_chat_history(bot_id)
            if chat_history:
                backup['chat_histories'][bot_id] = chat_history
        
        return backup
    
    def restore_user_data(self, backup: Dict) -> bool:
        try:
            user_id = backup.get('user_id')
            if not user_id:
                return False
            
            for bot_id, bot_config in backup.get('bots', {}).items():
                self.save_bot(bot_id, bot_config)
            
            for bot_id, messages in backup.get('chat_histories', {}).items():
                self.save_chat_history(bot_id, messages)
            
            return True
        except Exception as e:
            st.error(f"Error restoring data: {str(e)}")
            return False

def auto_save_user_data():
    try:
        if hasattr(st.session_state, 'data_manager') and st.session_state.data_manager:
            st.session_state.data_manager.save_all_user_data()
    except Exception as e:
        st.warning(f"Auto-save failed: {str(e)}")

def get_data_manager():
    try:
        if 'data_manager' not in st.session_state:
            st.session_state.data_manager = DataManager()
        return st.session_state.data_manager
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
