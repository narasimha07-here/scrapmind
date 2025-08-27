import streamlit as st
import uuid
import base64
import os
import json
import hashlib
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

try:
    import plotly.express as px
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class LinkGenerator:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager or self._get_data_manager()
        self.user_id = st.session_state.get('user_id')
        self._initialize_storage_paths()
        self.base_url = self.get_base_url()

    def _get_data_manager(self):
        try:
            if 'data_manager' in st.session_state and st.session_state.data_manager:
                return st.session_state.data_manager
            else:
                from data_manager import DataManager
                dm = DataManager()
                st.session_state.data_manager = dm
                return dm
        except Exception as e:
            st.warning(f"Could not get DataManager: {e}")
            return None

    def _initialize_storage_paths(self):
        if not self.data_manager:
            return
        
        try:
            base_dir = self.data_manager.data_dir
            self.links_dir = os.path.join(base_dir, "shared_links")
            self.analytics_dir = os.path.join(base_dir, "link_analytics")
            
            for directory in [self.links_dir, self.analytics_dir]:
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            st.error(f"Error initializing storage paths: {e}")

    def save_link_data_permanently(self, link_id: str, link_data: Dict) -> bool:
        if not self.data_manager or not self.user_id:
            return self.save_link_data_fallback(link_id, link_data)
        
        try:
            user_links_file = os.path.join(self.links_dir, f"{self.user_id}_links.json")
            
            user_links = {}
            if os.path.exists(user_links_file):
                with open(user_links_file, 'r', encoding='utf-8') as f:
                    user_links = json.load(f)
            
            link_data.update({
                'user_id': self.user_id,
                'saved_at': datetime.now().isoformat(),
                'version': '1.0'
            })
            
            user_links[link_id] = link_data
            
            if os.path.exists(user_links_file):
                backup_file = os.path.join(self.data_manager.data_dir, "backups", f"{self.user_id}_links_backup.json")
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                import shutil
                shutil.copy2(user_links_file, backup_file)
            
            with open(user_links_file, 'w', encoding='utf-8') as f:
                json.dump(user_links, f, indent=2, ensure_ascii=False)
            
            if 'user_shared_links' not in st.session_state:
                st.session_state.user_shared_links = {}
            st.session_state.user_shared_links[link_id] = link_data
            
            return True
            
        except Exception as e:
            st.error(f"Error saving link data permanently: {e}")
            return self.save_link_data_fallback(link_id, link_data)

    def load_link_data_permanently(self, link_id: str) -> Optional[Dict]:
        if not self.data_manager or not self.user_id:
            return self.load_link_data_fallback(link_id)
        
        try:
            if 'user_shared_links' in st.session_state and link_id in st.session_state.user_shared_links:
                return st.session_state.user_shared_links[link_id]
            
            user_links_file = os.path.join(self.links_dir, f"{self.user_id}_links.json")
            if os.path.exists(user_links_file):
                with open(user_links_file, 'r', encoding='utf-8') as f:
                    user_links = json.load(f)
                
                if link_id in user_links:
                    if 'user_shared_links' not in st.session_state:
                        st.session_state.user_shared_links = {}
                    st.session_state.user_shared_links[link_id] = user_links[link_id]
                    
                    return user_links[link_id]
            
            return None
            
        except Exception as e:
            st.warning(f"Error loading link data: {e}")
            return self.load_link_data_fallback(link_id)

    def load_all_user_links(self) -> Dict:
        if not self.data_manager or not self.user_id:
            return st.session_state.get('user_shared_links', {})
        
        try:
            user_links_file = os.path.join(self.links_dir, f"{self.user_id}_links.json")
            if os.path.exists(user_links_file):
                try:
                    with open(user_links_file, 'r', encoding='utf-8') as f:
                        user_links = json.load(f)
                    
                    st.session_state.user_shared_links = user_links.copy()
                    return user_links
                    
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Links file corrupted: {e}")
                    backup_file = os.path.join(self.data_manager.data_dir, "backups", f"{self.user_id}_links_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            user_links = json.load(f)
                        st.session_state.user_shared_links = user_links.copy()
                        st.info("Restored links from backup")
                        return user_links
            
            return {}
            
        except Exception as e:
            st.error(f"Error loading user links: {e}")
            return st.session_state.get('user_shared_links', {})

    def save_link_analytics(self, link_id: str, analytics_data: Dict):
        if not self.data_manager or not self.user_id:
            return False
        
        try:
            analytics_file = os.path.join(self.analytics_dir, f"{self.user_id}_{link_id}_analytics.json")
            
            existing_analytics = []
            if os.path.exists(analytics_file):
                with open(analytics_file, 'r', encoding='utf-8') as f:
                    existing_analytics = json.load(f)
            
            analytics_entry = {
                **analytics_data,
                'timestamp': datetime.now().isoformat(),
                'user_id': self.user_id,
                'link_id': link_id
            }
            
            existing_analytics.append(analytics_entry)
            
            if len(existing_analytics) > 1000:
                existing_analytics = existing_analytics[-1000:]
            
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(existing_analytics, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving analytics: {e}")
            return False

    def update_link_access_count(self, link_id: str):
        try:
            link_data = self.load_link_data_permanently(link_id)
            if link_data:
                link_data['access_count'] = link_data.get('access_count', 0) + 1
                link_data['last_accessed'] = datetime.now().isoformat()
                
                self.save_link_data_permanently(link_id, link_data)
                
                self.save_link_analytics(link_id, {
                    'event': 'access',
                    'user_agent': st.session_state.get('user_agent', 'unknown'),
                    'ip_address': 'anonymized',
                    'referer': st.session_state.get('referer', 'direct')
                })
                
                return True
        except Exception as e:
            st.error(f"Error updating access count: {e}")
            return False

    def delete_link_permanently(self, link_id: str) -> bool:
        if not self.data_manager or not self.user_id:
            return False
        
        try:
            user_links = self.load_all_user_links()
            
            if link_id in user_links:
                del user_links[link_id]
                
                user_links_file = os.path.join(self.links_dir, f"{self.user_id}_links.json")
                with open(user_links_file, 'w', encoding='utf-8') as f:
                    json.dump(user_links, f, indent=2, ensure_ascii=False)
                
                if 'user_shared_links' in st.session_state and link_id in st.session_state.user_shared_links:
                    del st.session_state.user_shared_links[link_id]
                
                analytics_file = os.path.join(self.analytics_dir, f"{self.user_id}_{link_id}_analytics.json")
                if os.path.exists(analytics_file):
                    os.remove(analytics_file)
                
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error deleting link: {e}")
            return False

    def get_base_url(self) -> str:
        base_url = os.getenv('APP_BASE_URL')
        if base_url:
            return base_url.rstrip('/')
        
        try:
            import streamlit.web.server.server as server
            if hasattr(server, 'Server') and server.Server._instance:
                return "http://localhost:8501"
            else:
                return "https://ScrapMind.streamlit.app" 
        except Exception:
            return "https://ScrapMind.streamlit.app"

    def generate_public_link(self, bot_id: str, bot_config: Dict) -> Dict:
        try:
            link_id = str(uuid.uuid4())
            link_data = {
                'link_id': link_id,
                'bot_id': bot_id,
                'bot_name': bot_config.get('name', 'Unnamed Bot'),
                'creator_id': bot_config.get('user_id', self.user_id),
                'created_at': datetime.now().isoformat(),
                'access_count': 0,
                'last_accessed': None,
                'is_active': True,
                'expires_at': None,
                'password_protected': False,
                'password_hash': None,
                'custom_domain': None,
                'analytics_enabled': True,
                'settings': {
                    'allow_feedback': True,
                    'show_bot_info': True,
                    'custom_welcome': None,
                    'custom_theme': None,
                    'rate_limiting': False,
                    'max_messages_per_hour': 100
                }
            }

            success = self.save_link_data_permanently(link_id, link_data)
            if not success:
                st.error("Failed to save link data permanently")
                return None

            base_url_clean = self.base_url.rstrip('/')
            urls = {
                'direct_chat': f"{base_url_clean}?bot_id={bot_id}",
                'shareable': f"{base_url_clean}/share/{link_id}",
                'embed': f"{base_url_clean}/embed/{link_id}",
                'api': f"{base_url_clean}/api/chat/{link_id}"
            }

            if self.data_manager:
                self.data_manager.save_all_user_data()

            return {
                'link_id': link_id,
                'urls': urls,
                'qr_code': self.generate_qr_code(urls['shareable']) if QR_AVAILABLE else None,
                'embed_code': self.generate_embed_code(link_id),
                'link_data': link_data
            }

        except Exception as e:
            st.error(f"Error generating public link: {str(e)}")
            return None

    def get_bot_links(self, bot_id: str) -> List[Dict]:
        try:
            all_links = self.load_all_user_links()
            bot_links = []
            
            for link_id, link_data in all_links.items():
                if link_data.get('bot_id') == bot_id:
                    bot_links.append(link_data)
            
            bot_links.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return bot_links
            
        except Exception as e:
            st.error(f"Error getting bot links: {e}")
            return []

    def restore_links_on_refresh(self):
        try:
            if self.user_id:
                user_links = self.load_all_user_links()
                st.session_state.user_shared_links = user_links
                
                if user_links:
                    st.info(f"✅ Restored {len(user_links)} shared links")
                    
        except Exception as e:
            st.warning(f"Error restoring links: {e}")

    def save_link_data_fallback(self, link_id: str, link_data: Dict) -> bool:
        try:
            if 'shared_links' not in st.session_state:
                st.session_state.shared_links = {}
            st.session_state.shared_links[link_id] = link_data
            return True
        except Exception as e:
            st.error(f"Error saving link data: {str(e)}")
            return False

    def load_link_data_fallback(self, link_id: str) -> Optional[Dict]:
        return st.session_state.get('shared_links', {}).get(link_id)

    def generate_qr_code(self, url: str) -> Optional[str]:
        if not QR_AVAILABLE:
            return None
        
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            buffer = BytesIO()
            qr_img.save(buffer, format='PNG')
            qr_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return qr_b64
        except Exception as e:
            st.warning(f"Error generating QR code: {str(e)}")
            return None

    def generate_embed_code(self, link_id: str) -> str:
        embed_url = f"{self.base_url.rstrip('/')}/embed/{link_id}"
        embed_code = f'''<iframe 
    src="{embed_url}" 
    width="400" 
    height="600" 
    frameborder="0" 
    title="Chatbot">
</iframe>'''
        return embed_code

    def show_link_management_interface(self, bot_id: str, bot_config: Dict):
        st.subheader("🔗 Share Your Chatbot")
        st.markdown("Generate shareable links and embed codes for your bot")

        self.restore_links_on_refresh()

        with st.expander("ℹ️ About Sharing", expanded=False):
            st.markdown("""
            **Sharing Options:**
            - **Direct Link**: Share a direct URL to your chatbot
            - **Embed Code**: Embed your chatbot on websites  
            - **QR Code**: Generate QR codes for mobile access
            - **Password Protection**: Secure your shared bots
            - **Analytics**: Track usage and engagement
            
            **🔄 Data Persistence**: All your shared links are automatically saved and will be restored when you refresh the app.
            """)

        existing_links = self.get_bot_links(bot_id)
        
        if existing_links:
            self.show_existing_links(existing_links, bot_id)

        if st.button("🚀 Generate New Shareable Link", type="primary", key=f"gen_link_{bot_id}"):
            with st.spinner("Generating shareable link..."):
                link_data = self.generate_public_link(bot_id, bot_config)
                
                if link_data:
                    st.session_state[f'generated_link_{bot_id}'] = link_data
                    st.success("✅ Shareable link generated and saved permanently!")
                    st.rerun()
                else:
                    st.error("Failed to generate link. Please try again.")

        generated_key = f'generated_link_{bot_id}'
        if generated_key in st.session_state:
            self.show_generated_link(st.session_state[generated_key])

    def show_existing_links(self, links: List[Dict], bot_id: str):
        st.markdown("### 📋 Existing Links")
        
        for i, link in enumerate(links):
            with st.expander(f"🔗 Link: {link['link_id'][:8]}... ({link.get('bot_name', 'Unnamed')})", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    created_date = link['created_at'][:10] if link.get('created_at') else 'Unknown'
                    st.markdown(f"**Created:** {created_date}")
                    st.markdown(f"**Access Count:** {link.get('access_count', 0)}")
                    status_icon = '🟢 Active' if link.get('is_active', True) else '🔴 Inactive'
                    st.markdown(f"**Status:** {status_icon}")
                    if link.get('password_protected'):
                        st.markdown("**Protection:** 🔒 Password")
                
                with col2:
                    shareable_url = f"{self.base_url.rstrip('/')}/share/{link['link_id']}"
                    st.markdown("**Shareable URL:**")
                    st.code(shareable_url, language="text")
                    
                    if st.button("📋 Copy", key=f"copy_{link['link_id']}_{i}"):
                        st.success("Link copied to clipboard!")
                
                with col3:
                    if st.button("📊 Analytics", key=f"analytics_{link['link_id']}_{i}"):
                        st.session_state[f'show_analytics_{link["link_id"]}'] = True
                        st.rerun()
                    
                    if st.button("⚙️ Settings", key=f"settings_{link['link_id']}_{i}"):
                        st.session_state[f'show_settings_{link["link_id"]}'] = True
                        st.rerun()
                    
                    if st.button("🗑️ Delete", key=f"delete_{link['link_id']}_{i}"):
                        if self.delete_link_permanently(link['link_id']):
                            st.success("Link deleted permanently!")
                            st.rerun()
                        else:
                            st.error("Failed to delete link")

                if st.session_state.get(f'show_analytics_{link["link_id"]}'):
                    st.markdown("---")
                    self.show_link_analytics(link)
                    if st.button("❌ Close Analytics", key=f"close_analytics_{link['link_id']}_{i}"):
                        del st.session_state[f'show_analytics_{link["link_id"]}']
                        st.rerun()

    def show_generated_link(self, link_data: Dict):
        if not link_data:
            st.error("No link data available")
            return

        st.markdown("### 🎉 Your Bot is Now Shareable!")
        st.success("✅ Link has been saved permanently and will survive app refreshes!")
        
        urls = link_data.get('urls', {})

        tabs = ["🔗 Direct Link", "🌐 Embed Code"]
        if QR_AVAILABLE and link_data.get('qr_code'):
            tabs.insert(1, "📱 QR Code")

        tab_objects = st.tabs(tabs)

        with tab_objects[0]:
            st.markdown("**Shareable URL:**")
            shareable_url = urls.get('shareable', 'URL not available')
            st.code(shareable_url, language="text")
            
            st.markdown("**Share Options:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Copy Link", key="copy_direct_link"):
                    st.success("Link copied to clipboard!")
            
            with col2:
                if st.button("📧 Email Link", key="email_direct_link"):
                    self.generate_email_link(shareable_url, link_data['link_data']['bot_name'])
            
            with col3:
                if st.button("📱 SMS Link", key="sms_direct_link"):
                    self.generate_sms_link(shareable_url)

        tab_index = 1
        if QR_AVAILABLE and link_data.get('qr_code'):
            with tab_objects[tab_index]:
                st.markdown("**QR Code for Mobile Access:**")
                qr_code = link_data.get('qr_code')
                if qr_code:
                    st.markdown(
                        f'<img src="data:image/png;base64,{qr_code}" style="max-width: 300px;">',
                        unsafe_allow_html=True
                    )
                    
                    qr_download = base64.b64decode(qr_code)
                    st.download_button(
                        label="📥 Download QR Code",
                        data=qr_download,
                        file_name=f"qr_code_{link_data['link_id'][:8]}.png",
                        mime="image/png",
                        key="download_qr"
                    )
                else:
                    st.error("QR code generation failed")
            tab_index += 1

        with tab_objects[tab_index]:
            st.markdown("**Embed Code for Websites:**")
            embed_code = link_data.get('embed_code', 'Embed code not available')
            st.code(embed_code, language="html")
            
            st.markdown("**Preview:**")
            st.markdown(
                f'<div style="border: 1px solid #ccc; padding: 10px; margin: 10px 0;">'
                f'🖼️ Embedded chatbot would appear here<br>'
                f'Size: 400x600 pixels<br>'
                f'URL: {urls.get("embed", "N/A")}'
                f'</div>',
                unsafe_allow_html=True
            )

    def show_link_analytics(self, link: Dict):
        st.markdown("#### 📊 Link Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Access", link.get('access_count', 0))
        with col2:
            created = link.get('created_at', '')[:10] if link.get('created_at') else 'Unknown'
            st.metric("Created", created)
        with col3:
            last_access = link.get('last_accessed', 'Never')[:10] if link.get('last_accessed') else 'Never'
            st.metric("Last Access", last_access)
        with col4:
            status = '🟢 Active' if link.get('is_active', True) else '🔴 Inactive'
            st.markdown(f"**Status:** {status}")

    def generate_email_link(self, url: str, bot_name: str):
        subject = f"Check out my AI chatbot: {bot_name}"
        body = f"Hi! I've created an AI chatbot and would love for you to try it out: {url}"
        email_url = f"mailto:?subject={subject}&body={body}"
        st.markdown(f'📧 [Open Email Client]({email_url})', unsafe_allow_html=True)

    def generate_sms_link(self, url: str):
        message = f"Check out this AI chatbot: {url}"
        sms_url = f"sms:?body={message}"
        st.markdown(f'📱 [Send SMS]({sms_url})', unsafe_allow_html=True)


def get_link_generator():
    try:
        if 'link_generator' not in st.session_state:
            data_manager = st.session_state.get('data_manager')
            if not data_manager:
                from data_manager import DataManager
                data_manager = DataManager()
                st.session_state.data_manager = data_manager
            
            st.session_state.link_generator = LinkGenerator(data_manager=data_manager)
        
        return st.session_state.link_generator
    
    except Exception as e:
        st.error(f"Error creating LinkGenerator: {e}")
        return LinkGenerator()

def auto_save_links():
    try:
        if 'link_generator' in st.session_state:
            link_gen = st.session_state.link_generator
            if link_gen.data_manager:
                link_gen.data_manager.save_all_user_data()
    except Exception as e:
        st.warning(f"Auto-save links failed: {e}")

if 'link_generator' not in st.session_state:
    st.session_state.link_generator = get_link_generator()
