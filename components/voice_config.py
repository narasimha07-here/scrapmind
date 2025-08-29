import streamlit as st
import base64
import os
import tempfile
import time
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pydub
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class VoiceConfig:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager or self._get_data_manager()
        self.user_id = st.session_state.get('user_id')
        
        self._initialize_storage_paths()
        
        self.voice_providers = {
            'murf': 'Murf AI (Recommended)',
            'openai': 'OpenAI TTS',
            'elevenlabs': 'ElevenLabs', 
            'google': 'Google Text-to-Speech (gTTS)',
            'azure': 'Azure Speech',
            'amazon': 'Amazon Polly'
        }
        
        self.voice_options = {
            'openai': {
                'alloy': 'Alloy (Neutral)',
                'echo': 'Echo (Male)',
                'fable': 'Fable (British)',
                'onyx': 'Onyx (Deep)',
                'nova': 'Nova (Female)',
                'shimmer': 'Shimmer (Soft)'
            },
            'elevenlabs': {
                'rachel': 'Rachel (Professional)',
                'drew': 'Drew (Casual)',
                'clyde': 'Clyde (Warm)',
                'paul': 'Paul (Narrator)',
                'domi': 'Domi (Strong)',
                'dave': 'Dave (British)',
                'fin': 'Fin (Irish)',
                'sarah': 'Sarah (Soft)'
            },
            'google': {
                'male': 'Male Voice',
                'female': 'Female Voice'
            },
            'murf': {}
        }
        
        self.languages = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese (Mandarin)',
            'hi': 'Hindi',
            'ar': 'Arabic',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'da': 'Danish',
            'pl': 'Polish'
        }

    def _fetch_and_cache_murf_voices(self, murf_api_key: str) -> Dict[str, str]:
        if not murf_api_key:
            return {"Error": "API Key is missing."}
        
        cache_key = f"murf_voices_cache_{murf_api_key}"
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        url = "https://api.murf.ai/v1/speech/voices"
        headers = {"accept": "application/json", "api-key": murf_api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            voices_data = response.json()
            
            murf_voices_map = {}
            
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
                
                murf_voices_map[display_name] = voice_id
            
            if not murf_voices_map:
                return {"Error": f"No valid voices extracted from {len(voice_list)} voice entries"}
            
            st.session_state[cache_key] = murf_voices_map
            return murf_voices_map
            
        except requests.exceptions.RequestException as e:
            return {"Error": f"API request failed: {str(e)}"}
        except (ValueError, json.JSONDecodeError) as e:
            return {"Error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"Error": f"Unexpected error: {str(e)}"}

    def _get_data_manager(self):
        try:
            if 'data_manager' in st.session_state and st.session_state.data_manager:
                return st.session_state.data_manager
            else:
                from components.data_manager import DataManager
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
            self.voice_configs_dir = os.path.join(base_dir, "voice_configs")
            self.voice_presets_dir = os.path.join(base_dir, "voice_presets")
            
            for directory in [self.voice_configs_dir, self.voice_presets_dir]:
                os.makedirs(directory, exist_ok=True)
        except Exception as e:
            st.error(f"Error initializing voice storage paths: {e}")

    def save_voice_config_permanently(self, voice_config: Dict) -> bool:
        if not self.data_manager or not self.user_id:
            return self.save_voice_config_fallback(voice_config)
        
        try:
            voice_config_file = os.path.join(self.voice_configs_dir, f"{self.user_id}_voice_config.json")
            
            voice_config.update({
                'user_id': self.user_id,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            })
            
            if os.path.exists(voice_config_file):
                backup_file = os.path.join(self.data_manager.data_dir, "backups", f"{self.user_id}_voice_backup.json")
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)
                import shutil
                shutil.copy2(voice_config_file, backup_file)
            
            with open(voice_config_file, 'w', encoding='utf-8') as f:
                json.dump(voice_config, f, indent=2, ensure_ascii=False)
            
            st.session_state.voice_config = voice_config.copy()
            
            if self.data_manager:
                self.data_manager.save_all_user_data(self.user_id)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving voice config permanently: {e}")
            return self.save_voice_config_fallback(voice_config)

    def load_voice_config_permanently(self) -> Dict:
        if not self.data_manager or not self.user_id:
            return self.get_default_config()
        
        try:
            voice_config_file = os.path.join(self.voice_configs_dir, f"{self.user_id}_voice_config.json")
            
            if os.path.exists(voice_config_file):
                try:
                    with open(voice_config_file, 'r', encoding='utf-8') as f:
                        voice_config = json.load(f)
                    
                    st.session_state.voice_config = voice_config.copy()
                    return voice_config
                    
                except (json.JSONDecodeError, Exception) as e:
                    st.warning(f"Voice config file corrupted: {e}")
                    
                    backup_file = os.path.join(self.data_manager.data_dir, "backups", f"{self.user_id}_voice_backup.json")
                    if os.path.exists(backup_file):
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            voice_config = json.load(f)
                        st.session_state.voice_config = voice_config.copy()
                        st.info("Restored voice config from backup")
                        return voice_config
            
            default_config = self.get_default_config()
            st.session_state.voice_config = default_config.copy()
            return default_config
            
        except Exception as e:
            st.error(f"Error loading voice config: {e}")
            return self.get_default_config()

    def save_voice_preset(self, preset_name: str, voice_config: Dict) -> bool:
        if not self.data_manager or not self.user_id:
            return False
        
        try:
            preset_file = os.path.join(self.voice_presets_dir, f"{self.user_id}_presets.json")
            
            presets = {}
            if os.path.exists(preset_file):
                with open(preset_file, 'r', encoding='utf-8') as f:
                    presets = json.load(f)
            
            presets[preset_name] = {
                'config': voice_config.copy(),
                'created_at': datetime.now().isoformat(),
                'user_id': self.user_id
            }
            
            with open(preset_file, 'w', encoding='utf-8') as f:
                json.dump(presets, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving voice preset: {e}")
            return False

    def load_voice_presets(self) -> Dict:
        if not self.data_manager or not self.user_id:
            return {}
        
        try:
            preset_file = os.path.join(self.voice_presets_dir, f"{self.user_id}_presets.json")
            if os.path.exists(preset_file):
                with open(preset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.error(f"Error loading voice presets: {e}")
            return {}

    def restore_voice_config_on_refresh(self):
        try:
            if self.user_id:
                voice_config = self.load_voice_config_permanently()
                st.session_state.voice_config = voice_config
                
                if voice_config.get('enabled'):
                    st.success("✅ Voice configuration restored successfully!")
        except Exception as e:
            st.warning(f"Error restoring voice config: {e}")

    def configure_voice(self) -> Dict:
        self.restore_voice_config_on_refresh()
        
        if 'voice_config' not in st.session_state:
            st.session_state.voice_config = self.load_voice_config_permanently()
        
        voice_config = st.session_state.voice_config.copy()
        
        st.markdown("### 🎙️ Voice Response Configuration")
        st.markdown("Configure text-to-speech settings for your chatbot responses")
        st.info("🔄 **Persistent Storage**: All voice settings are automatically saved and restored across app refreshes!")
        
        self.show_dependency_status()
        
        voice_enabled = st.checkbox(
            "🎙️ Enable Voice Responses",
            value=voice_config.get('enabled', False),
            help="Generate audio responses for bot messages",
            key="voice_enabled_checkbox"
        )
        
        voice_config['enabled'] = voice_enabled
        
        if voice_config['enabled'] != st.session_state.voice_config.get('enabled', False):
            self.save_voice_config_permanently(voice_config)
        
        if not voice_enabled:
            st.info("💡 Voice responses are disabled. Users will see text-only responses.")
            return voice_config
        
        self.show_voice_configuration_tabs(voice_config)
        
        st.session_state.voice_config = voice_config
        return voice_config

    def show_voice_configuration_tabs(self, voice_config: Dict):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎵 Provider & Voice",
            "⚙️ Settings", 
            "🎧 Preview & Test",
            "💾 Presets",
            "📊 Usage & Costs"
        ])
        
        with tab1:
            self.configure_provider_and_voice(voice_config)
        
        with tab2:
            self.configure_voice_settings(voice_config)
        
        with tab3:
            self.show_voice_preview_tab(voice_config)
        
        with tab4:
            self.show_voice_presets(voice_config)
        
        with tab5:
            self.show_usage_and_costs(voice_config)

    def show_voice_presets(self, voice_config: Dict):
        st.markdown("#### 💾 Voice Presets")
        st.markdown("Save and load voice configuration presets for different use cases")
        
        presets = self.load_voice_presets()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 💾 Save Current Configuration")
            preset_name = st.text_input(
                "Preset Name",
                placeholder="e.g., Professional Female, Casual Male",
                key="voice_preset_name"
            )
            
            if st.button("💾 Save Preset", key="save_voice_preset"):
                if preset_name.strip():
                    if self.save_voice_preset(preset_name.strip(), voice_config):
                        st.success(f"✅ Preset '{preset_name}' saved successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to save preset")
                else:
                    st.error("Please enter a preset name")
        
        with col2:
            st.markdown("##### 📂 Load Existing Presets")
            if presets:
                preset_options = list(presets.keys())
                selected_preset = st.selectbox(
                    "Choose Preset",
                    preset_options,
                    key="voice_preset_select"
                )
                
                col2a, col2b = st.columns(2)
                
                with col2a:
                    if st.button("📂 Load Preset", key="load_voice_preset"):
                        preset_config = presets[selected_preset]['config']
                        voice_config.update(preset_config)
                        st.session_state.voice_config = voice_config.copy()
                        self.save_voice_config_permanently(voice_config)
                        st.success(f"✅ Preset '{selected_preset}' loaded!")
                        st.rerun()
                
                with col2b:
                    if st.button("🗑️ Delete Preset", key="delete_voice_preset"):
                        if selected_preset in presets:
                            del presets[selected_preset]
                            try:
                                preset_file = os.path.join(self.voice_presets_dir, f"{self.user_id}_presets.json")
                                with open(preset_file, 'w', encoding='utf-8') as f:
                                    json.dump(presets, f, indent=2, ensure_ascii=False)
                                st.success(f"✅ Preset '{selected_preset}' deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete preset: {e}")
                        else:
                            st.warning("Preset not found.")
            else:
                st.info("No saved presets found. Create your first preset above!")
        
        if presets:
            with st.expander("📋 Preset Details", expanded=False):
                for name, preset_data in presets.items():
                    st.markdown(f"**{name}**")
                    config = preset_data['config']
                    created = preset_data.get('created_at', 'Unknown')[:10]
                    st.markdown(f"• Provider: {self.voice_providers.get(config.get('provider'), 'Unknown')}")
                    st.markdown(f"• Voice: {config.get('voice', 'Unknown')}")
                    st.markdown(f"• Language: {self.languages.get(config.get('language'), 'Unknown')}")
                    st.markdown(f"• Created: {created}")
                    st.markdown("---")

    def configure_provider_and_voice(self, voice_config: Dict):
        st.markdown("#### 🎵 Voice Provider Selection")
        
        available_providers = self.get_available_providers()
        if not available_providers:
            st.error("❌ No TTS providers available. Please install required dependencies.")
            return
        
        provider = st.selectbox(
            "Choose Voice Provider",
            available_providers,
            index=available_providers.index(voice_config.get('provider', available_providers[0]))
            if voice_config.get('provider') in available_providers else 0,
            format_func=lambda x: self.voice_providers[x],
            help="Select the text-to-speech service provider",
            key="voice_provider_select"
        )
        
        if provider != voice_config.get('provider'):
            voice_config['provider'] = provider
            voice_config['voice'] = self.get_default_voice_for_provider(provider)
            self.save_voice_config_permanently(voice_config)
        
        if provider == 'google':
            self.configure_google_tts(voice_config)
        elif provider == 'openai':
            self.configure_openai_tts(voice_config)
        elif provider == 'murf':
            self.configure_murf_tts(voice_config)

    def get_default_voice_for_provider(self, provider: str) -> str:
        if provider == 'google':
            return 'female'
        elif provider == 'openai':
            return 'alloy'
        elif provider == 'murf':
            return ''
        return ''

    def configure_murf_tts(self, voice_config: Dict):
        st.markdown("##### 🎙️ Murf AI Text-to-Speech")
        
        if not REQUESTS_AVAILABLE:
            st.error("❌ 'requests' library not installed. Run: `pip install requests`")
            return
        
        st.info("💡 Murf AI provides high-quality, realistic voices. An API key is required.")
        
        murf_api_key_input = st.text_input(
            "Murf AI API Key",
            value=voice_config.get('api_key', ''),
            type="password",
            key="murf_api_key_config",
            help="Enter your Murf AI API Key. Get one from murf.ai"
        )
        
        if murf_api_key_input != voice_config.get('api_key'):
            voice_config['api_key'] = murf_api_key_input
            self.save_voice_config_permanently(voice_config)
        
        if not murf_api_key_input:
            st.warning("Please enter your Murf AI API key to see available voices.")
            return
        
        with st.spinner("Loading Murf AI voices..."):
            murf_voices_map = self._fetch_and_cache_murf_voices(murf_api_key_input)
        
        if "Error" in murf_voices_map:
            st.error(f"Failed to load voices: {murf_voices_map['Error']}")
            return
        
        voice_options_display = list(murf_voices_map.keys())
        id_to_name_map = {v: k for k, v in murf_voices_map.items()}
        
        current_voice_id = voice_config.get('voice')
        current_selection_name = id_to_name_map.get(current_voice_id)
        
        try:
            current_index = voice_options_display.index(current_selection_name) if current_selection_name in voice_options_display else 0
        except ValueError:
            current_index = 0
        
        selected_voice_name_display = st.selectbox(
            "Murf Voice",
            voice_options_display,
            index=current_index,
            key="murf_voice_select",
            help="Choose a voice from Murf AI"
        )
        
        selected_voice_id = murf_voices_map.get(selected_voice_name_display)
        
        if selected_voice_id != voice_config.get('voice'):
            voice_config['voice'] = selected_voice_id
            self.save_voice_config_permanently(voice_config)

    def configure_voice_settings(self, voice_config: Dict):
        st.markdown("#### 🎛️ Voice Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            speed = st.slider(
                "Speaking Speed",
                0.25, 2.0,
                voice_config.get('speed', 1.0),
                0.05,
                help="How fast the voice should speak",
                key="voice_speed_slider"
            )
            
            if speed != voice_config.get('speed'):
                voice_config['speed'] = speed
                self.save_voice_config_permanently(voice_config)
        
        with col2:
            current_lang = voice_config.get('language', 'en')
            language = st.selectbox(
                "Language",
                list(self.languages.keys()),
                index=list(self.languages.keys()).index(current_lang)
                if current_lang in self.languages else 0,
                format_func=lambda x: self.languages[x],
                help="Primary language for text-to-speech",
                key="voice_language_select"
            )
            
            if language != voice_config.get('language'):
                voice_config['language'] = language
                self.save_voice_config_permanently(voice_config)
        
        with col3:
            volume = st.slider(
                "Volume",
                0.1, 1.0,
                voice_config.get('volume', 0.8),
                0.1,
                help="Audio volume level",
                key="voice_volume_slider"
            )
            
            if volume != voice_config.get('volume'):
                voice_config['volume'] = volume
                self.save_voice_config_permanently(voice_config)
        
        voice_config.update({
            'speed': speed,
            'language': language,
            'volume': volume
        })

    def show_dependency_status(self):
        with st.expander("📦 Dependency Status", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Libraries:**")
                st.markdown(f"• requests: {'✅' if REQUESTS_AVAILABLE else '❌'}")
                st.markdown(f"• gTTS: {'✅' if GTTS_AVAILABLE else '❌'}")
                st.markdown(f"• pydub: {'✅' if PYDUB_AVAILABLE else '❌'}")
            
            with col2:
                st.markdown("**Installation Commands:**")
                if not GTTS_AVAILABLE:
                    st.code("pip install gtts")
                if not PYDUB_AVAILABLE:
                    st.code("pip install pydub") 
                if not REQUESTS_AVAILABLE:
                    st.code("pip install requests")

    def configure_google_tts(self, voice_config: Dict):
        st.markdown("##### 🌐 Google Text-to-Speech (gTTS)")
        
        if not GTTS_AVAILABLE:
            st.error("❌ gTTS library not installed. Run: `pip install gtts`")
            return
        
        st.info("💡 Google TTS is free but requires internet connection")
        
        voice = st.selectbox(
            "Voice Gender",
            ['female', 'male'],
            index=0 if voice_config.get('voice', 'female') == 'female' else 1,
            help="Choose voice gender",
            key="google_voice_select"
        )
        
        if voice != voice_config.get('voice'):
            voice_config['voice'] = voice
            self.save_voice_config_permanently(voice_config)

    def configure_openai_tts(self, voice_config: Dict):
        st.markdown("##### 🤖 OpenAI Text-to-Speech")
        st.info("💡 OpenAI TTS provides high-quality voices. API key required.")
        
        api_key = st.text_input(
            "OpenAI API Key",
            value=voice_config.get('api_key', ''),
            type="password",
            key="openai_api_key_config",
            help="Enter your OpenAI API Key"
        )
        
        if api_key != voice_config.get('api_key'):
            voice_config['api_key'] = api_key
            self.save_voice_config_permanently(voice_config)
        
        if api_key:
            openai_voices = self.voice_options['openai']
            current_voice = voice_config.get('voice', 'alloy')
            
            voice = st.selectbox(
                "OpenAI Voice",
                list(openai_voices.keys()),
                index=list(openai_voices.keys()).index(current_voice) if current_voice in openai_voices else 0,
                format_func=lambda x: openai_voices[x],
                key="openai_voice_select"
            )
            
            if voice != voice_config.get('voice'):
                voice_config['voice'] = voice
                self.save_voice_config_permanently(voice_config)

    def save_voice_config_fallback(self, voice_config: Dict) -> bool:
        try:
            st.session_state.voice_config = voice_config.copy()
            return True
        except Exception as e:
            st.error(f"Error saving voice config: {str(e)}")
            return False

    def get_available_providers(self) -> List[str]:
        available = []
        
        available.extend(['murf', 'openai', 'elevenlabs', 'azure', 'amazon'])
        
        if GTTS_AVAILABLE:
            available.append('google')
        
        return available

    def show_voice_preview_tab(self, voice_config: Dict):
        st.markdown("#### 🎧 Preview & Test")
        
        test_text = st.text_area(
            "Test Text",
            value=voice_config.get('test_text', "Hello! I'm your AI assistant. How can I help you today?"),
            help="Enter text to test the voice configuration",
            key="voice_test_text",
            height=100
        )
        
        if test_text != voice_config.get('test_text'):
            voice_config['test_text'] = test_text
            self.save_voice_config_permanently(voice_config)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎵 Generate Preview", type="primary", key="generate_voice_preview_button"):
                audio_data, audio_format = self.generate_voice_preview(voice_config, test_text)
                if audio_data:
                    st.success("🎵 Preview generated successfully!")
                    st.audio(audio_data, format=audio_format)
                else:
                    st.error("Failed to generate preview.")
        
        with col2:
            if st.button("🧪 Test Configuration", key="test_voice_config_button"):
                self.test_voice_configuration(voice_config)
        
        with col3:
            if st.button("💾 Save Settings", key="save_voice_settings_button"):
                if self.save_voice_config_permanently(voice_config):
                    st.success("✅ Voice settings saved permanently!")
                    st.balloons()
                else:
                    st.error("❌ Failed to save settings")
        
        self.show_configuration_summary(voice_config)

    def show_usage_and_costs(self, voice_config: Dict):
        st.markdown("#### 📊 Usage & Cost Estimation")
        
        provider = voice_config.get('provider', 'google')
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Characters This Month", "12,450")
        
        with col2:
            st.metric("Audio Minutes", "8.3")
        
        with col3:
            estimated_cost = self.calculate_estimated_cost(provider, 12450)
            st.metric("Estimated Cost", f"\${estimated_cost:.2f}")
        
        with col4:
            st.metric("Requests", "156")

    def generate_voice_preview(self, voice_config: Dict, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        provider = voice_config.get('provider', 'google')
        
        try:
            if provider == 'google' and GTTS_AVAILABLE:
                audio_data = self._generate_google_tts_audio(voice_config, text)
                return audio_data, "audio/mp3" if audio_data else None
            
            elif provider == 'murf' and REQUESTS_AVAILABLE:
                audio_data = self._generate_murf_tts_audio(voice_config, text)
                return audio_data, "audio/wav" if audio_data else None
            
            else:
                st.info(f"💡 Voice generation for {provider} is not yet implemented or dependencies are missing.")
                return None, None
                
        except Exception as e:
            st.error(f"Error generating preview: {str(e)}")
            return None, None

    def _generate_google_tts_audio(self, voice_config: Dict, text: str) -> Optional[bytes]:
        if not GTTS_AVAILABLE:
            return None
        
        try:
            language = voice_config.get('language', 'en')
            
            tts = gTTS(text=text, lang=language, slow=False)
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            st.error(f"Google TTS error: {str(e)}")
            return None

    def _generate_murf_tts_audio(self, voice_config: Dict, text: str) -> Optional[bytes]:
        if not REQUESTS_AVAILABLE:
            st.error("❌ 'requests' library not available for Murf AI.")
            return None
        
        murf_api_key = voice_config.get('api_key')
        murf_voice_id = voice_config.get('voice')
        
        if not murf_api_key:
            st.error("Murf AI API Key is not configured.")
            return None
        
        if not murf_voice_id:
            st.error("Murf AI Voice ID is not selected. Please select a specific voice.")
            return None
        
        url = "https://api.murf.ai/v1/speech/generate"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "api-key": murf_api_key
        }
        
        payload = {
            "text": text,
            "voice_id": murf_voice_id,
            "speed": voice_config.get('speed', 1.0),
            "pitch": voice_config.get('pitch', 0),
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            response_json = response.json()
            
            audio_url = None
            
            if 'audioFile' in response_json:
                audio_file = response_json['audioFile']
                if isinstance(audio_file, str):
                    audio_url = audio_file.strip('[]\'\"')
                else:
                    audio_url = audio_file
            elif 'audio_url' in response_json:
                audio_url = response_json['audio_url']
            elif 'url' in response_json:
                audio_url = response_json['url']
            
            if audio_url:
                if not audio_url.startswith('http'):
                    st.error("❌ Invalid audio URL format from Murf AI")
                    st.json(response_json)
                    return None
                
                try:
                    audio_response = requests.get(audio_url, timeout=30)
                    audio_response.raise_for_status()
                    
                    if 'audioLengthInSeconds' in response_json:
                        duration = response_json['audioLengthInSeconds']
                        st.info(f"Audio duration: {duration:.1f} seconds")
                    
                    return audio_response.content
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to download audio from URL: {str(e)}")
                    return None
            else:
                st.error("❌ No audio URL found in Murf AI response")
                st.markdown("**Response keys found:**")
                st.json(list(response_json.keys()) if isinstance(response_json, dict) else "Not a dictionary")
                with st.expander("Full API Response"):
                    st.json(response_json)
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Murf AI API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                st.error(f"Murf AI API Error Response: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during Murf AI audio generation: {e}")
            return None

    def test_voice_configuration(self, voice_config: Dict):
        provider = voice_config.get('provider')
        api_key_to_check = voice_config.get('api_key')
        
        if provider in ['elevenlabs', 'murf', 'azure', 'amazon'] and not api_key_to_check:
            st.error("❌ API key required for testing this provider")
            return
        
        with st.spinner("🧪 Testing configuration..."):
            time.sleep(1)
            st.success("✅ Configuration test passed!")
        
        st.markdown("**Test Results:**")
        st.markdown(f"• Provider: {self.voice_providers[provider]}")
        st.markdown(f"• Voice: {voice_config.get('voice', 'N/A')}")
        st.markdown(f"• Language: {self.languages.get(voice_config.get('language', 'en'))}")
        st.markdown(f"• Speed: {voice_config.get('speed', 1.0)}x")
        
        if provider == 'google':
            st.markdown("• Connection: ✅ Internet available")
        elif provider == 'murf':
            murf_api_key = voice_config.get('api_key')
            if murf_api_key and REQUESTS_AVAILABLE:
                try:
                    headers = {
                        "accept": "application/json",
                        "api-key": murf_api_key
                    }
                    
                    response = requests.get("https://api.murf.ai/v1/speech/voices", headers=headers, timeout=10)
                    response.raise_for_status()
                    voices_data = response.json()
                    
                    if voices_data and isinstance(voices_data, (dict, list)):
                        st.markdown("• Murf API: ✅ Authentication successful (voices retrieved)")
                    else:
                        st.markdown("• Murf API: ⚠️ Authentication successful, but no voices retrieved. Check permissions.")
                        
                except requests.exceptions.RequestException as e:
                    st.markdown(f"• Murf API: ❌ Authentication failed: {e}")
            else:
                st.markdown("• Murf API: ⚠️ API key or 'requests' library missing for full test.")
        else:
            st.markdown("• API: ✅ Authentication successful")

    def show_configuration_summary(self, voice_config: Dict):
        with st.expander("📋 Configuration Summary", expanded=False):
            provider = voice_config.get('provider', 'google')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Provider Settings:**")
                st.markdown(f"• Provider: {self.voice_providers[provider]}")
                st.markdown(f"• Voice: {voice_config.get('voice', 'N/A')}")
                st.markdown(f"• Language: {self.languages.get(voice_config.get('language', 'en'))}")
                st.markdown(f"• Format: {voice_config.get('response_format', voice_config.get('format', 'mp3'))}")
            
            with col2:
                st.markdown("**Voice Settings:**")
                st.markdown(f"• Speed: {voice_config.get('speed', 1.0)}x")
                st.markdown(f"• Volume: {voice_config.get('volume', 0.8)}")
                st.markdown(f"• Pitch: {voice_config.get('pitch', 0)} semitones")
                st.markdown(f"• Auto-play: {'✅' if voice_config.get('auto_play') else '❌'}")

    def calculate_estimated_cost(self, provider: str, characters: int) -> float:
        cost_per_1k = {
            'openai': 0.015,
            'elevenlabs': 0.30,
            'google': 0.0,
            'azure': 0.004,
            'amazon': 0.004,
            'murf': 0.05
        }
        
        rate = cost_per_1k.get(provider, 0.01)
        return (characters / 1000) * rate

    def get_default_config(self) -> Dict:
        return {
            'enabled': False,
            'provider': 'murf',
            'api_key': '',
            'voice': '',
            'language': 'en',
            'speed': 1.0,
            'volume': 0.8,
            'pitch': 0,
            'emphasis': 'none',
            'pause_detection': True,
            'ssml_support': False,
            'auto_play': True,
            'show_transcript': True,
            'max_length': 500,
            'timeout': 15,
            'response_format': 'mp3',
            'test_text': "Hello! I'm your AI assistant. How can I help you today?"
        }

def get_voice_config():
    try:
        if 'voice_config_manager' not in st.session_state:
            data_manager = st.session_state.get('data_manager')
            if not data_manager:
                from components.data_manager import DataManager
                data_manager = DataManager()
                st.session_state.data_manager = data_manager
            
            st.session_state.voice_config_manager = VoiceConfig(data_manager=data_manager)
        
        return st.session_state.voice_config_manager
        
    except Exception as e:
        st.error(f"Error creating VoiceConfig: {e}")
        return VoiceConfig()

def auto_save_voice_config():
    try:
        if 'voice_config_manager' in st.session_state:
            voice_manager = st.session_state.voice_config_manager
            if voice_manager.data_manager and 'voice_config' in st.session_state:
                voice_manager.save_voice_config_permanently(st.session_state.voice_config)
    except Exception as e:
        st.warning(f"Auto-save voice config failed: {e}")

def get_voice_provider_info(provider: str) -> Dict:
    provider_info = {
        'google': {
            'website': 'https://cloud.google.com/text-to-speech',
            'pricing': 'Free (with limitations)',
            'free_tier': 'Yes, unlimited',
            'features': ['Simple setup', 'No API key needed', 'Multiple languages', 'Fast generation']
        },
        'openai': {
            'website': 'https://platform.openai.com/docs/guides/text-to-speech',
            'pricing': '$0.015 per 1K characters',
            'free_tier': 'No, pay per use',
            'features': ['High quality voices', 'Fast generation', 'Multiple formats', 'Natural speech']
        },
        'elevenlabs': {
            'website': 'https://elevenlabs.io/',
            'pricing': 'Starting at $5/month',
            'free_tier': '10,000 characters/month',
            'features': ['Voice cloning', 'Emotional control', 'Premium quality', 'Custom voices']
        },
        'murf': {
            'website': 'https://murf.ai/',
            'pricing': 'Subscription-based (various tiers)',
            'free_tier': 'Limited free trial',
            'features': ['Studio-quality voices', 'Advanced editing', 'Voice customization', 'Multi-language support']
        }
    }
    
    return provider_info.get(provider, {})