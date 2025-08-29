import streamlit as st
import base64
import os
import sys
import time
import json
import io
import threading
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Check for required dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pydub
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

class VoiceConfig:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager or self._get_data_manager()
        self.user_id = st.session_state.get('user_id')
        self._initialize_storage_paths()
        
        # Murf WebSocket constants (based on working sample)
        self.MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"
        self.MURF_SAMPLE_RATE = 44100
        self.MURF_CHANNELS = 1
        self.MURF_FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

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

    async def _generate_murf_tts_audio(self, voice_config: Dict, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        IMPROVED: Murf AI TTS generation using WebSocket streaming.
        Based on the working sample code provided.
        """
        if not WEBSOCKETS_AVAILABLE or not PYAUDIO_AVAILABLE:
            st.error("Murf AI requires 'websockets' and 'pyaudio' libraries.")
            st.code("pip install websockets pyaudio")
            return None, None

        murf_api_key = voice_config.get('api_key')
        murf_voice_id = voice_config.get('voice')

        if not murf_api_key:
            st.error("Murf AI API key is missing.")
            return None, None

        if not murf_voice_id:
            st.error("Murf AI voice ID is not selected.")
            return None, None

        # Construct WebSocket URL with parameters (exactly as in working sample)
        ws_url = (
            f"{self.MURF_WS_URL}?"
            f"api-key={murf_api_key}&"
            f"sample_rate={self.MURF_SAMPLE_RATE}&"
            f"channel_type=MONO&"
            f"format=WAV"
        )

        full_audio_data = io.BytesIO()
        first_chunk = True

        try:
            st.info("🔗 Connecting to Murf AI WebSocket...")
            
            async with websockets.connect(ws_url, timeout=20) as ws:
                st.info("✅ Connected! Sending voice configuration...")
                
                # Step 1: Send voice config first (as in working sample)
                voice_config_msg = {
                    "voice_config": {
                        "voiceId": murf_voice_id,
                        "style": "Conversational",
                        "rate": int((voice_config.get('speed', 1.0) - 1.0) * 100),  # Convert to Murf format
                        "pitch": voice_config.get('pitch', 0),
                        "variation": 1
                    }
                }
                
                await ws.send(json.dumps(voice_config_msg))
                st.info("📢 Voice config sent. Sending text...")

                # Step 2: Send text message with end flag (as in working sample)
                text_msg = {
                    "text": text,
                    "end": True  # This closes the context for concurrency
                }
                
                await ws.send(json.dumps(text_msg))
                st.info("📝 Text sent. Receiving audio data...")

                # Step 3: Receive and process audio chunks
                while True:
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        data = json.loads(response)
                        
                        if "audio" in data and data["audio"]:
                            audio_bytes = base64.b64decode(data["audio"])
                            
                            # CRITICAL: Skip WAV header for first chunk only (as in working sample)
                            if first_chunk and len(audio_bytes) > 44:
                                audio_bytes = audio_bytes[44:]  # Skip WAV header
                                first_chunk = False
                            
                            full_audio_data.write(audio_bytes)
                            st.info(f"🎵 Received {len(audio_bytes)} audio bytes. Total: {full_audio_data.tell()}")

                        # Check for completion
                        if data.get("final", False):
                            st.info("🏁 Final audio chunk received.")
                            break
                            
                        if "error" in data:
                            st.error(f"❌ Murf API error: {data['error']}")
                            return None, None

                    except asyncio.TimeoutError:
                        st.warning("⏰ Timeout waiting for audio data. Connection may have ended.")
                        break
                    except json.JSONDecodeError as e:
                        st.error(f"📋 Invalid JSON response: {e}")
                        break

        except websockets.exceptions.InvalidStatusCode as e:
            if e.status_code == 401:
                st.error("🔑 Invalid Murf AI API key. Please verify your credentials.")
            elif e.status_code == 403:
                st.error("🚫 API key does not have permission. Check your Murf subscription.")
            else:
                st.error(f"🌐 WebSocket connection failed: HTTP {e.status_code}")
            return None, None
            
        except websockets.exceptions.ConnectionClosedError as e:
            st.error(f"🔌 Connection closed unexpectedly: {e.code} - {e.reason}")
            return None, None
            
        except Exception as e:
            st.error(f"💥 Unexpected error during Murf AI TTS: {str(e)}")
            return None, None

        # Get final audio data
        audio_content = full_audio_data.getvalue()
        
        if len(audio_content) == 0:
            st.error("📭 No audio data received from Murf AI.")
            return None, None

        st.success(f"🎉 Successfully generated {len(audio_content)} bytes of Murf AI audio!")
        return audio_content, "audio/wav"

    def _run_async_murf_generation(self, voice_config: Dict, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        IMPROVED: Synchronous wrapper for async Murf generation.
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                result = loop.run_until_complete(
                    self._generate_murf_tts_audio(voice_config, text)
                )
                return result
            finally:
                loop.close()
                
        except Exception as e:
            st.error(f"⚡ Error in async Murf generation: {str(e)}")
            return None, None

    def generate_voice_preview_sync(self, voice_config: Dict, text: str) -> Tuple[Optional[bytes], Optional[str]]:
        """
        IMPROVED: Synchronous voice preview generation with better error handling.
        """
        provider = voice_config.get('provider', 'google')
        
        try:
            with st.spinner(f"🎤 Generating voice preview with {provider}..."):
                
                if provider == 'murf':
                    if not (WEBSOCKETS_AVAILABLE and PYAUDIO_AVAILABLE):
                        st.error("❌ Murf AI requires 'websockets' and 'pyaudio' libraries.")
                        st.code("pip install websockets pyaudio")
                        return None, None
                    
                    # Use thread pool to run async function
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            self._run_async_murf_generation, 
                            voice_config, 
                            text
                        )
                        
                        try:
                            audio_data, audio_format = future.result(timeout=45)  # Increased timeout
                            
                            if audio_data:
                                st.success("✅ Murf AI preview generated successfully!")
                                # Display audio player
                                st.audio(audio_data, format=audio_format)
                                return audio_data, audio_format
                            else:
                                st.error("❌ Failed to generate Murf AI audio.")
                                return None, None
                                
                        except TimeoutError:
                            st.error("⏰ Murf AI generation timed out. Please try again.")
                            return None, None
                            
                elif provider == 'google' and GTTS_AVAILABLE:
                    audio_data = self._generate_google_tts_audio(voice_config, text)
                    if audio_data:
                        st.success("✅ Google TTS preview generated!")
                        st.audio(audio_data, format="audio/mp3")
                        return audio_data, "audio/mp3"
                    else:
                        st.error("❌ Failed to generate Google TTS audio.")
                        return None, None
                        
                elif provider == 'openai':
                    audio_data = self._generate_openai_tts_audio(voice_config, text)
                    if audio_data:
                        st.success("✅ OpenAI TTS preview generated!")
                        st.audio(audio_data, format="audio/mp3")
                        return audio_data, "audio/mp3"
                    else:
                        st.error("❌ Failed to generate OpenAI TTS audio.")
                        return None, None
                        
                else:
                    st.info(f"ℹ️ Voice generation for {provider} is not implemented yet.")
                    return None, None
                    
        except Exception as e:
            st.error(f"💥 Error generating preview: {str(e)}")
            return None, None

    def generate_tts_audio(self, text: str, voice_config: Optional[Dict] = None) -> Tuple[Optional[bytes], Optional[str]]:
        """
        IMPROVED: Main TTS generation method with better error handling.
        """
        if not voice_config:
            voice_config = st.session_state.get('voice_config', self.get_default_config())

        if not voice_config.get('enabled', False):
            return None, None

        provider = voice_config.get('provider', 'google')
        
        # Truncate long text to prevent timeouts
        max_length = voice_config.get('max_length', 1000)
        if len(text) > max_length:
            text = text[:max_length] + "..."
            st.warning(f"⚠️ Text truncated to {max_length} characters for TTS generation.")

        try:
            if provider == 'google' and GTTS_AVAILABLE:
                audio_data = self._generate_google_tts_audio(voice_config, text)
                return audio_data, "audio/mp3"
                
            elif provider == 'murf' and WEBSOCKETS_AVAILABLE and PYAUDIO_AVAILABLE:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._run_async_murf_generation, 
                        voice_config, 
                        text
                    )
                    
                    try:
                        result = future.result(timeout=45)
                        return result
                    except TimeoutError:
                        st.error("⏰ TTS generation timed out")
                        return None, None
                        
            elif provider == 'openai' and REQUESTS_AVAILABLE:
                audio_data = self._generate_openai_tts_audio(voice_config, text)
                return audio_data, "audio/mp3"
                
            else:
                st.warning(f"⚠️ TTS provider '{provider}' not available or not implemented")
                return None, None
                
        except Exception as e:
            st.error(f"💥 Error generating TTS audio: {str(e)}")
            return None, None

    # === EXISTING METHODS PRESERVED ===

    def _fetch_and_cache_murf_voices(self, murf_api_key: str) -> Dict[str, str]:
        if not murf_api_key:
            return {"Error": "API Key is missing."}

        if not REQUESTS_AVAILABLE:
            return {"Error": "requests library not available. Install with: pip install requests"}

        cache_key = f"murf_voices_cache_{murf_api_key[:8]}"
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

    def _generate_google_tts_audio(self, voice_config: Dict, text: str) -> Optional[bytes]:
        """Generate TTS audio using Google Text-to-Speech."""
        if not GTTS_AVAILABLE:
            return None

        try:
            language = voice_config.get('language', 'en')
            if language not in self.languages:
                language = 'en'

            tts = gTTS(text=text, lang=language, slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            return audio_buffer.getvalue()

        except Exception as e:
            st.error(f"Google TTS error: {str(e)}")
            return None

    def _generate_openai_tts_audio(self, voice_config: Dict, text: str) -> Optional[bytes]:
        """Generate TTS audio using OpenAI API."""
        if not REQUESTS_AVAILABLE:
            st.error("requests library not available for OpenAI TTS")
            return None

        api_key = voice_config.get('api_key')
        if not api_key:
            st.error("OpenAI API key is missing.")
            return None

        voice = voice_config.get('voice', 'alloy')
        speed = voice_config.get('speed', 1.0)

        try:
            url = "https://api.openai.com/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "tts-1",
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": "mp3"
            }

            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.content

        except requests.exceptions.RequestException as e:
            st.error(f"OpenAI TTS API error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"OpenAI TTS error: {str(e)}")
            return None

    def get_default_config(self) -> Dict:
        """Get default voice configuration."""
        return {
            'enabled': False,
            'provider': 'google',
            'api_key': '',
            'voice': 'female',
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

    def configure_voice(self) -> Dict:
        """Main voice configuration interface."""
        if 'voice_config' not in st.session_state:
            st.session_state.voice_config = self.get_default_config()

        voice_config = st.session_state.voice_config.copy()

        st.markdown("### 🎙️ Voice Response Configuration")
        st.markdown("Configure text-to-speech settings for your chatbot responses")

        voice_enabled = st.checkbox(
            "Enable Voice Responses",
            value=voice_config.get('enabled', False),
            help="Generate audio responses for bot messages",
            key="voice_enabled_checkbox"
        )

        voice_config['enabled'] = voice_enabled

        if not voice_enabled:
            st.info("💡 Voice responses are disabled. Users will see text-only responses.")
            return voice_config

        # Provider selection
        available_providers = self.get_available_providers()
        if not available_providers:
            st.error("❌ No TTS providers available. Please install required dependencies.")
            return voice_config

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

        # Configure based on provider
        if provider == 'murf':
            self.configure_murf_tts(voice_config)
        elif provider == 'google':
            self.configure_google_tts(voice_config)
        elif provider == 'openai':
            self.configure_openai_tts(voice_config)

        # Voice settings
        self.configure_voice_settings(voice_config)

        # Test section
        if st.button("🎵 Test Voice Configuration", key="test_voice_btn"):
            test_text = voice_config.get('test_text', "Hello! This is a test of your voice configuration.")
            self.generate_voice_preview_sync(voice_config, test_text)

        st.session_state.voice_config = voice_config
        return voice_config

    def get_available_providers(self) -> List[str]:
        available = []
        if WEBSOCKETS_AVAILABLE and PYAUDIO_AVAILABLE:
            available.append('murf')
        if REQUESTS_AVAILABLE:
            available.extend(['openai', 'elevenlabs', 'azure', 'amazon'])
        if GTTS_AVAILABLE:
            available.append('google')
        return available if available else ['google']

    def get_default_voice_for_provider(self, provider: str) -> str:
        defaults = {
            'google': 'female',
            'openai': 'alloy',
            'murf': '',
            'elevenlabs': 'rachel',
            'azure': 'aria',
            'amazon': 'joanna'
        }
        return defaults.get(provider, '')

    def configure_murf_tts(self, voice_config: Dict):
        st.markdown("##### 🎤 Murf AI Text-to-Speech")
        
        if not WEBSOCKETS_AVAILABLE or not PYAUDIO_AVAILABLE:
            st.error("❌ Murf AI requires 'websockets' and 'pyaudio' libraries.")
            st.code("pip install websockets pyaudio")
            return

        st.info("🌟 Murf AI provides high-quality, realistic voices using WebSocket streaming.")

        murf_api_key_input = st.text_input(
            "Murf AI API Key",
            value=voice_config.get('api_key', ''),
            type="password",
            key="murf_api_key_config",
            help="Enter your Murf AI API Key. Get one from murf.ai"
        )

        if murf_api_key_input != voice_config.get('api_key'):
            voice_config['api_key'] = murf_api_key_input
            # Clear cached voices when API key changes
            cache_keys_to_remove = [k for k in st.session_state.keys() if k.startswith('murf_voices_cache_')]
            for key in cache_keys_to_remove:
                del st.session_state[key]

        if not murf_api_key_input:
            st.warning("⚠️ Please enter your Murf AI API key to see available voices.")
            return

        with st.spinner("🔄 Loading Murf AI voices..."):
            murf_voices_map = self._fetch_and_cache_murf_voices(murf_api_key_input)

        if "Error" in murf_voices_map:
            st.error(f"❌ Failed to load voices: {murf_voices_map['Error']}")
            return

        voice_options_display = list(murf_voices_map.keys())
        id_to_name_map = {v: k for k, v in murf_voices_map.items()}

        current_voice_id = voice_config.get('voice')
        current_selection_name = id_to_name_map.get(current_voice_id)

        try:
            current_index = voice_options_display.index(current_selection_name) if current_selection_name in voice_options_display else 0
        except (ValueError, TypeError):
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

    def configure_google_tts(self, voice_config: Dict):
        st.markdown("##### 🌐 Google Text-to-Speech (gTTS)")
        
        if not GTTS_AVAILABLE:
            st.error("❌ gTTS library not installed. Run: `pip install gtts`")
            return

        st.info("✅ Google TTS is free but requires internet connection")

        voice = st.selectbox(
            "Voice Gender",
            ['female', 'male'],
            index=0 if voice_config.get('voice', 'female') == 'female' else 1,
            help="Choose voice gender",
            key="google_voice_select"
        )

        if voice != voice_config.get('voice'):
            voice_config['voice'] = voice

    def configure_openai_tts(self, voice_config: Dict):
        st.markdown("##### 🤖 OpenAI Text-to-Speech")
        st.info("🌟 OpenAI TTS provides high-quality voices. API key required.")

        api_key = st.text_input(
            "OpenAI API Key",
            value=voice_config.get('api_key', ''),
            type="password",
            key="openai_api_key_config",
            help="Enter your OpenAI API Key"
        )

        if api_key != voice_config.get('api_key'):
            voice_config['api_key'] = api_key

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

    def configure_voice_settings(self, voice_config: Dict):
        st.markdown("#### 🎛️ Voice Settings")

        col1, col2, col3 = st.columns(3)

        with col1:
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
            voice_config['language'] = language

        with col2:
            speed = st.slider(
                "Speaking Speed",
                0.25, 2.0,
                float(voice_config.get('speed', 1.0)),
                0.05,
                help="How fast the voice should speak",
                key="voice_speed_slider"
            )
            voice_config['speed'] = speed

        with col3:
            volume = st.slider(
                "Volume",
                0.1, 1.0,
                float(voice_config.get('volume', 0.8)),
                0.1,
                help="Audio volume level",
                key="voice_volume_slider"
            )
            voice_config['volume'] = volume

        # Test text area
        test_text = st.text_area(
            "Test Text",
            value=voice_config.get('test_text', "Hello! I'm your AI assistant. How can I help you today?"),
            help="Enter text to test the voice configuration",
            key="voice_test_text",
            height=100
        )
        voice_config['test_text'] = test_text

    # Additional helper methods...
    def save_voice_config_permanently(self, voice_config: Dict) -> bool:
        """Save voice configuration permanently."""
        try:
            st.session_state.voice_config = voice_config.copy()
            return True
        except Exception as e:
            st.error(f"Error saving voice config: {str(e)}")
            return False

    def load_voice_config_permanently(self) -> Dict:
        """Load voice configuration permanently."""
        return st.session_state.get('voice_config', self.get_default_config())