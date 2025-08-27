import streamlit as st
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from components.data_manager import DataManager

class Authentication:
    def __init__(self):
        self.users_file = "data/users/users.json"
        self.data_manager = DataManager()
        self.ensure_users_directory()
    
    def ensure_users_directory(self):
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def load_users(self) -> Dict:
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(self, users: Dict):
        try:
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
        except Exception as e:
            st.error(f"Error saving users: {str(e)}")
    
    def register_user(self, username: str, email: str, password: str) -> bool:
        users = self.load_users()
        
        if username in users or any(u.get('email') == email for u in users.values()):
            return False
        
        users[username] = {
            'email': email,
            'password_hash': self.hash_password(password),
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'is_active': True
        }
        
        self.save_users(users)
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        users = self.load_users()
        
        if username not in users:
            return False
        
        user = users[username]
        
        if not user.get('is_active', True):
            return False
        
        if user['password_hash'] == self.hash_password(password):
            users[username]['last_login'] = datetime.now().isoformat()
            self.save_users(users)
            
            self.data_manager.initialize_user_session(username, user.get('email'))
            return True
        
        return False
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        users = self.load_users()
        return users.get(username)

def show_authentication():
    auth = Authentication()
    
    if st.session_state.get('authenticated') and not st.session_state.get('user_data_loaded'):
        username = st.session_state.get('username')
        if username:
            auth.data_manager.initialize_user_session(username)
            st.session_state.user_data_loaded = True
            st.rerun()
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    with tab1:
        show_login_form(auth)
    
    with tab2:
        show_signup_form(auth)

def show_login_form(auth: Authentication):
    st.subheader("Welcome Back!")
    st.markdown("Sign in to access your chatbots")
    
    with st.form("login_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            help="Your unique username"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            help="Your account password"
        )
        
        remember_me = st.checkbox("Remember me", value=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_button = st.form_submit_button("🔑 Login", type="primary", use_container_width=True)
        
        with col2:
            demo_button = st.form_submit_button("🎮 Demo Mode", use_container_width=True)
        
        if login_button:
            if username and password:
                if auth.authenticate_user(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_id = username
                    st.session_state.user_data_loaded = True
                    st.session_state.member_since = auth.get_user_info(username).get('created_at', '')[:10]
                    
                    if remember_me:
                        st.session_state.remember_login = True
                    
                    st.success("🎉 Login successful!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.error("❌ Please enter both username and password")
        
        if demo_button:
            demo_username = f"demo_user_{datetime.now().strftime('%H%M%S')}"
            st.session_state.authenticated = True
            st.session_state.username = demo_username
            st.session_state.user_id = demo_username
            st.session_state.demo_mode = True
            st.session_state.member_since = datetime.now().strftime('%Y-%m-%d')
            st.session_state.user_data_loaded = True
            
            st.success("🎮 Welcome to Demo Mode!")
            st.info("Demo mode gives you full access. Your data won't be saved permanently.")
            st.rerun()

def show_signup_form(auth: Authentication):
    st.subheader("Join Us!")
    st.markdown("Create your account to start building chatbots")
    
    with st.form("signup_form"):
        username = st.text_input(
            "Choose Username",
            placeholder="Enter a unique username",
            help="This will be your unique identifier"
        )
        
        email = st.text_input(
            "Email Address",
            placeholder="Enter your email",
            help="We'll use this to send you important updates"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Choose a strong password",
            help="At least 6 characters recommended"
        )
        
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            help="Must match your password"
        )
        
        agree_terms = st.checkbox(
            "I agree to the Terms of Service and Privacy Policy",
            help="Required to create an account"
        )
        
        signup_button = st.form_submit_button("📝 Create Account", type="primary", use_container_width=True)
        
        if signup_button:
            errors = []
            
            if not username or len(username) < 3:
                errors.append("Username must be at least 3 characters")
            
            if not email or '@' not in email:
                errors.append("Please enter a valid email address")
            
            if not password or len(password) < 6:
                errors.append("Password must be at least 6 characters")
            
            if password != confirm_password:
                errors.append("Passwords don't match")
            
            if not agree_terms:
                errors.append("You must agree to the terms and conditions")
            
            if errors:
                for error in errors:
                    st.error(f"❌ {error}")
            else:
                if auth.register_user(username, email, password):
                    st.success("🎉 Account created successfully!")
                    st.info("You can now login with your credentials")
                    
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_id = username
                    st.session_state.member_since = datetime.now().strftime('%Y-%m-%d')
                    st.session_state.user_data_loaded = True
                    
                    auth.data_manager.initialize_user_session(username, email)
                    
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Username or email already exists")

def logout_user():
    if 'data_manager' in st.session_state:
        st.session_state.data_manager.save_all_user_data()
    
    keys_to_clear = [
        'authenticated', 'username', 'user_id', 'demo_mode', 'member_since',
        'current_bot', 'bots', 'chat_history', 'available_models', 'user_config',
        'user_data_loaded', 'data_manager'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("👋 Logged out successfully!")
    st.rerun()

def check_authentication():
    return st.session_state.get('authenticated', False)

def require_authentication():
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not check_authentication():
                st.error("🔒 Authentication required")
                st.info("Please login to access this page")
                if st.button("Go to Login"):
                    st.switch_page("app.py")
                return
            
            if not st.session_state.get('user_data_loaded'):
                auth = Authentication()
                username = st.session_state.get('username')
                if username:
                    auth.data_manager.initialize_user_session(username)
                    st.session_state.user_data_loaded = True
                    st.rerun()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
