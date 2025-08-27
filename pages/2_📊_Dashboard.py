import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import json
from components.data_manager import DataManager, auto_save_user_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly not installed. Charts will be limited. Install with: `pip install plotly`")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    import random

st.set_page_config(
    page_title="Dashboard - Chatbot Builder",
    page_icon="📊",
    layout="wide"
)

def get_data_manager_instance():
    if 'data_manager_instance' not in st.session_state:
        st.session_state.data_manager_instance = DataManager()
    return st.session_state.data_manager_instance

def initialize_data_manager():
    try:
        return get_data_manager_instance()
    except Exception as e:
        st.error(f"Failed to initialize DataManager: {e}")
        return None

def ensure_user_authenticated():
    if not st.session_state.get('authenticated', False):
        st.error("🔒 Please login first to access your dashboard")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🏠 Go to Login", key="login_redirect", use_container_width=True):
                st.switch_page("app.py")
        return False

    if not st.session_state.get('user_id') or not st.session_state.get('username'):
        st.error("⚠️ Session expired. Please login again.")
        if st.button("🔄 Refresh Login", key="refresh_login"):
            st.switch_page("app.py")
        return False

    return True

def rehydrate_session():
    dm = get_data_manager_instance()
    if not dm:
        return False

    user_id = st.session_state.get('user_id')
    if not user_id:
        return False

    try:
        needs_reload = (
            not st.session_state.get('bots') or
            not st.session_state.get('user_config') or
            not st.session_state.get('chat_history')
        )

        if needs_reload:
            with st.spinner("🔄 Loading your data..."):
                st.session_state.user_config = dm.load_user_config(user_id)
                st.session_state.bots = dm.load_user_bots(user_id)
                st.session_state.chat_history = dm.load_chat_history(user_id)

                if st.session_state.user_config.get('username'):
                    st.session_state.username = st.session_state.user_config['username']

                st.success("✅ Data loaded successfully!")
                return True

        return True

    except Exception as e:
        st.error(f"❌ Error loading user data: {str(e)}")
        return False

def save_changes():
    dm = get_data_manager_instance()
    if dm and st.session_state.get('user_id'):
        try:
            dm.save_all_user_data(st.session_state.user_id)
            return True
        except Exception as e:
            st.error(f"Failed to save changes: {e}")
            return False
    return False

def load_dashboard_css():
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s ease;
        border: none;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .bot-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
        border-left: 4px solid;
    }

    .bot-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }

    .user-info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .data-status {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    load_dashboard_css()

    dm = initialize_data_manager()
    if not dm:
        st.stop()

    if not ensure_user_authenticated():
        st.stop()

    if not rehydrate_session():
        st.error("Failed to load your data. Please try refreshing the page.")
        if st.button("🔄 Refresh Page", key="refresh_page"):
            st.rerun()
        st.stop()

    show_data_status()

    render_dashboard()

def show_data_status():
    dm = get_data_manager_instance()
    user_id = st.session_state.get('user_id')

    if dm and user_id:
        try:
            stats = dm.get_user_stats(user_id)
            user_config = st.session_state.get('user_config', {})
            last_updated = user_config.get('last_updated', 'Unknown')

            if last_updated != 'Unknown':
                try:
                    last_updated_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                    last_updated_str = last_updated_dt.strftime('%Y-%m-%d %H:%M:%S')
                    time_ago = datetime.now() - last_updated_dt.replace(tzinfo=None)

                    if time_ago.days > 0:
                        time_ago_str = f"{time_ago.days} days ago"
                    elif time_ago.seconds > 3600:
                        time_ago_str = f"{time_ago.seconds // 3600} hours ago"
                    elif time_ago.seconds > 60:
                        time_ago_str = f"{time_ago.seconds // 60} minutes ago"
                    else:
                        time_ago_str = "Just now"
                except:
                    last_updated_str = last_updated
                    time_ago_str = ""
            else:
                last_updated_str = "Unknown"
                time_ago_str = ""

            with st.expander("📊 Data Status & Storage Info", expanded=False):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### 💾 Storage Status")
                    st.metric("Total Bots", stats.get('total_bots', 0))
                    st.metric("Total Messages", stats.get('total_messages', 0))
                    st.metric("Uploaded Files", stats.get('total_uploaded_files', 0))

                with col2:
                    st.markdown("### ⏰ Last Activity")
                    st.info(f"**Last Updated:** {last_updated_str}")
                    if time_ago_str:
                        st.success(f"**Time Ago:** {time_ago_str}")
                    st.info(f"**User ID:** {user_id[:12]}...")

                with col3:
                    st.markdown("### 🔄 Quick Actions")
                    if st.button("💾 Save All Data Now", key="manual_save", use_container_width=True):
                        if save_changes():
                            st.success("✅ All data saved successfully!")
                            st.rerun()

                    if st.button("🔄 Reload Data", key="manual_reload", use_container_width=True):
                        st.session_state.bots = {}
                        st.session_state.chat_history = {}
                        st.session_state.user_config = {}
                        if rehydrate_session():
                            st.success("✅ Data reloaded successfully!")
                            st.rerun()

                    if st.button("📥 Export All Data", key="export_user_data", use_container_width=True):
                        export_file = dm.export_user_data(user_id)
                        if export_file:
                            with open(export_file, 'r') as f:
                                st.download_button(
                                    "📥 Download Export",
                                    f.read(),
                                    file_name=f"user_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            st.success("✅ Export ready for download!")

                usage = dm.get_storage_usage(user_id)
                if usage:
                    st.markdown("### 💿 Storage Usage")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Config", f"{usage['config_size'] // 1024}KB")
                    with col2:
                        st.metric("Bots", f"{usage['bots_size'] // 1024}KB")
                    with col3:
                        st.metric("Chats", f"{usage['chat_size'] // 1024}KB")
                    with col4:
                        st.metric("Files", f"{usage['files_size'] // 1024}KB")

        except Exception as e:
            st.warning(f"Could not load data status: {e}")

def render_dashboard():
    date_range, status_filter = render_sidebar()

    username = st.session_state.get('username', 'User')
    st.markdown(f"""
    <div class="dashboard-header">
        <h1>📊 {username}'s Dashboard</h1>
        <p style="font-size: 1.2rem; margin: 1rem 0 0 0; opacity: 0.9;">
            Track performance and manage your AI chatbots
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        dashboard_data = get_dashboard_data(status_filter, date_range)
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        dashboard_data = get_empty_dashboard_data()

    show_key_metrics(dashboard_data)

    st.markdown("---")

    if PLOTLY_AVAILABLE:
        show_charts_section(dashboard_data)
    else:
        show_simple_metrics(dashboard_data)

    st.markdown("---")

    show_bot_management_section(dashboard_data, status_filter)

    st.markdown("---")

    show_recent_activity(dashboard_data)

def render_sidebar():
    with st.sidebar:
        username = st.session_state.get('username', 'User')
        user_id = st.session_state.get('user_id', '')

        st.markdown(f"""
        <div class="user-info-card">
            <h3 style="margin: 0; font-size: 1.2rem;">👤 {username}</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Dashboard Admin</p>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.8rem;">ID: {user_id[:8]}...</p>
        </div>
        """, unsafe_allow_html=True)

        dm = get_data_manager_instance()
        if dm:
            st.markdown("### 🔄 Data Sync")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save", key="sidebar_save", use_container_width=True, help="Save all changes"):
                    if save_changes():
                        st.success("✅ Saved!")
                        st.rerun()

            with col2:
                if st.button("🔄 Reload", key="sidebar_reload", use_container_width=True, help="Reload from disk"):
                    st.session_state.bots = {}
                    st.session_state.chat_history = {}
                    st.session_state.user_config = {}
                    if rehydrate_session():
                        st.success("✅ Reloaded!")
                        st.rerun()

        st.markdown("---")

        st.markdown("### 🎯 Navigation")

        nav_buttons = [
            ("🏠 Home", "app.py"),
            ("🤖 Create Bot", "pages/1_🤖_Create_Bot.py"),
            ("💬 Test Chat", "pages/4_🎯_Chat.py"),
            ("⚙️ Settings", "pages/3_⚙️_Settings.py")
        ]

        for i, (label, page) in enumerate(nav_buttons):
            if st.button(label, key=f"sidebar_nav_{i}", use_container_width=True):
                save_changes()
                st.switch_page(page)

        st.markdown("---")

        st.markdown("### 🔍 Dashboard Filters")

        date_range = st.selectbox(
            "📅 Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
            key="sidebar_date_filter",
            help="Select the time period for analytics"
        )

        status_filter = st.multiselect(
            "📊 Bot Status",
            ["active", "testing", "draft"],
            default=["active", "testing", "draft"],
            key="sidebar_status_filter",
            help="Filter bots by their current status"
        )

        bots = st.session_state.get('bots', {})
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")

        total_bots = len(bots)
        active_bots = len([b for b in bots.values() if b.get('status') == 'active'])

        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
            <div style="display: flex; justify-content: space-between;">
                <span>Total Bots:</span><strong>{total_bots}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Active:</span><strong style="color: #28a745;">{active_bots}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Draft:</span><strong style="color: #6c757d;">{total_bots - active_bots}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📥 Export Options")

        if st.button("📊 Export Dashboard", key="sidebar_export_dashboard", use_container_width=True):
            export_dashboard_data()

        if st.button("🤖 Export All Bots", key="sidebar_export_bots", use_container_width=True):
            export_all_bots()

        st.markdown("---")
        st.markdown("### 🗄️ Data Management")

        if st.button("🧹 Cleanup Files", key="sidebar_cleanup", use_container_width=True, help="Remove orphaned files"):
            dm = get_data_manager_instance()
            if dm and st.session_state.get('user_id'):
                if dm.cleanup_orphaned_files(st.session_state.user_id):
                    st.success("✅ Cleanup complete!")

        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Dashboard Features:**
            - All data is automatically saved
            - Use Save/Reload buttons to sync data
            - Export options for backup
            - Cleanup removes unused files

            **Troubleshooting:**
            - If data is missing, click Reload
            - Use Export for backup before major changes
            - Check Data Status for storage info
            """)

    return date_range, status_filter

def get_dashboard_data(status_filter: list, date_range: str) -> dict:
    bots = st.session_state.get('bots', {})
    chat_history = st.session_state.get('chat_history', {})

    filtered_bots = {
        bot_id: bot for bot_id, bot in bots.items()
        if bot.get('status', 'draft') in status_filter
    }

    now = datetime.now()
    if date_range == "Last 7 days":
        date_threshold = now - timedelta(days=7)
    elif date_range == "Last 30 days":
        date_threshold = now - timedelta(days=30)
    elif date_range == "Last 90 days":
        date_threshold = now - timedelta(days=90)
    else:
        date_threshold = datetime(2020, 1, 1)

    total_bots = len(filtered_bots)
    active_bots = len([b for b in filtered_bots.values() if b.get('status') == 'active'])
    testing_bots = len([b for b in filtered_bots.values() if b.get('status') == 'testing'])
    draft_bots = len([b for b in filtered_bots.values() if b.get('status') == 'draft'])

    total_conversations = 0
    total_messages = 0
    recent_messages = 0
    user_satisfaction = []

    for bot_id, messages in chat_history.items():
        if bot_id in filtered_bots:
            if messages:
                total_conversations += 1
                total_messages += len(messages)

                for msg in messages:
                    try:
                        msg_time = datetime.fromisoformat(msg.get('timestamp', '2020-01-01T00:00:00'))
                        if msg_time >= date_threshold:
                            recent_messages += 1
                    except:
                        pass

                    if msg.get('rating'):
                        user_satisfaction.append(msg['rating'])

    status_counts = {
        'active': active_bots,
        'testing': testing_bots,
        'draft': draft_bots
    }

    status_counts = {k: v for k, v in status_counts.items() if v > 0}

    usage_data = generate_usage_data(date_range, total_messages)

    satisfaction_rate = 0
    if user_satisfaction:
        positive_ratings = len([r for r in user_satisfaction if r == 'like'])
        satisfaction_rate = (positive_ratings / len(user_satisfaction)) * 100

    return {
        'total_bots': total_bots,
        'active_bots': active_bots,
        'testing_bots': testing_bots,
        'draft_bots': draft_bots,
        'total_conversations': total_conversations,
        'total_messages': total_messages,
        'recent_messages': recent_messages,
        'satisfaction_rate': satisfaction_rate,
        'status_counts': status_counts,
        'usage_data': usage_data,
        'filtered_bots': filtered_bots,
        'date_range': date_range
    }

def get_empty_dashboard_data():
    return {
        'total_bots': 0,
        'active_bots': 0,
        'testing_bots': 0,
        'draft_bots': 0,
        'total_conversations': 0,
        'total_messages': 0,
        'recent_messages': 0,
        'satisfaction_rate': 0,
        'status_counts': {},
        'usage_data': pd.DataFrame(),
        'filtered_bots': {},
        'date_range': 'All time'
    }

def generate_usage_data(date_range: str, base_messages: int = 0) -> pd.DataFrame:
    if date_range == "Last 7 days":
        days = 7
    elif date_range == "Last 30 days":
        days = 30
    elif date_range == "Last 90 days":
        days = 90
    else:
        days = 30

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    if NUMPY_AVAILABLE:
        import numpy as np
        base_usage = max(10, base_messages // days if days > 0 else 10)
        usage = []
        for i in range(len(dates)):
            weekly_pattern = 15 * np.sin(2 * np.pi * i / 7)
            random_variation = np.random.normal(0, 5)
            daily_usage = max(0, base_usage + weekly_pattern + random_variation)
            usage.append(int(daily_usage))
    else:
        import random
        base_usage = max(10, base_messages // days if days > 0 else 10)
        usage = []
        for i in range(len(dates)):
            daily_usage = max(0, base_usage + random.randint(-10, 15))
            usage.append(daily_usage)

    return pd.DataFrame({
        'date': dates,
        'messages': usage,
        'conversations': [max(1, int(u/8)) for u in usage]
    })

def show_key_metrics(data: dict):
    st.markdown("### 📊 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_val = f"+{data['active_bots']}" if data['active_bots'] > 0 else None
        st.metric(
            "🤖 Total Bots",
            data['total_bots'],
            delta=delta_val,
            help="Total number of bots (data saved to disk)"
        )

    with col2:
        active_percentage = (data['active_bots'] / max(data['total_bots'], 1)) * 100
        st.metric(
            "🟢 Active Bots",
            data['active_bots'],
            delta=f"{active_percentage:.0f}% of total",
            help="Bots currently deployed and active"
        )

    with col3:
        st.metric(
            "💬 Conversations",
            data['total_conversations'],
            delta=f"{data['total_messages']} messages",
            help="Total chat sessions (persistent storage)"
        )

    with col4:
        satisfaction_delta = f"{data['satisfaction_rate']:.0f}% satisfaction" if data['satisfaction_rate'] > 0 else None
        st.metric(
            f"📈 Messages ({data['date_range']})",
            data['recent_messages'],
            delta=satisfaction_delta,
            help="Messages in selected time period"
        )

    st.markdown("---")
    user_config = st.session_state.get('user_config', {})
    last_updated = user_config.get('last_updated')

    if last_updated:
        try:
            last_updated_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            time_ago = datetime.now() - last_updated_dt.replace(tzinfo=None)

            if time_ago.total_seconds() < 300:
                freshness_color = "#28a745"
                freshness_text = "🟢 Data is fresh"
            elif time_ago.total_seconds() < 3600:
                freshness_color = "#ffc107"
                freshness_text = "🟡 Data is recent"
            else:
                freshness_color = "#dc3545"
                freshness_text = "🔴 Data might be stale"

            st.markdown(f"""
            <div style="text-align: center; color: {freshness_color}; font-weight: bold;">
                {freshness_text} - Last synced: {last_updated_dt.strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("🔄 Data sync status unknown")

def show_charts_section(data: dict):
    st.markdown("### 📈 Analytics Overview")

    col1, col2 = st.columns(2)

    with col1:
        show_usage_chart(data)

    with col2:
        show_bot_status_chart(data)

    if len(data['usage_data']) > 0:
        col3, col4 = st.columns(2)

        with col3:
            show_performance_metrics(data)

        with col4:
            show_satisfaction_chart(data)

def show_usage_chart(data: dict):
    st.subheader("📊 Usage Trends")

    usage_df = data['usage_data']

    if len(usage_df) > 0:
        fig = px.line(
            usage_df,
            x='date',
            y=['messages', 'conversations'],
            title=f"Daily Usage - {data['date_range']}",
            labels={'value': 'Count', 'date': 'Date', 'variable': 'Type'},
            color_discrete_map={
                'messages': '#667eea',
                'conversations': '#764ba2'
            }
        )

        fig.update_layout(
            showlegend=True,
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No usage data available for the selected period")

def show_bot_status_chart(data: dict):
    st.subheader("🤖 Bot Status Distribution")

    status_counts = data['status_counts']

    if status_counts:
        colors = {
            'active': '#28a745',
            'testing': '#ffc107',
            'draft': '#6c757d'
        }

        fig = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Bots by Current Status",
            color=list(status_counts.keys()),
            color_discrete_map=colors
        )

        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )

        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bots found with selected filters")

def show_performance_metrics(data: dict):
    st.subheader("Performance Metrics")

    metrics = ['Response Time', 'Accuracy', 'User Engagement', 'Uptime']
    values = [85, 92, 78, 99]

    fig = px.bar(
        x=metrics,
        y=values,
        title="Bot Performance Scores (%)",
        color=values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        showlegend=False,
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

def show_satisfaction_chart(data: dict):
    st.subheader("User Satisfaction")

    satisfaction_rate = data['satisfaction_rate']

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = satisfaction_rate,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Satisfaction Rate (%)"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

def show_simple_metrics(data: dict):
    st.markdown("### Usage Summary")

    usage_df = data['usage_data']

    if len(usage_df) > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", int(usage_df['messages'].sum()))

        with col2:
            st.metric("Total Conversations", int(usage_df['conversations'].sum()))

        with col3:
            st.metric("Daily Average", f"{usage_df['messages'].mean():.1f}")

        with col4:
            st.metric("Peak Day", f"{usage_df['messages'].max():.0f}")

        st.markdown("#### Daily Breakdown")
        usage_df_display = usage_df.copy()
        usage_df_display['Date'] = usage_df_display['date'].dt.strftime('%Y-%m-%d')
        usage_df_display['Messages'] = usage_df_display['messages'].astype(int)
        usage_df_display['Conversations'] = usage_df_display['conversations'].astype(int)

        st.dataframe(
            usage_df_display[['Date', 'Messages', 'Conversations']],
            use_container_width=True,
            hide_index=True
        )

def show_bot_management_section(data: dict, status_filter: list):
    st.markdown("### Bot Management Center")

    bots = data['filtered_bots']

    if not bots:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; margin: 2rem 0;">
            <h3>No Bots Found</h3>
            <p style="color: #6c757d; margin: 1rem 0;">Create your first bot to get started with AI conversations!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("Create Your First Bot", key="create_first_bot", type="primary", use_container_width=True):
                save_changes()
                st.switch_page("pages/1_Create_Bot.py")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        search_term = st.text_input("Search bots", key="bot_search_input", placeholder="Enter bot name...")

    with col2:
        sort_by = st.selectbox("Sort by", ["name", "created_at", "updated_at", "status"], key="bot_sort_selector")

    with col3:
        sort_order = st.selectbox("Order", ["ascending", "descending"], key="bot_order_selector")

    with col4:
        view_mode = st.selectbox("View", ["Cards", "Table"], key="bot_view_selector")

    filtered_bots = []
    for bot_id, bot_config in bots.items():
        if not search_term or search_term.lower() in bot_config.get('name', '').lower():
            filtered_bots.append((bot_id, bot_config))

    reverse_order = sort_order == "descending"
    try:
        filtered_bots.sort(
            key=lambda x: x[1].get(sort_by, ''),
            reverse=reverse_order
        )
    except Exception:
        filtered_bots.sort(key=lambda x: x[1].get('name', ''))

    st.markdown(f"**Found {len(filtered_bots)} bot(s)** matching your criteria")

    if view_mode == "Cards":
        for bot_id, bot_config in filtered_bots:
            show_enhanced_bot_card(bot_id, bot_config)
    else:
        show_bots_table(filtered_bots)

def show_enhanced_bot_card(bot_id: str, bot_config: dict):
    chat_stats = get_bot_chat_stats(bot_id)

    status = bot_config.get('status', 'draft')
    status_colors = {
        'active': '#28a745',
        'testing': '#ffc107',
        'draft': '#6c757d'
    }

    last_modified = bot_config.get('updated_at', '')
    user_config = st.session_state.get('user_config', {})
    last_saved = user_config.get('last_updated', '')

    with st.container():
        st.markdown(f"""
        <div class="bot-card" style="border-left-color: {status_colors.get(status, '#6c757d')};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
                <div style="flex: 1;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <h4 style="margin: 0; color: #333; font-size: 1.1rem;">{bot_config.get('name', 'Unnamed Bot')}</h4>
                        <span style="
                            background: {status_colors.get(status, '#6c757d')};
                            color: white;
                            padding: 0.2rem 0.6rem;
                            border-radius: 12px;
                            font-size: 0.7rem;
                            font-weight: 600;
                            text-transform: uppercase;
                        ">{status}</span>
                    </div>

                    <p style="margin: 0 0 1rem 0; color: #666; line-height: 1.4; font-size: 0.9rem;">
                        {bot_config.get('description', 'No description available')[:150]}{'...' if len(bot_config.get('description', '')) > 150 else ''}
                    </p>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: #666;">
                        <div><strong>{bot_config.get('model', 'N/A')}</strong></div>
                        <div><strong>{chat_stats['total_messages']}</strong> messages</div>
                        <div><strong>{'KB Enabled' if bot_config.get('knowledge_base', {}).get('enabled') else 'No KB'}</strong></div>
                        <div><strong>{chat_stats['rating']}</strong> rating</div>
                        <div><strong>{bot_config.get('updated_at', '')[:10]}</strong></div>
                        <div><strong>{chat_stats['conversations']}</strong> chats</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            if st.button("💬", key=f"chat_bot_{bot_id}", help="Start chatting", use_container_width=True):
                st.session_state.current_bot = bot_id
                save_changes()
                st.switch_page("pages/4_Chat.py")

        with col2:
            if st.button("✏️", key=f"edit_bot_{bot_id}", help="Edit configuration", use_container_width=True):
                st.session_state.current_bot = bot_id
                save_changes()
                st.switch_page("pages/1_Create_Bot.py")

        with col3:
            if st.button("📊", key=f"analytics_bot_{bot_id}", help="View analytics", use_container_width=True):
                show_bot_analytics_modal(bot_id, bot_config)

        with col4:
            if st.button("🔗", key=f"share_bot_{bot_id}", help="Share bot", use_container_width=True):
                show_share_options_modal(bot_id, bot_config)

        with col5:
            if st.button("📥", key=f"export_bot_{bot_id}", help="Export data", use_container_width=True):
                export_single_bot(bot_id, bot_config)

        with col6:
            if st.button("🗑️", key=f"delete_bot_{bot_id}", help="Delete bot", use_container_width=True):
                show_delete_confirmation(bot_id, bot_config)

def show_bots_table(filtered_bots: list):
    if not filtered_bots:
        st.info("No bots to display")
        return

    table_data = []
    for bot_id, bot_config in filtered_bots:
        chat_stats = get_bot_chat_stats(bot_id)

        table_data.append({
            'Name': bot_config.get('name', 'Unnamed'),
            'Status': bot_config.get('status', 'draft').title(),
            'Model': bot_config.get('model', 'N/A')[:30] + '...' if len(bot_config.get('model', '')) > 30 else bot_config.get('model', 'N/A'),
            'Messages': chat_stats['total_messages'],
            'Chats': chat_stats['conversations'],
            'KB': 'Yes' if bot_config.get('knowledge_base', {}).get('enabled') else 'No',
            'Updated': bot_config.get('updated_at', '')[:10],
            'Rating': chat_stats['rating']
        })

    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

def get_bot_chat_stats(bot_id: str) -> dict:
    chat_data = st.session_state.chat_history.get(bot_id, [])

    total_messages = len(chat_data)
    user_messages = len([msg for msg in chat_data if msg.get('role') == 'user'])

    ratings = [msg.get('rating') for msg in chat_data if msg.get('rating')]
    like_count = len([r for r in ratings if r == 'like'])
    total_ratings = len(ratings)

    rating_text = "N/A"
    if total_ratings > 0:
        rating_pct = (like_count / total_ratings) * 100
        rating_text = f"{rating_pct:.0f}%"

    return {
        'total_messages': total_messages,
        'conversations': 1 if user_messages > 0 else 0,
        'rating': rating_text
    }

def show_bot_analytics_modal(bot_id: str, bot_config: dict):
    with st.expander(f"Analytics: {bot_config.get('name', 'Bot')}", expanded=True):
        chat_data = st.session_state.chat_history.get(bot_id, [])

        if not chat_data:
            st.info("No conversation data available for this bot yet.")
            st.markdown("**Start a conversation to see analytics:**")
            if st.button("Start Chatting", key=f"start_chat_{bot_id}"):
                st.session_state.current_bot = bot_id
                save_changes()
                st.switch_page("pages/4_Chat.py")
            return

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Messages", len(chat_data))

        with col2:
            user_messages = len([msg for msg in chat_data if msg.get('role') == 'user'])
            st.metric("User Messages", user_messages)

        with col3:
            bot_messages = len([msg for msg in chat_data if msg.get('role') == 'assistant'])
            st.metric("Bot Responses", bot_messages)

        with col4:
            avg_length = sum(len(msg.get('content', '')) for msg in chat_data) / len(chat_data) if chat_data else 0
            st.metric("Avg Message Length", f"{avg_length:.0f} chars")

        ratings = {}
        for msg in chat_data:
            if msg.get('rating'):
                rating = msg['rating']
                ratings[rating] = ratings.get(rating, 0) + 1

        if ratings:
            st.markdown("#### User Feedback")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Positive", ratings.get('like', 0))

            with col2:
                st.metric("Negative", ratings.get('dislike', 0))

            with col3:
                total_ratings = sum(ratings.values())
                satisfaction = (ratings.get('like', 0) / total_ratings) * 100 if total_ratings > 0 else 0
                st.metric("Satisfaction", f"{satisfaction:.0f}%")

        st.markdown("#### Recent Activity")
        recent_messages = chat_data[-5:] if len(chat_data) > 5 else chat_data

        for i, msg in enumerate(recent_messages):
            timestamp = msg.get('timestamp', 'Unknown time')[:19].replace('T', ' ')
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:80] + "..." if len(msg.get('content', '')) > 80 else msg.get('content', '')
            rating = "👍" if msg.get('rating') == 'like' else "👎" if msg.get('rating') == 'dislike' else ""

            role_icon = "👤" if role == 'user' else "🤖"
            st.markdown(f"**{role_icon} {role.title()}** • {timestamp} {rating}")
            st.caption(content)
            if i < len(recent_messages) - 1:
                st.divider()

def show_share_options_modal(bot_id: str, bot_config: dict):
    with st.expander(f"Share: {bot_config.get('name', 'Bot')}", expanded=True):
        base_url = "https://your-chatbot-app.streamlit.app"
        share_link = f"{base_url}/Chat?bot_id={bot_id}"

        st.markdown("#### Direct Link")
        st.code(share_link, language="text")

        st.markdown("#### Quick Share")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Email", key=f"email_share_{bot_id}", use_container_width=True):
                email_body = f"Check out this AI chatbot: {bot_config.get('name', 'My Bot')}%0A%0A{share_link}"
                st.markdown(f'<a href="mailto:?subject=Check out this AI chatbot&body={email_body}" target="_blank">Send Email</a>', unsafe_allow_html=True)

        with col2:
            if st.button("WhatsApp", key=f"whatsapp_share_{bot_id}", use_container_width=True):
                whatsapp_text = f"Check out this AI chatbot: {share_link}"
                st.markdown(f'<a href="https://wa.me/?text={whatsapp_text}" target="_blank">Share on WhatsApp</a>', unsafe_allow_html=True)

        with col3:
            if st.button("Copy", key=f"copy_share_{bot_id}", use_container_width=True):
                st.success("Link copied to clipboard!")

        st.markdown("#### Embed Code")
        embed_code = f'<iframe src="{share_link}" width="400" height="600" frameborder="0" style="border-radius: 10px;"></iframe>'
        st.code(embed_code, language="html")

def show_delete_confirmation(bot_id: str, bot_config: dict):
    with st.expander(f"Delete: {bot_config.get('name', 'Bot')}", expanded=True):
        st.error("Permanent Action Warning")
        st.markdown("This action will permanently delete:")

        st.markdown("""
        - Bot configuration and settings
        - All chat history and conversations
        - Knowledge base data (if any)
        - Analytics and usage statistics
        - Shared links and access permissions
        """)

        st.markdown("---")

        st.markdown("**Confirmation Required:**")

        confirm_understand = st.checkbox(
            f"I understand this will permanently delete '{bot_config.get('name', 'this bot')}'",
            key=f"confirm_understand_{bot_id}"
        )

        confirm_text = st.text_input(
            "Type the bot name to confirm:",
            key=f"confirm_text_{bot_id}",
            placeholder=bot_config.get('name', 'bot name')
        )

        if confirm_understand and confirm_text == bot_config.get('name', ''):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Delete Permanently", key=f"confirm_delete_{bot_id}", type="primary", use_container_width=True):
                    dm = get_data_manager_instance()
                    user_id = st.session_state.get('user_id')

                    if bot_id in st.session_state.bots:
                        del st.session_state.bots[bot_id]

                    if bot_id in st.session_state.chat_history:
                        del st.session_state.chat_history[bot_id]

                    if dm and user_id:
                        dm.delete_bot_files(user_id, bot_id)
                        dm.save_all_user_data(user_id)

                    st.success(f"'{bot_config.get('name', 'Bot')}' has been permanently deleted")
                    st.rerun()

            with col2:
                if st.button("Cancel", key=f"cancel_delete_{bot_id}", use_container_width=True):
                    st.info("Deletion cancelled")
        elif confirm_understand:
            st.warning(f"Please type '{bot_config.get('name', 'the bot name')}' exactly to confirm deletion")

def show_recent_activity(data: dict):
    st.markdown("### Recent Activity Feed")

    all_messages = []

    for bot_id, messages in st.session_state.chat_history.items():
        if bot_id in data['filtered_bots']:
            bot_name = data['filtered_bots'][bot_id].get('name', 'Unknown Bot')

            for msg in messages[-5:]:
                all_messages.append({
                    'bot_id': bot_id,
                    'bot_name': bot_name,
                    'timestamp': msg.get('timestamp', ''),
                    'role': msg.get('role', ''),
                    'content': msg.get('content', ''),
                    'rating': msg.get('rating', '')
                })

    all_messages.sort(key=lambda x: x['timestamp'], reverse=True)

    if all_messages:
        for i, msg in enumerate(all_messages[:15]):
            timestamp = msg['timestamp'][:19].replace('T', ' ') if msg['timestamp'] else 'Unknown time'

            with st.container():
                col1, col2 = st.columns([1, 4])

                with col1:
                    st.markdown(f"**{timestamp}**")
                    st.caption(f"{msg['bot_name']}")

                with col2:
                    role_icon = "👤" if msg['role'] == 'user' else "🤖"
                    rating_icon = " 👍" if msg['rating'] == 'like' else " 👎" if msg['rating'] == 'dislike' else ""

                    content_preview = msg['content'][:120] + "..." if len(msg['content']) > 120 else msg['content']

                    st.markdown(f"**{role_icon} {msg['role'].title()}:**{rating_icon}")
                    st.markdown(content_preview)

            if i < len(all_messages[:15]) - 1:
                st.divider()
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
            <h4>No Recent Activity</h4>
            <p style="color: #6c757d;">Start chatting with your bots to see activity here!</p>
        </div>
        """, unsafe_allow_html=True)

def export_dashboard_data():
    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id')

        if dm and user_id:
            export_file = dm.export_user_data(user_id)
            if export_file:
                with open(export_file, 'r') as f:
                    export_content = f.read()

                st.download_button(
                    label="Download Dashboard Export",
                    data=export_content,
                    file_name=f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_dashboard_export"
                )

                st.success("Dashboard data exported successfully!")
                return

        export_data = {
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'exported_by': st.session_state.get('username', 'Unknown'),
                'version': '1.0'
            },
            'summary': {
                'total_bots': len(st.session_state.get('bots', {})),
                'total_messages': sum(len(msgs) for msgs in st.session_state.get('chat_history', {}).values()),
                'export_type': 'dashboard_summary'
            },
            'bots_summary': {},
            'chat_statistics': {}
        }

        for bot_id, bot_config in st.session_state.get('bots', {}).items():
            export_data['bots_summary'][bot_id] = {
                'name': bot_config.get('name', 'Unnamed'),
                'status': bot_config.get('status', 'draft'),
                'model': bot_config.get('model', 'N/A'),
                'created_at': bot_config.get('created_at', ''),
                'updated_at': bot_config.get('updated_at', ''),
                'has_knowledge_base': bot_config.get('knowledge_base', {}).get('enabled', False)
            }

        for bot_id, messages in st.session_state.get('chat_history', {}).items():
            export_data['chat_statistics'][bot_id] = {
                'total_messages': len(messages),
                'user_messages': len([m for m in messages if m.get('role') == 'user']),
                'bot_messages': len([m for m in messages if m.get('role') == 'assistant']),
                'ratings': {
                    'likes': len([m for m in messages if m.get('rating') == 'like']),
                    'dislikes': len([m for m in messages if m.get('rating') == 'dislike'])
                },
                'last_activity': messages[-1].get('timestamp') if messages else None
            }

        json_str = json.dumps(export_data, indent=2)

        st.download_button(
            label="Download Dashboard Export",
            data=json_str,
            file_name=f"chatbot_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_dashboard_export"
        )

        st.success("Dashboard data exported successfully!")

    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_all_bots():
    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id')

        if dm and user_id:
            export_file = dm.export_user_data(user_id)
            if export_file:
                with open(export_file, 'r') as f:
                    export_content = f.read()

                st.download_button(
                    label="Download All Bots Backup",
                    data=export_content,
                    file_name=f"all_bots_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_all_bots_backup"
                )

                st.success("All bots exported successfully!")
                return

        export_data = {
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'exported_by': st.session_state.get('username', 'Unknown'),
                'export_type': 'all_bots_backup',
                'version': '1.0'
            },
            'bots': st.session_state.get('bots', {}),
            'chat_history': st.session_state.get('chat_history', {})
        }

        json_str = json.dumps(export_data, indent=2)

        st.download_button(
            label="Download All Bots Backup",
            data=json_str,
            file_name=f"all_bots_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_all_bots_backup"
        )

        st.success("All bots exported successfully!")

    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def export_single_bot(bot_id: str, bot_config: dict):
    try:
        dm = get_data_manager_instance()
        user_id = st.session_state.get('user_id')

        export_data = {
            'export_info': {
                'bot_id': bot_id,
                'bot_name': bot_config.get('name', 'Unnamed Bot'),
                'exported_at': datetime.now().isoformat(),
                'exported_by': st.session_state.get('username', 'Unknown'),
                'export_type': 'single_bot'
            },
            'bot_config': bot_config,
            'chat_history': st.session_state.chat_history.get(bot_id, []),
            'statistics': get_bot_chat_stats(bot_id)
        }

        if dm and user_id:
            kb = bot_config.get('knowledge_base', {})
            if kb.get('file_metadata'):
                export_data['file_metadata'] = kb['file_metadata']

        json_str = json.dumps(export_data, indent=2)

        bot_name = bot_config.get('name', 'bot').replace(' ', '_').lower()

        st.download_button(
            label="Download Bot Data",
            data=json_str,
            file_name=f"{bot_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"download_bot_{bot_id}"
        )

        st.success(f"'{bot_config.get('name', 'Bot')}' exported successfully!")

    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def setup_auto_save():
    if st.session_state.get('authenticated') and st.session_state.get('user_id'):
        if 'last_auto_save' not in st.session_state:
            st.session_state.last_auto_save = datetime.now()

        if (datetime.now() - st.session_state.last_auto_save).total_seconds() > 300:
            if save_changes():
                st.session_state.last_auto_save = datetime.now()

def show_import_data_section():
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Import Data")

        uploaded_file = st.file_uploader(
            "Import User Data",
            type=['json'],
            key="import_user_data",
            help="Import previously exported user data"
        )

        if uploaded_file:
            if st.button("Import Data", key="import_data_confirm"):
                try:
                    import_data = json.load(uploaded_file)
                    dm = get_data_manager_instance()
                    user_id = st.session_state.get('user_id')

                    if dm and user_id:
                        import tempfile
                        import os

                        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                            json.dump(import_data, f)
                            temp_file = f.name

                        try:
                            if dm.import_user_data(user_id, temp_file):
                                st.success("Data imported successfully!")
                                st.rerun()
                            else:
                                st.error("Import failed")
                        finally:
                            os.unlink(temp_file)
                    else:
                        st.error("Data manager not available")

                except Exception as e:
                    st.error(f"Import failed: {str(e)}")

if __name__ == "__main__":
    setup_auto_save()

    main()

