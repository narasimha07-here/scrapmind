import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict
import json
from collections import Counter

class ChatbotAnalytics:
    
    def __init__(self):
        self.analytics_file = "data/analytics/bot_analytics.json"
        self.ensure_analytics_file()
    
    def ensure_analytics_file(self):
        import os
        os.makedirs(os.path.dirname(self.analytics_file), exist_ok=True)
        
        if not os.path.exists(self.analytics_file):
            with open(self.analytics_file, 'w') as f:
                json.dump({}, f)
    
    def track_interaction(self, bot_id: str, interaction_data: Dict):
        try:
            with open(self.analytics_file, 'r') as f:
                analytics = json.load(f)
            
            if bot_id not in analytics:
                analytics[bot_id] = {
                    'total_interactions': 0,
                    'total_messages': 0,
                    'daily_stats': {},
                    'user_satisfaction': [],
                    'response_times': [],
                    'topics': [],
                    'error_count': 0,
                    'created_at': datetime.now().isoformat()
                }
            
            bot_analytics = analytics[bot_id]
            
            bot_analytics['total_interactions'] += 1
            bot_analytics['total_messages'] += interaction_data.get('message_count', 1)
            
            today = datetime.now().strftime('%Y-%m-%d')
            if today not in bot_analytics['daily_stats']:
                bot_analytics['daily_stats'][today] = {
                    'interactions': 0,
                    'messages': 0,
                    'avg_response_time': 0,
                    'user_ratings': []
                }
            
            daily_stats = bot_analytics['daily_stats'][today]
            daily_stats['interactions'] += 1
            daily_stats['messages'] += interaction_data.get('message_count', 1)
            
            response_time = interaction_data.get('response_time', 0)
            if response_time > 0:
                bot_analytics['response_times'].append(response_time)
                daily_stats['avg_response_time'] = response_time
            
            rating = interaction_data.get('rating')
            if rating:
                bot_analytics['user_satisfaction'].append(rating)
                daily_stats['user_ratings'].append(rating)
            
            topics = interaction_data.get('topics', [])
            bot_analytics['topics'].extend(topics)
            
            if interaction_data.get('error'):
                bot_analytics['error_count'] += 1
            
            with open(self.analytics_file, 'w') as f:
                json.dump(analytics, f, indent=2)
                
        except Exception as e:
            st.error(f"Error tracking interaction: {str(e)}")
    
    def get_bot_analytics(self, bot_id: str) -> Dict:
        try:
            with open(self.analytics_file, 'r') as f:
                analytics = json.load(f)
            
            return analytics.get(bot_id, {})
        except:
            return {}
    
    def show_analytics_dashboard(self, bot_id: str, bot_config: Dict):

        st.title("📊 Bot Analytics Dashboard")
        st.markdown(f"**Bot:** {bot_config.get('name', 'Unknown')}")
        
        analytics_data = self.get_bot_analytics(bot_id)
        
        if not analytics_data:
            st.info("📈 No analytics data available yet. Start chatting to see insights!")
            return
        
        self.show_overview_metrics(analytics_data)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.show_usage_trends(analytics_data)
            self.show_response_time_chart(analytics_data)
        
        with col2:
            self.show_user_satisfaction(analytics_data)
            self.show_topic_analysis(analytics_data)
        
        st.markdown("---")
        
        self.show_detailed_analytics(analytics_data)
    
    def show_overview_metrics(self, analytics_data: Dict):

        st.markdown("### 📈 Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_interactions = analytics_data.get('total_interactions', 0)
            st.metric("Total Conversations", total_interactions)
        
        with col2:
            total_messages = analytics_data.get('total_messages', 0)
            st.metric("Total Messages", total_messages)
        
        with col3:
            avg_messages = total_messages / max(total_interactions, 1)
            st.metric("Avg Messages/Chat", f"{avg_messages:.1f}")
        
        with col4:
            satisfaction = analytics_data.get('user_satisfaction', [])
            if satisfaction:
                positive_ratings = sum(1 for rating in satisfaction if rating in ['like', 'good', 5, 4])
                satisfaction_rate = (positive_ratings / len(satisfaction)) * 100
                st.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
            else:
                st.metric("Satisfaction Rate", "N/A")
        
        with col5:
            response_times = analytics_data.get('response_times', [])
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
            else:
                st.metric("Avg Response Time", "N/A")
    
    def show_usage_trends(self, analytics_data: Dict):

        st.markdown("#### 📈 Usage Trends")
        
        daily_stats = analytics_data.get('daily_stats', {})
        
        if not daily_stats:
            st.info("No usage data available yet")
            return
        
        dates = []
        interactions = []
        messages = []
        
        for date, stats in daily_stats.items():
            dates.append(pd.to_datetime(date))
            interactions.append(stats.get('interactions', 0))
            messages.append(stats.get('messages', 0))
        
        df = pd.DataFrame({
            'Date': dates,
            'Conversations': interactions,
            'Messages': messages
        })
        
        df = df.sort_values('Date')
        
        fig = px.line(
            df, 
            x='Date', 
            y=['Conversations', 'Messages'],
            title="Daily Usage Trends"
        )
        
        fig.update_layout(
            showlegend=True,
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_response_time_chart(self, analytics_data: Dict):

        st.markdown("#### ⏱️ Response Time Distribution")
        
        response_times = analytics_data.get('response_times', [])
        
        if not response_times:
            st.info("No response time data available")
            return
        
        fig = px.histogram(
            x=response_times,
            nbins=20,
            title="Response Time Distribution",
            labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
        )
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Time", f"{min(response_times):.1f}s")
        
        with col2:
            st.metric("Avg Time", f"{sum(response_times) / len(response_times):.1f}s")
        
        with col3:
            st.metric("Max Time", f"{max(response_times):.1f}s")
    
    def show_user_satisfaction(self, analytics_data: Dict):

        st.markdown("#### 😊 User Satisfaction")
        
        satisfaction = analytics_data.get('user_satisfaction', [])
        
        if not satisfaction:
            st.info("No satisfaction data available")
            return
        
        rating_counts = Counter(satisfaction)
        
        labels = []
        values = []
        colors = []
        
        rating_mapping = {
            'like': {'label': '👍 Positive', 'color': '#28a745'},
            'dislike': {'label': '👎 Negative', 'color': '#dc3545'},
            'good': {'label': '😊 Good', 'color': '#28a745'},
            'bad': {'label': '😞 Bad', 'color': '#dc3545'},
            'neutral': {'label': '😐 Neutral', 'color': '#ffc107'}
        }
        
        for rating, count in rating_counts.items():
            mapping = rating_mapping.get(rating, {'label': str(rating), 'color': '#6c757d'})
            labels.append(mapping['label'])
            values.append(count)
            colors.append(mapping['color'])
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.4
        )])
        
        fig.update_layout(
            title="User Satisfaction Distribution",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_topic_analysis(self, analytics_data: Dict):
        
        st.markdown("#### 🏷️ Popular Topics")
        
        topics = analytics_data.get('topics', [])
        
        if not topics:
            st.info("No topic data available")
            return
        
        topic_counts = Counter(topics)
        most_common = topic_counts.most_common(10)
        
        if most_common:
            topics_df = pd.DataFrame(most_common, columns=['Topic', 'Count'])
            
            fig = px.bar(
                topics_df,
                x='Count',
                y='Topic',
                orientation='h',
                title="Most Discussed Topics"
            )
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topics identified yet")
    
    def show_detailed_analytics(self, analytics_data: Dict):

        st.markdown("### 📋 Detailed Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Daily Breakdown", "🔍 Performance", "💬 Conversations", "🚨 Issues"])
        
        with tab1:
            self.show_daily_breakdown(analytics_data)
        
        with tab2:
            self.show_performance_analysis(analytics_data)
        
        with tab3:
            self.show_conversation_analysis(analytics_data)
        
        with tab4:
            self.show_issues_analysis(analytics_data)
    
    def show_daily_breakdown(self, analytics_data: Dict):
    
        st.markdown("#### 📅 Daily Activity Breakdown")
        
        daily_stats = analytics_data.get('daily_stats', {})
        
        if not daily_stats:
            st.info("No daily data available")
            return
        
        daily_data = []
        
        for date, stats in daily_stats.items():
            daily_data.append({
                'Date': date,
                'Conversations': stats.get('interactions', 0),
                'Messages': stats.get('messages', 0),
                'Avg Response Time': f"{stats.get('avg_response_time', 0):.1f}s",
                'User Ratings': len(stats.get('user_ratings', []))
            })
        
        df = pd.DataFrame(daily_data)
        df = df.sort_values('Date', ascending=False)
        
        st.dataframe(df, use_container_width=True)
        
        if st.button("📥 Export Daily Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"daily_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    def show_performance_analysis(self, analytics_data: Dict):

        st.markdown("#### 🔍 Performance Analysis")
        
        response_times = analytics_data.get('response_times', [])
        
        if response_times:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Response Time Statistics:**")
                
                stats_data = {
                    'Metric': ['Minimum', 'Average', 'Maximum', 'Median', '95th Percentile'],
                    'Time (seconds)': [
                        f"{min(response_times):.2f}",
                        f"{sum(response_times) / len(response_times):.2f}",
                        f"{sorted(response_times)[len(response_times)//2]:.2f}",
                        f"{max(response_times):.2f}",
                        f"{sorted(response_times)[int(len(response_times)*0.95)]:.2f}"
                    ]
                }
                
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            with col2:
                st.markdown("**Performance Insights:**")
                
                avg_time = sum(response_times) / len(response_times)
                
                if avg_time < 2:
                    st.success("✅ Excellent response time performance")
                elif avg_time < 5:
                    st.info("ℹ️ Good response time performance")
                else:
                    st.warning("⚠️ Response time could be improved")
                
                slow_responses = [t for t in response_times if t > 10]
                if slow_responses:
                    st.warning(f"🐌 {len(slow_responses)} slow responses (>10s) detected")
        
        error_count = analytics_data.get('error_count', 0)
        total_interactions = analytics_data.get('total_interactions', 0)
        
        if total_interactions > 0:
            error_rate = (error_count / total_interactions) * 100
            
            st.markdown("**Error Rate Analysis:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Errors", error_count)
                st.metric("Error Rate", f"{error_rate:.2f}%")
            
            with col2:
                if error_rate < 1:
                    st.success("✅ Low error rate")
                elif error_rate < 5:
                    st.info("ℹ️ Moderate error rate")
                else:
                    st.error("❌ High error rate - needs attention")
    
    def show_conversation_analysis(self, analytics_data: Dict):

        st.markdown("#### 💬 Conversation Patterns")
        
        total_interactions = analytics_data.get('total_interactions', 0)
        total_messages = analytics_data.get('total_messages', 0)
        
        if total_interactions > 0:
            avg_length = total_messages / total_interactions
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Conversation Length", f"{avg_length:.1f} messages")
            
            with col2:
                if avg_length > 5:
                    engagement = "High"
                    color = "🟢"
                elif avg_length > 2:
                    engagement = "Medium"
                    color = "🟡"
                else:
                    engagement = "Low"
                    color = "🔴"
                
                st.metric("Engagement Level", f"{color} {engagement}")
            
            with col3:
                return_rate = min(100, (total_interactions / max(total_interactions, 1)) * 20)
                st.metric("Estimated Return Rate", f"{return_rate:.0f}%")
            
            st.markdown("**Insights:**")
            
            if avg_length > 5:
                st.success("✅ Users are having engaging conversations")
            elif avg_length > 2:
                st.info("ℹ️ Average engagement level")
            else:
                st.warning("⚠️ Users may be having difficulty or losing interest quickly")
    
    def show_issues_analysis(self, analytics_data: Dict):
        
        st.markdown("#### 🚨 Issues & Recommendations")
        
        issues = []
        recommendations = []
        
        response_times = analytics_data.get('response_times', [])
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time > 5:
                issues.append("Slow average response time")
                recommendations.append("Consider using a faster AI model or optimizing your knowledge base")
        
        error_count = analytics_data.get('error_count', 0)
        total_interactions = analytics_data.get('total_interactions', 0)
        
        if total_interactions > 0:
            error_rate = (error_count / total_interactions) * 100
            if error_rate > 5:
                issues.append("High error rate")
                recommendations.append("Review bot configuration and knowledge base content")
        
        satisfaction = analytics_data.get('user_satisfaction', [])
        if satisfaction:
            positive_ratings = sum(1 for rating in satisfaction if rating in ['like', 'good'])
            satisfaction_rate = (positive_ratings / len(satisfaction)) * 100
            
            if satisfaction_rate < 70:
                issues.append("Low user satisfaction")
                recommendations.append("Review bot responses and consider improving personality or knowledge base")
        
        total_messages = analytics_data.get('total_messages', 0)
        if total_interactions > 0:
            avg_length = total_messages / total_interactions
            if avg_length < 2:
                issues.append("Low conversation engagement")
                recommendations.append("Review welcome message and bot personality settings")
        
        if issues:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 Issues Identified:**")
                for issue in issues:
                    st.warning(f"• {issue}")
            
            with col2:
                st.markdown("**💡 Recommendations:**")
                for rec in recommendations:
                    st.info(f"• {rec}")
        else:
            st.success("✅ No major issues identified. Your bot is performing well!")
        
        score = 100
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time > 10:
                score -= 20
            elif avg_time > 5:
                score -= 10
        
        if total_interactions > 0:
            error_rate = (error_count / total_interactions) * 100
            score -= min(30, error_rate * 3)
        
        if satisfaction:
            positive_ratings = sum(1 for rating in satisfaction if rating in ['like', 'good'])
            satisfaction_rate = (positive_ratings / len(satisfaction)) * 100
            score -= max(0, (80 - satisfaction_rate) * 0.5)
        
        score = max(0, int(score))
        
        st.markdown("---")
        st.markdown("### 🎯 Overall Performance Score")
        
        col1, col2, col3 = st.columns(3)
        
        with col2:
            if score >= 90:
                color = "🟢"
                status = "Excellent"
            elif score >= 70:
                color = "🟡"
                status = "Good"
            elif score >= 50:
                color = "🟠"
                status = "Fair"
            else:
                color = "🔴"
                status = "Needs Improvement"
            
            st.metric("Performance Score", f"{color} {score}/100")
            st.markdown(f"**Status:** {status}")
    
    def export_analytics_report(self, bot_id: str, bot_config: Dict):
        
        analytics_data = self.get_bot_analytics(bot_id)
        
        if not analytics_data:
            st.error("No analytics data to export")
            return
        
        report = {
            'bot_info': {
                'id': bot_id,
                'name': bot_config.get('name', 'Unknown'),
                'model': bot_config.get('model', 'Unknown'),
                'created_at': bot_config.get('created_at', 'Unknown')
            },
            'summary': {
                'total_interactions': analytics_data.get('total_interactions', 0),
                'total_messages': analytics_data.get('total_messages', 0),
                'error_count': analytics_data.get('error_count', 0),
                'avg_response_time': sum(analytics_data.get('response_times', [])) / len(analytics_data.get('response_times', [1])),
                'satisfaction_rate': len([r for r in analytics_data.get('user_satisfaction', []) if r in ['like', 'good']]) / len(analytics_data.get('user_satisfaction', [1])) * 100
            },
            'daily_stats': analytics_data.get('daily_stats', {}),
            'topics': Counter(analytics_data.get('topics', [])),
            'exported_at': datetime.now().isoformat()
        }
        
        report_json = json.dumps(report, indent=2)
        
        st.download_button(
            label="📥 Download Analytics Report",
            data=report_json,
            file_name=f"analytics_report_{bot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
