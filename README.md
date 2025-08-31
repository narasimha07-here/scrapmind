# 🤖 ScrapMind – No-Code Chatbot Builder

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-blue)](https://scrapmind.streamlit.app/)

Create intelligent AI chatbots **without writing a single line of code!**  
ScrapMind lets you build, deploy, and manage AI-powered chatbots with **persistent storage, voice integration, knowledge base support, and FastAPI export** — all through an intuitive web interface.

---

## 🚀 Live Demo

👉 Try it now: [https://scrapmind.streamlit.app/](https://scrapmind.streamlit.app/)

---

## ✨ Features

- **No-Code Bot Creation** – Step-by-step wizard for easy chatbot setup.  
- **Knowledge Base Integration** – Upload docs, paste text, add URLs, or create FAQ datasets.  
- **Voice Response Support** – Multiple providers (Murf AI, OpenAI TTS, ElevenLabs, Google TTS, Azure, Amazon Polly).  
- **Multiple AI Models** – Choose from OpenRouter’s free & paid models.  
- **Advanced Settings** – Fine-tune context, response style, error handling, system instructions.  
- **Persistent Data Storage** – Save users, bots, chat history, and uploaded files across sessions.  
- **User Authentication** – Secure login & signup system for personalized bots.  
- **Interactive Chat Interface** – Streaming responses, voice playback, debug mode.  
- **Analytics Dashboard** – Track usage, satisfaction, response times, and more.  
- **FastAPI Export** – Auto-generate backend API with RAG & voice support.  
- **Data Management** – Backup, import/export, and cleanup tools.  

---

## ⚡ Getting Started

### ✅ Prerequisites
- Python **3.11+**
- [Streamlit](https://streamlit.io/)  
- (Optional) API keys: OpenRouter, Murf AI, OpenAI, ElevenLabs, etc.

### 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/narasimha07-here/scrapmind.git
   cd scrapmind
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🎯 Usage

1. **Sign Up / Login** – Create an account or log in.  
2. **Create a Bot** – Configure name, personality, AI model, voice, and knowledge base.  
3. **Upload Knowledge** – Add documents, text, URLs, or FAQs.  
4. **Test Chat** – Interact with your bot in real-time (text + voice).  
5. **Analytics** – Monitor usage and performance.  
6. **Export FastAPI** – Generate backend code for production deployment.  
7. **Manage Data** – Backup, restore, or clean data.  

---

## 📂 Project Structure

```
/AUTO_CHAT_BOT
│
├── app.py                      # Main Streamlit app entry point
├── components/
│   ├── auth.py                 # User authentication system
│   ├── bot_creator.py          # Bot creation logic
│   ├── chat_interface.py       # Chat UI + LLM interaction
│   ├── data_manager.py         # Persistent storage
│   ├── knowledge_processor.py  # RAG embeddings + vectorstore
│   ├── voice_config.py         # Voice synthesis setup
│   └── analytics.py            # Bot analytics dashboard
│
├── data/                       # Local data storage
│   ├── users/
│   ├── bots/
│   ├── files/
│   ├── vectordb/
│   └── backups/
│
├── pages/                      # Streamlit multipage routes
│   ├── 1_🤖_Create_Bot.py
│   ├── 2_📊_Dashboard.py
│   ├── 3_⚙️_Settings.py
│   └── 4_🎯_Chat.py
│
├── requirements.txt            # Dependencies
└── .streamlit/config.toml      # Streamlit settings
```

---

## ⚙️ Configuration

- Manage **API keys** (OpenRouter, voice providers, etc.) securely via user settings.  
- Customize **default AI models & embeddings** per bot.  
- Adjust **voice settings** (language, speed, volume).  

---

## 🚀 FastAPI Export

Generate a ready-to-deploy **FastAPI backend** with:

- Chat endpoints (streaming + voice).  
- Knowledge base RAG integration.  
- Dockerfile & env setup.  
- Swagger UI for API docs.  

---

## 🤝 Contributing

Contributions are welcome! 🎉  
Feel free to open an **issue** or submit a **pull request** for bug fixes, new features, or improvements.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/)  
- [OpenRouter](https://openrouter.ai/)  
- [HuggingFace](https://huggingface.co/)  
- [LangChain](https://langchain.com/)  
- Voice APIs: Murf AI, OpenAI, ElevenLabs, Google TTS, Azure, Amazon Polly  

---

⭐ *Build your AI chatbot today — no coding required!*  
