import sys
import os

from .knowledge_processor import KnowledgeProcessor
from .bot_creator import BotCreator
from .chat_interface import ChatInterface

__all__ = ['KnowledgeProcessor', 'BotCreator', 'ChatInterface']

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
