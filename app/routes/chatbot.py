from flask import Blueprint, request, jsonify
from app.chatbot.model import get_response
from app.utils.logger import get_logger

chatbot_bp = Blueprint('chatbot', __name__)
logger = get_logger(__name__)


@chatbot_bp.route('/get_response', methods=['POST'])
def chat_response():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'response': "Please type a message."})

    logger.info("Chatbot query received.")
    try:
        response = get_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({'response': "Sorry, I encountered an error. Please try again."})