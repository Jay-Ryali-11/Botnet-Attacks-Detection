import os
import json
import random
import torch
import torch.nn as nn
from flask import current_app
from app.chatbot.utils import tokenize, bag_of_words
from app.utils.logger import get_logger

logger = get_logger(__name__)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out


_chatbot_model = None
_chatbot_data = None
_intents = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_chatbot():
    global _chatbot_model, _chatbot_data, _intents

    if _chatbot_model is not None:
        return

    chatbot_dir = current_app.config['CHATBOT_DATA_DIR']

    intents_path = os.path.join(chatbot_dir, 'intents.json')
    with open(intents_path, 'r') as f:
        _intents = json.load(f)

    data_path = os.path.join(chatbot_dir, 'data.pth')
    _chatbot_data = torch.load(data_path, map_location=device)

    model = NeuralNet(
        _chatbot_data['input_size'],
        _chatbot_data['hidden_size'],
        _chatbot_data['output_size']
    ).to(device)
    model.load_state_dict(_chatbot_data['model_state'])
    model.eval()
    _chatbot_model = model
    logger.info("Chatbot model loaded.")


def get_response(message):
    if _chatbot_model is None:
        load_chatbot()

    tokens = tokenize(message)
    X = bag_of_words(tokens, _chatbot_data['all_words'])
    X = torch.from_numpy(X).unsqueeze(0).to(device)

    output = _chatbot_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = _chatbot_data['tags'][predicted.item()]
    prob = torch.softmax(output, dim=1)[0][predicted.item()].item()

    if prob > 0.6:
        for intent in _intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

    for intent in _intents['intents']:
        if intent['tag'] == 'unknown_query':
            return random.choice(intent['responses'])

    return "I'm not sure how to help with that. Please visit our Support page."