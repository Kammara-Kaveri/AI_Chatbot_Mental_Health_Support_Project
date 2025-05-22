import logging
from flask import Flask, render_template, request, session, jsonify
from test_model import generate_response_rule_based
import uuid
from datetime import datetime
import os

# Configure the root logger with a simple format (for Werkzeug/Flask logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    force=True
)

# Create a custom logger for the chatbot
chatbot_logger = logging.getLogger('chatbot')
chatbot_logger.setLevel(logging.INFO)

# Create a file handler for session_logs.log
file_handler = logging.FileHandler('session_logs.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] SessionID:%(session_id)s IP:%(ip)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the chatbot logger
chatbot_logger.addHandler(file_handler)

# Custom filter to add session ID and IP to log records
class SessionFilter(logging.Filter):
    def filter(self, record):
        # Safely access session and request, default to 'N/A' if not in context
        try:
            record.session_id = session.get('_id', 'N/A')
        except RuntimeError:
            record.session_id = 'N/A'
        
        try:
            record.ip = request.remote_addr if request.environ.get('HTTP_X_FORWARDED_FOR') is None else request.environ['HTTP_X_FORWARDED_FOR']
        except RuntimeError:
            record.ip = 'N/A'
        return True

# Add the filter to the chatbot logger's file handler
file_handler.addFilter(SessionFilter())

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_123')

@app.before_request
def before_request():
    # Assign a session ID if not present
    if '_id' not in session:
        session['_id'] = str(uuid.uuid4())
        chatbot_logger.info("New session started")

@app.route('/')
def home():
    if 'conversation' not in session:
        session['conversation'] = []
    return render_template('chat.html', conversation=session['conversation'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    chatbot_logger.info(f"User message: {user_input}")
    
    response = generate_response_rule_based(user_input)
    session['conversation'].append(('User', user_input))
    session['conversation'].append(('Bot', response))
    session.modified = True
    
    return jsonify({'response': response})

@app.route('/clear', methods=['POST'])
def clear():
    chatbot_logger.info("Chat cleared")
    session['conversation'] = []
    session.modified = True
    return jsonify({'status': 'cleared'})

@app.route('/end', methods=['POST'])
def end():
    chatbot_logger.info("Session ended")
    session.clear()
    return jsonify({'status': 'ended'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)