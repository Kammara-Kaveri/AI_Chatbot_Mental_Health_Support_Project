import logging
from flask import Flask, render_template, request, session, jsonify
from test_model import generate_response_rule_based
import uuid
from datetime import datetime
import os  # Added for environment variables

# Set up logging configuration
logging.basicConfig(
    filename='session_logs.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] SessionID:%(session_id)s IP:%(ip)s - %(message)s'
)

app = Flask(__name__)
# Use environment variable for secret key in production, fallback for development
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_123')

# Custom filter to add session ID and IP to log records
class SessionFilter(logging.Filter):
    def filter(self, record):
        record.session_id = session.get('_id', 'N/A')  # Get session ID
        record.ip = request.remote_addr if request.environ.get('HTTP_X_FORWARDED_FOR') is None else request.environ['HTTP_X_FORWARDED_FOR']
        return True

# Add the filter to the logger
app.logger.addFilter(SessionFilter())

@app.before_request
def before_request():
    # Assign a session ID if not present
    if '_id' not in session:
        session['_id'] = str(uuid.uuid4())
        app.logger.info("New session started")

@app.route('/')
def home():
    if 'conversation' not in session:
        session['conversation'] = []
    return render_template('chat.html', conversation=session['conversation'])

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    app.logger.info(f"User message: {user_input}")
    
    response = generate_response_rule_based(user_input)
    session['conversation'].append(('User', user_input))
    session['conversation'].append(('Bot', response))
    session.modified = True
    
    return jsonify({'response': response})

@app.route('/clear', methods=['POST'])
def clear():
    app.logger.info("Chat cleared")
    session['conversation'] = []
    session.modified = True
    return jsonify({'status': 'cleared'})

@app.route('/end', methods=['POST'])
def end():
    app.logger.info("Session ended")
    session.clear()
    return jsonify({'status': 'ended'})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)