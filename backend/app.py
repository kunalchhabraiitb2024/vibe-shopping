from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import os
from agent import VibeShoppingAgent

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
template_dir = os.path.join(project_root, 'frontend', 'templates')
static_dir = os.path.join(project_root, 'frontend', 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent = VibeShoppingAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'response_type': 'error',
                'message': 'No query provided'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'response_type': 'error',
                'message': 'Empty query provided'
            }), 400
        
        logger.info(f"Processing query: {query}")
        
        response = agent.process_query(query)
        
        logger.info(f"Agent response: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'response_type': 'error',
            'message': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
