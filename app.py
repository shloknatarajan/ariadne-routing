from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from run_agents import run_agent
import json
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        print('hi')
        from stream_router import StreamRouter
        import pickle
        data = request.get_json()
        message = data.get('message', '')
        # Load the state and create a new router
        print('hello')
        with open('router_state.pkl', 'rb') as f:
            state = pickle.load(f)
            
        new_router = StreamRouter(
            state['agents'],
            embedding_dim=state['embedding_dim'],
            learning_rate=state['learning_rate'],
            min_samples=state['min_samples']
        )
        new_router.clusters = state['clusters']
        new_router.agent_embeddings = state['agent_embeddings']
        predicted_agent = new_router.inference(message)
        
        def generate():
            try:
                # First message - prediction
                yield json.dumps({
                    "type": "prediction",
                    "content": str(predicted_agent)  # Ensure it's a string
                }, ensure_ascii=False).strip() + '\n'
                
                # Status message
                yield json.dumps({
                    "type": "status",
                    "content": f"Running {str(predicted_agent)} agent..."
                }, ensure_ascii=False).strip() + '\n'
                
                # Run the agent
                result = run_agent(predicted_agent, message)
                
                # Ensure result is JSON-safe
                try:
                    result_content = str(result.raw) if len(str(result.raw)) < 100 else "agent is done running."
                    yield json.dumps({
                        "type": "result",
                        "content": result_content
                    }, ensure_ascii=False).strip() + '\n'
                except Exception as e:
                    yield json.dumps({
                        "type": "error",
                        "content": f"Error processing result: {str(e)}"
                    }, ensure_ascii=False).strip() + '\n'
                    
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "content": f"Stream error: {str(e)}"
                }, ensure_ascii=False).strip() + '\n'

        return Response(generate(), mimetype='application/x-ndjson')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed port to 5001