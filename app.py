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
            # Each response needs to be properly formatted and end with a newline
            yield json.dumps({
                "type": "prediction",
                "content": f"The agent to use is {predicted_agent}"
            }).strip() + '\n'
            
            yield json.dumps({
                "type": "status",
                "content": f"Running {predicted_agent} agent..."
            }).strip() + '\n'
            
            # Run the agent
            result = run_agent(predicted_agent, message)
            
            # Final response - result
            if len(result.raw) < 100:
                yield json.dumps({
                    "type": "result",
                    "content": result.raw
                }).strip() + '\n'
            else:
                yield json.dumps({
                    "type": "result",
                    "content": "agent is done running."
                }).strip() + '\n'

        return Response(generate(), mimetype='application/x-ndjson')
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed port to 5001