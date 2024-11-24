# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Remove Cross-Origin restrictions

# Load Whisper model (model size can be changed as needed)
model = whisper.load_model("small")

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    # Save as temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio:
        audio_file.save(temp_audio.name)
        
        try:
            # Transcribe using Whisper
            result = model.transcribe(temp_audio.name, language='ja')
            # Delete temporary file
            os.unlink(temp_audio.name)
            
            return jsonify({
                'text': result['text'],
                'segments': result['segments']
            })
        
        except Exception as e:
            # Delete temporary file even if error occurs
            os.unlink(temp_audio.name)
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)