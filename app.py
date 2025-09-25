from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import logging
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to extract video ID from URL or video ID
def extract_video_id(url_or_id):
    video_id_match = re.match(r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:v\/|watch\?v=|embed\/|shorts\/)|youtu\.be\/)?([\w-]{11})', url_or_id)
    if video_id_match:
        return video_id_match.group(1)
    return None

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Check if the request contains valid JSON
        if not request.is_json:
            logging.error("Invalid JSON in request")
            return jsonify({'error': 'Invalid request format. JSON expected.'}), 400
        
        # Get the URL or video ID from the request
        data = request.json
        url_or_id = data.get('video_id')

        if not url_or_id:
            logging.error("Video ID or URL is missing")
            return jsonify({'error': 'Video ID or URL is missing'}), 400

        # Extract the video ID
        video_id = extract_video_id(url_or_id)
        if not video_id:
            logging.error(f"Invalid YouTube URL or Video ID: {url_or_id}")
            return jsonify({'error': 'Invalid YouTube URL or Video ID'}), 400

        logging.info(f"Fetching transcript for video ID: {video_id}")

        # Fetch captions using YouTubeTranscriptApi
        try:
            captions = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            logging.error(f"Failed to fetch captions for video ID {video_id}: {e}")
            return jsonify({'error': f'Failed to fetch captions: {str(e)}'}), 500

        transcript = " ".join([caption['text'] for caption in captions])

        # Tokenize the transcript
        inputs = tokenizer([transcript], max_length=1024, return_tensors='pt', truncation=True)
        inputs = inputs.to(device)

        # Generate the summary
        summary_ids = model.generate(
            inputs['input_ids'], 
            num_beams=4, 
            length_penalty=2.0, 
            max_length=500, 
            min_length=180, 
            no_repeat_ngram_size=3
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        logging.info(f"Summary generated for video ID {video_id}")

        return jsonify({'summary': summary})
    
    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
