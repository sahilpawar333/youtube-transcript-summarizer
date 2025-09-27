from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BartForConditionalGeneration, BartTokenizer
from youtube_transcript_api import YouTubeTranscriptApi
import torch
import logging
import re

app = Flask(__name__)
CORS(app) 

logging.basicConfig(level=logging.INFO)

# Load BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Extract video ID from YouTube URL or ID
def extract_video_id(url_or_id):
    video_id_match = re.match(
        r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:v\/|watch\?v=|embed\/|shorts\/)|youtu\.be\/)?([\w-]{11})',
        url_or_id
    )
    if video_id_match:
        return video_id_match.group(1)
    return None

# Split transcript into chunks within model limits
def chunk_text(text, tokenizer, max_tokens=1024):
    tokens = tokenizer.encode(text, truncation=False)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i+max_tokens])

# Summarize a single chunk
def summarize_chunk(text, min_len=50, max_len=200):
    inputs = tokenizer([text], return_tensors='pt', truncation=True, max_length=1024).to(device)
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        length_penalty=2.0,
        max_length=max_len,
        min_length=min_len,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if not request.is_json:
            logging.error("Invalid JSON in request")
            return jsonify({'error': 'Invalid request format. JSON expected.'}), 400
        
        data = request.json
        url_or_id = data.get('video_id')

        if not url_or_id:
            logging.error("Video ID or URL is missing")
            return jsonify({'error': 'Video ID or URL is missing'}), 400

        video_id = extract_video_id(url_or_id)
        if not video_id:
            logging.error(f"Invalid YouTube URL or Video ID: {url_or_id}")
            return jsonify({'error': 'Invalid YouTube URL or Video ID'}), 400

        logging.info(f"Fetching transcript for video ID: {video_id}")

        try:
            captions = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            logging.error(f"Failed to fetch captions for video ID {video_id}: {e}")
            return jsonify({'error': f'Failed to fetch captions: {str(e)}'}), 404

        transcript = " ".join([c['text'] for c in captions])

        if not transcript.strip():
            return jsonify({'error': 'Transcript is empty'}), 404

        # Split transcript into chunks and summarize each
        chunk_summaries = []
        for chunk in chunk_text(transcript, tokenizer):
            summary = summarize_chunk(chunk, min_len=50, max_len=200)
            chunk_summaries.append(summary)

        # Combine summaries into a final summary
        combined_text = " ".join(chunk_summaries)
        final_summary = summarize_chunk(combined_text, min_len=100, max_len=300)

        logging.info(f"Summary generated for video ID {video_id}")

        return jsonify({
            'summary': final_summary,
            'chunks': chunk_summaries 
        })

    except Exception as e:
        logging.error(f"Error in summarization: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
