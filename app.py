from flask import Flask, request, jsonify
from main import extract_video_id, get_transcript, split_text, store_chunks, get_relevant_chunks, ask_bot
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"], supports_credentials=True)

cache = {}

@app.route('/preprocess', methods=['POST', 'OPTIONS'])
def preprocess():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200  # Preflight response

    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        video_id = extract_video_id(url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        if video_id in cache:
            return jsonify({"message": "Already processed."}), 200

        video_id, transcript = get_transcript(url)
        if not transcript:
            return jsonify({"error": "Give link of video with English language."}), 404
        chunks = split_text(transcript)
        vectorstore = store_chunks(chunks)
        cache[video_id] = vectorstore
        return jsonify({"message": "Processing complete."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask', methods=['POST', 'OPTIONS'])
def ask():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200

    data = request.get_json()
    url = data.get('url')
    question = data.get('question')

    if not url or not question:
        return jsonify({"error": "URL and question are required"}), 400

    try:
        video_id = extract_video_id(url)

        if video_id in cache:
            vectorstore = cache[video_id]
        else:
            video_id, transcript = get_transcript(url)
            if not transcript:
                return jsonify({"error": "Give link of video with English language."}), 404
            chunks = split_text(transcript)
            vectorstore = store_chunks(chunks)
            cache[video_id] = vectorstore

        relevant_chunks = get_relevant_chunks(vectorstore, question)
        answer = ask_bot(question, relevant_chunks)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
