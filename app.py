import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import logging
from retrieval_system import HybridVideoRetrievalSystem 
import config

log_file = "system.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    search_system = HybridVideoRetrievalSystem(re_ingest=False)
    logger.info("✅ Search system loaded successfully!")
except Exception as e:
    logger.error(f"💥 Failed to load search system: {e}")
    search_system = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_api():
    """
    The core API endpoint.  
    It receives the query data, calls the search system, and returns results.
    """
    if not search_system:
        return jsonify({"error": "Search system is not available."}), 500

    query_data = request.get_json()
    if not query_data:
        return jsonify({"error": "Invalid input: No JSON data received."}), 400

    logger.info(f"Received search request: {query_data}")

    try:
        results = search_system.search(query_data=query_data, top_k=20)
        return jsonify(results)
    except Exception as e:
        logger.error(f"An error occurred during search: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during search."}), 500

@app.route('/frames/<path:video_id>/<int:keyframe_index>')
def serve_frame_image(video_id, keyframe_index):
    """
    Serves the actual keyframe image file to the front-end.
    This allows the <img> tag to have a valid src URL.
    """
    try:
        filename = f"{keyframe_index:03d}.jpg" 
        return send_from_directory(
            os.path.join(config.KEYFRAMES_DIR, video_id), 
            filename
        )
    except FileNotFoundError:
        return send_from_directory('static', 'placeholder.png'), 404

@app.route('/videos/<path:video_id>')
def serve_video_file(video_id):
    """
    Phục vụ tệp video đầy đủ để phát lại trong modal.
    """
    try:
        # Giả sử video của bạn là tệp .mp4. Thay đổi phần mở rộng nếu cần.
        filename = f"{video_id}.mp4" 
        
        # Gửi tệp từ thư mục chứa video của bạn.
        # Đảm bảo bạn đã định nghĩa VIDEOS_DIR trong config.py
        return send_from_directory(
            config.VIDEOS_DIR, 
            filename,
            as_attachment=False # Quan trọng: Đảm bảo trình duyệt phát tệp thay vì tải xuống
        )
    except FileNotFoundError:
        return "Video not found", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)