import streamlit as st
from pathlib import Path
import json
import config
from retrieval_system import HybridVideoRetrievalSystem

def find_video_file(video_id: str):
    """Searches for a video file with common extensions."""
    video_dir = Path(config.VIDEOS_DIR)
    for ext in ["mp4", "webm", "mkv", "avi"]:
        video_path = video_dir / f"{video_id}.{ext}"
        if video_path.exists():
            return video_path
    return None

def parse_structured_query(query_text: str) -> dict:
    """Parses the multi-line text input into a dictionary for the search function."""
    parsed = {}
    lines = query_text.strip().split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            # Map UI keys to backend keys if they differ
            if key in ["query", "objects", "text", "metadata"]:
                if key == "objects":
                    # Parse object counts
                    object_pairs = value.split(',')
                    object_list = []
                    for pair in object_pairs:
                        if ':' in pair:
                            obj, count = pair.split(':')
                            try:
                                object_list.append((obj.strip(),int(count.strip())))
                            except ValueError:
                                continue
                    value = object_list
                parsed[key] = value
    return parsed

# --- Thiết lập giao diện ---
st.set_page_config(page_title="AIC2025", layout="wide")

st.title("Hybrid Video Retrieval System 🚀")
st.write(
    "Nhập một câu mô tả để tìm kiếm video."
)

# --- Tải hệ thống (sử dụng cache của Streamlit) ---
# @st.cache_resource sẽ đảm bảo hệ thống chỉ được khởi tạo một lần duy nhất.
@st.cache_resource
def load_system():
    """
    Tải và khởi tạo hệ thống tìm kiếm.
    Đây là một tác vụ nặng nên sẽ được cache lại.
    """
    print("Đang khởi tạo HybridVideoRetrievalSystem...")
    system = HybridVideoRetrievalSystem()
    print("Hệ thống đã sẵn sàng.")
    return system

try:
    system = load_system()

    # --- Thanh tìm kiếm ---
    query_text = st.text_area(
        "Nội dung tìm kiếm:", 
        height=150,
        placeholder="Query: a man is walking\nObject: Person: 1, Car: 1\nText: exit sign\nMetadata: sports"
    )

    if st.button("Search") and query_text:
        # Parse the structured text from the text area into a dictionary
        query_data = parse_structured_query(query_text)

        # Check if all fields are empty
        if not any(query_data.values()):
            st.warning("Vui lòng nhập ít nhất một trường để tìm kiếm.")
        else:
            with st.spinner("🧠 Đang phân tích và tìm kiếm..."):
                # Pass the dictionary directly to the search method
                results = system.search(query_data, top_k=20)

            st.divider()
            st.subheader(f"Kết quả hàng đầu cho truy vấn của bạn")

            if not results:
                st.warning("Không tìm thấy video nào phù hợp với mô tả của bạn.")
            else:
                # --- Hiển thị kết quả ---
                for result in results:
                    main_col, popover_col = st.columns([4, 1])

                    with main_col:
                        metadata_path = Path(config.METADATA_DIR) / f"{result['video_id']}.json"
                        title = "Không có tiêu đề"
                        description = "Không có mô tả."
                        keywords = []
                        if metadata_path.exists():
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                                title = metadata.get('title', 'N/A')
                                description = metadata.get('description', 'Không có mô tả.')
                                keywords = metadata.get('keywords', [])

                        st.subheader(f"🎬 {title}")

                        # --- Hiển thị ảnh và thông tin chi tiết ---
                        best_frame_col, info_col = st.columns([1, 2])

                        with best_frame_col:
                            # Hiển thị keyframe tốt nhất
                            keyframe_index = result.get("keyframe_index")
                            if keyframe_index is not None:
                                keyframe_path = (
                                    Path(config.KEYFRAMES_DIR) / 
                                    result["video_id"] / 
                                    f"{keyframe_index:03d}.jpg"
                                )
                                if keyframe_path.exists():
                                    st.image(str(keyframe_path), caption=f"Frame phù hợp nhất: {keyframe_index}")
                                else:
                                    st.warning(f"Ảnh {keyframe_path.name} không tồn tại.")
                            else:
                                st.info("Không có keyframe từ Milvus.")
                        
                        with info_col:
                            # Hiển thị điểm và expander cho metadata
                            score_col1, score_col2, score_col3 = st.columns(3)
                            score_col1.metric(label="🏆 Điểm RRF", value=f"{result['rrf_score']:.4f}", help="Điểm kết hợp cuối cùng. Càng cao càng tốt.")
                            
                            if result.get('milvus_dist') is not None:
                                score_col2.metric(label="🖼️ Milvus Dist", value=f"{result['milvus_dist']:.4f}", help="Độ tương đồng vector. Càng thấp càng tốt.")
                            
                            if result.get('es_frame_score') is not None:
                                score_col3.metric(label="📝 ES Score", value=f"{result['es_frame_score']:.4f}", help="Độ liên quan văn bản. Càng cao càng tốt.")

                            with st.expander("Xem metadata"):
                                st.markdown(f"**Video ID:** `{result['video_id']}`")
                                st.markdown(f"**Mô tả:** {description}")
                                if keywords:
                                    st.markdown(f"**Keywords:** {', '.join(keywords)}")
                                else:
                                    st.markdown("**Keywords:** Không có.")

                    with popover_col:
                        # --- Nút bấm để mở subscreen (popover) ---
                        with st.popover("👁️ Watch Video"):
                            st.markdown("#### Video Player")
                            video_path = find_video_file(result['video_id'])
                            if video_path:
                                # Giả định keyframe index tương ứng với số giây
                                start_time = int(result.get('keyframe_index', 0))
                                st.video(str(video_path), start_time=start_time)

                            else:
                                st.warning(f"Không tìm thấy file video cho '{result['video_id']}'.")

                            st.divider()
                            
                            st.markdown("#### Keyframes liên quan nhất")
                            st.info("Detailed frame scores are not available in this view.")
                    st.divider()
except ConnectionError as e:
    st.error("Hãy chắc chắn rằng Docker (với Milvus và Elasticsearch) đang chạy và các đường dẫn trong config.py là chính xác.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi khởi tạo hệ thống: {e}")