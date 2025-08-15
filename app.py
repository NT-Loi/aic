import streamlit as st
from pathlib import Path
from PIL import Image
import json
import config
from hybrid_retrieval import HybridVideoRetrievalSystem

def find_video_file(video_id: str):
    """Searches for a video file with common extensions."""
    video_dir = Path(config.VIDEOS_DIR)
    for ext in ["mp4", "webm", "mkv", "avi"]:
        video_path = video_dir / f"{video_id}.{ext}"
        if video_path.exists():
            return video_path
    return None

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
    system = HybridVideoRetrievalSystem(re_ingest=False)
    print("Hệ thống đã sẵn sàng.")
    return system

try:
    system = load_system()

    # --- Thanh tìm kiếm ---
    query = st.text_input("Nội dung tìm kiếm:", placeholder="ví dụ: một chiếc xe màu đỏ")

    if query:
        with st.spinner("🧠 Đang phân tích và tìm kiếm..."):
            results = system.search(query, top_k=100)

        st.divider()
        st.subheader(f"Kết quả hàng đầu cho: '{query}'")

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
                        if result["milvus_best_frame"] is not None:
                            keyframe_path = (
                                Path(config.KEYFRAMES_DIR) / 
                                result["video_id"] / 
                                f"{result['milvus_best_frame']:03d}.jpg"
                            )
                            if keyframe_path.exists():
                                with Image.open(keyframe_path) as img:
                                    st.image(img.copy(), caption=f"Frame phù hợp nhất: {result['milvus_best_frame']:03d}")
                            else:
                                st.warning(f"Ảnh {keyframe_path.name} không tồn tại.")
                        else:
                            st.info("Không có keyframe từ Milvus.")
                    
                    with info_col:
                        # Hiển thị điểm và expander cho metadata
                        score_col1, score_col2, score_col3 = st.columns(3)
                        score_col1.metric(label="🏆 Điểm RRF", value=f"{result['rrf_score']:.4f}", help="Điểm kết hợp cuối cùng. Càng cao càng tốt.")
                        
                        if result['milvus_best_distance'] is not None:
                            score_col2.metric(label="🖼️ Milvus Dist", value=f"{result['milvus_best_distance']:.4f}", help="Độ tương đồng vector. Càng thấp càng tốt.")
                        
                        if result['es_score'] is not None:
                            score_col3.metric(label="📝 ES Score", value=f"{result['es_score']:.4f}", help="Độ liên quan văn bản. Càng cao càng tốt.")

                        with st.expander("Xem metadata"):
                            st.markdown(f"**Video ID:** `{result['video_id']}`")
                            st.markdown(f"**Mô tả:** {description}")
                            if keywords:
                                st.markdown(f"**Keywords:** {keywords}")
                            else:
                                st.markdown("**Keywords:** Không có.")

                with popover_col:
                    # --- Nút bấm để mở subscreen (popover) ---
                    with st.popover("👁️ Watch Video"):
                        st.markdown("#### Video Player")
                        video_path = find_video_file(result['video_id'])
                        if video_path:
                            # Giả định keyframe index tương ứng với số giây
                            best_frame = result.get('milvus_best_frame')
                            start_time = int(best_frame) if best_frame is not None else 0
                            # st.video(str(video_path), start_time=start_time)
                            with open(str(video_path), "rb") as f:
                                video_bytes = f.read()
                                st.video(video_bytes, start_time=start_time)
                        else:
                            st.warning(f"Không tìm thấy file video cho '{result['video_id']}'.")

                        st.divider()
                        
                        # --- Hiển thị danh sách keyframes và điểm số ---
                        st.markdown("#### Keyframes liên quan nhất")
                        frame_scores = result.get("frame_scores")
                        if frame_scores:
                            sorted_frames = sorted(frame_scores.items(), key=lambda item: item[1])
                            
                            num_frames_to_show = min(len(sorted_frames), 9)
                            num_cols = 3 
                            
                            for i in range(0, num_frames_to_show, num_cols):
                                cols = st.columns(num_cols)
                                row_frames = sorted_frames[i:i+num_cols]
                                
                                for j, (frame_index, distance) in enumerate(row_frames):
                                    with cols[j]:
                                        keyframe_path = Path(config.KEYFRAMES_DIR) / result["video_id"] / f"{frame_index:03d}.jpg"
                                        if keyframe_path.exists():
                                            with Image.open(keyframe_path) as img:
                                                st.image(img.copy())
                                            st.caption(f"Frame {frame_index} | Dist: {distance:.3f}")
                                        else:
                                            st.warning(f"Ảnh {frame_index:03d}.jpg không tồn tại.")
                        else:
                            st.info("Không có dữ liệu keyframe từ Milvus.")
                st.divider()

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi khởi tạo hệ thống: {e}")
    st.info("Hãy chắc chắn rằng Docker (với Milvus và Elasticsearch) đang chạy và các đường dẫn trong config.py là chính xác.")