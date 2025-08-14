import torch
import clip
import numpy as np
from pathlib import Path
import json
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from elasticsearch import Elasticsearch
import config 

class HybridVideoRetrievalSystem:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_clip_model()
        self._connect_to_milvus()
        self._connect_to_es()
        
        self._setup_milvus_collection()
        self._setup_es_index()

    def _load_clip_model(self):
        self.model, _ = clip.load(config.CLIP_MODEL, device=self.device)

    def _connect_to_milvus(self):
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        print("Kết nối Milvus thành công.")

    def _connect_to_es(self):
        self.es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")
        if not self.es.ping():
            raise ConnectionError("Không thể kết nối tới Elasticsearch.")
        print("Kết nối Elasticsearch thành công.")

    def _setup_milvus_collection(self):
        """
        Thiết lập Collection trong Milvus:
        - Định nghĩa schema.
        - Tạo collection nếu chưa tồn tại.
        - Nạp dữ liệu từ các file .npy vào collection.
        - Tạo index cho việc tìm kiếm.
        """
        # Nếu collection đã tồn tại, xóa đi để làm lại từ đầu (tốt cho việc demo)
        if utility.has_collection(config.COLLECTION_NAME):
            print(f"Collection '{config.COLLECTION_NAME}' đã tồn tại. Xóa đi để tạo mới...")
            utility.drop_collection(config.COLLECTION_NAME)

        # 1. Định nghĩa Schema cho collection
        # Mỗi bản ghi sẽ là một keyframe
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="keyframe_index", dtype=DataType.INT64),
            FieldSchema(name="keyframe_vector", dtype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIMENSION)
        ]
        schema = CollectionSchema(fields, "Collection chứa các vector đặc trưng của keyframe")
        
        # 2. Tạo Collection
        print(f"Đang tạo collection '{config.COLLECTION_NAME}'...")
        self.vector_keyframes = Collection(config.COLLECTION_NAME, schema)
        
        # 3. Nạp dữ liệu (Ingest Data)
        self._ingest_data()
        
        # 4. Tạo Index cho vector field để tăng tốc tìm kiếm
        print("Đang tạo index cho collection...")
        index_params = {
            "metric_type": "L2",  # Khoảng cách Euclidean, phù hợp với vector chưa chuẩn hóa
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128} # Số cụm để phân chia vector
        }
        self.vector_keyframes.create_index(
            field_name="keyframe_vector",
            index_params=index_params
        )
        
        # 5. Tải collection vào bộ nhớ để sẵn sàng tìm kiếm
        print("Đang tải collection vào bộ nhớ...")
        self.vector_keyframes.load()
        print("Hệ thống đã sẵn sàng!")
        
    def _ingest_data(self):
        """Đọc dữ liệu từ các file .npy và chèn vào Milvus."""
        print("Bắt đầu nạp dữ liệu vào Milvus...")        
        total_vectors_inserted = 0
        for npy_file in Path(config.CLIP_FEATURES_DIR).glob("*.npy"):
            video_id = npy_file.stem
            vectors = np.load(npy_file).astype(np.float32)
            entities = [
                [video_id] * len(vectors),  # List of video_id
                list(range(len(vectors))),  # List of keyframe_index
                vectors                     # List of vectors
            ]
            
            self.vector_keyframes.insert(entities)
            total_vectors_inserted += len(vectors)
            
        self.vector_keyframes.flush() # Đẩy dữ liệu vào storage
        print(f"Nạp dữ liệu hoàn tất. Đã chèn {total_vectors_inserted} vector.")

    def _setup_es_index(self):
        """Đọc file metadata và nạp vào Elasticsearch."""
        if self.es.indices.exists(index=config.ES_INDEX_NAME):
            print(f"Index '{config.ES_INDEX_NAME}' đã tồn tại. Xóa đi để tạo mới...")
            self.es.indices.delete(index=config.ES_INDEX_NAME)
            
        print(f"Đang tạo và nạp dữ liệu cho index '{config.ES_INDEX_NAME}'...")
        for metadata_file in Path(config.METADATA_DIR).glob("*.json"):
            video_id = metadata_file.stem
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.es.index(
                index=config.ES_INDEX_NAME,
                id=video_id,
                document=metadata
            )
        print("Nạp dữ liệu vào Elasticsearch hoàn tất.")

    def _search_milvus(self, query_vector, limit=20):
        """Thực hiện tìm kiếm vector trên Milvus."""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.vector_keyframes.search(
            data=query_vector,
            anns_field="keyframe_vector",
            param=search_params,
            limit=limit,
            output_fields=["video_id", "keyframe_index"]
        )
        
        # Trả về một dict {(video_id, keyframe_index): score}
        scores = {}
        for hit in results[0]:
            vid = hit.entity.get('video_id')
            frame = hit.entity.get('keyframe_index')
            
            # Khởi tạo dict cho video_id nếu chưa có
            if vid not in scores:
                scores[vid] = {}
            
            scores[vid][frame] = hit.distance
        return scores

    def _search_es(self, text_query, limit=20):
        """Thực hiện tìm kiếm từ khóa trên Elasticsearch."""
        resp = self.es.search(
            index=config.ES_INDEX_NAME,
            size=limit,
            query={
                "multi_match": {
                    "query": text_query,
                    "fields": ["title^2", "description", "keywords^1.5"] # Ưu tiên tiêu đề
                }
            }
        )
        # Trả về một dict {video_id: score}
        return {hit['_id']: hit['_score'] for hit in resp['hits']['hits']}

    def _fuse_results(self, milvus_results, es_results, k=60):
        """
        Kết hợp kết quả từ 2 hệ thống bằng kỹ thuật Reciprocal Rank Fusion (RRF).
        """
        fused_scores = {}
        
        # --- Xử lý kết quả Milvus ---
        # 1. Tìm điểm tốt nhất (distance nhỏ nhất) cho mỗi video_id từ kết quả của Milvus
        milvus_best_scores = {
            video_id: min(keyframe_scores.values())
            for video_id, keyframe_scores in milvus_results.items()
        }

        # 2. Lấy danh sách kết quả đã được sắp xếp
        sorted_milvus = sorted(milvus_best_scores.items(), key=lambda item: item[1]) # Sắp xếp theo distance tăng dần
        sorted_es = sorted(es_results.items(), key=lambda item: item[1], reverse=True) # Sắp xếp theo score giảm dần

        # Tính điểm RRF cho Milvus
        for rank, (video_id, _) in enumerate(sorted_milvus):
            if video_id not in fused_scores:
                fused_scores[video_id] = 0
            fused_scores[video_id] += 1 / (k + rank)
            
        # Tính điểm RRF cho Elasticsearch
        for rank, (video_id, _) in enumerate(sorted_es):
            if video_id not in fused_scores:
                fused_scores[video_id] = 0
            fused_scores[video_id] += 1 / (k + rank)

        # Sắp xếp lại theo điểm RRF tổng
        reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return reranked_results
    
    def encode_text(self, text_query: str):
        with torch.no_grad():
            text_tokens = clip.tokenize([text_query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        return text_features.float().cpu().numpy()

    def search(self, text_query: str, top_k: int = 5):
        print(f"\n--- Bắt đầu Hybrid Search cho: '{text_query}' ---")
        
        # 1. Mã hóa truy vấn
        query_vector = self.encode_text(text_query)
        
        # 2. Tìm kiếm song song
        milvus_results = self._search_milvus(query_vector)
        es_results = self._search_es(text_query)
        
        # 3. Kết hợp và xếp hạng lại
        fused_results = self._fuse_results(milvus_results, es_results)
        
        # 4. In kết quả
        print("--- KẾT QUẢ TÌM KIẾM KẾT HỢP ---")
        final_results = []
        for video_id, rrf_score in fused_results[:top_k]:
            es_score = es_results.get(video_id)
            milvus_video_scores = milvus_results.get(video_id)
            
            result_details = {
                "video_id": video_id,
                "rrf_score": rrf_score,
                "es_score": es_score,
                "milvus_best_frame": None,
                "milvus_best_distance": None,
                "frame_scores": milvus_video_scores
            }

            # Tìm keyframe tốt nhất từ kết quả của Milvus cho video này
            if milvus_video_scores:
                best_frame, best_distance = min(milvus_video_scores.items(), key=lambda item: item[1])
                result_details["milvus_best_frame"] = best_frame
                result_details["milvus_best_distance"] = best_distance

            final_results.append(result_details)

            # In ra thông tin
            print(f"\tVideo ID: {video_id}")
            print(f"\tĐiểm kết hợp (RRF): {result_details['rrf_score']:.4f}")
            
            es_info = f"ES score: {result_details['es_score']:.4f}" if result_details['es_score'] is not None else "ES score: N/A"
            
            if result_details['milvus_best_distance'] is not None:
                milvus_info = f"Milvus: frame {result_details['milvus_best_frame']} @ dist {result_details['milvus_best_distance']:.4f}"
            else:
                milvus_info = "Milvus: N/A"
                
            print(f"\t({milvus_info} | {es_info})")

        return final_results

if __name__ == "__main__":
    try:
        system = HybridVideoRetrievalSystem()
        system.search("một phương tiện giao thông trên đường")
    finally:
        connections.disconnect("default")
        print("\nĐã ngắt kết nối Milvus.")