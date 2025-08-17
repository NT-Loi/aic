import logging
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch
import torch
from PIL import Image, UnidentifiedImageError
import os

import config
from utils.text_encoder import TextEncoder
from utils.ranker import rrf_ranker, CrossModalReRanker
from retrievers import milvus_retriever, es_retriever

# --- Setup Logging ---
# log_file = "system.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),
#     ]
# )
logger = logging.getLogger(__name__)

class HybridVideoRetrievalSystem:
    def __init__(self, re_ingest=False):
        if re_ingest:
            from ingest_data import main
            main()

        logger.info("Initializing Hybrid Video Retrieval System...")

        # Initialize connections
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        logger.info("Successfully connected to Milvus.")

        self.es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}", timeout=30, retry_on_timeout=True, max_retries=3)
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch.")
        logger.info("Successfully connected to Elasticsearch.")
        
        # Load Milvus collections
        self.keyframes_collection = Collection(config.KEYFRAME_COLLECTION_NAME)
        self.keyframes_collection.load()
        
        # Initialize the text encoder and reranker
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = TextEncoder(device=self.device)
        self.reranker = CrossModalReRanker(device=self.device)

    def _load_keyframe_image(self, video_id: str, keyframe_index: int):
        """
        Loads a single keyframe image from disk as a PIL Image.
        This function is hardened to only return a valid Image object or None.
        """
        try:
            filename = f"{keyframe_index:04d}.jpg"
            image_path = os.path.join(config.KEYFRAMES_DIR, video_id, filename)
            
            # Check if file exists before trying to open it
            if not os.path.exists(image_path):
                # logger.warning(f"Image file not found: {image_path}")
                return None

            img = Image.open(image_path)
            
            # CRITICAL: Ensure the image is in RGB format. Some models fail on
            # single-channel (grayscale) or RGBA images. This standardizes it.
            if img.mode != 'RGB':
                img = img.convert('RGB')

            return img
        
        except FileNotFoundError:
            # This is redundant if we use os.path.exists, but good to have
            return None
        except UnidentifiedImageError:
            # This handles cases where the file exists but is corrupted or not an image
            logger.warning(f"Could not identify image file (corrupted?): {image_path}")
            return None
        except Exception as e:
            # Catch any other unexpected errors during image loading
            logger.error(f"Unexpected error loading image {image_path}: {e}")
            return None

    def search(self, query_data: dict, top_k: int = 20):
        """
        Performs a multi-stage search:
        1. Retrieval: Hybrid search (Milvus + ES) to get candidates.
        2. Re-ranking: Cross-Encoder model to re-order the top candidates.
        """
        logger.info(f"--- 💠 Starting search with data: {query_data} ---")
        
        query = query_data.get("query", "")
        object_list = query_data.get("objects")
        text = query_data.get("text", "")
        metadata = query_data.get("metadata", "")

        if not query:
            object_query = ""
            if object_list:
                temp = [str(label) + str(count) for label, count in object_list]
                object_query = " ".join(temp)
            query = ' '.join(filter(None, [object_query, text, metadata]))

        if not query:
            logger.warning("Search initiated with no query data.")
            return []

        logger.info("1/3: Searching...")
        query_vector = self.encoder.encode(query)
        vector_scores = milvus_retriever.search_keyframes(self.keyframes_collection, query_vector)
        meta_scores = es_retriever.search_metadata(self.es, metadata)
        content_scores = es_retriever.search_keyframes(self.es, text, object_list)

        logger.info("2/3: Fusing retrieval results...")
        ranked_vector_scores = sorted(vector_scores.items(), key=lambda item: item[1])
        ranked_content_scores = sorted(content_scores.items(), key=lambda item: item[1], reverse=True)
        candidate_frames_set = set(vector_scores.keys()) | set(content_scores.keys())
        meta_propagated = {frame: meta_scores.get(frame[0], 0) for frame in candidate_frames_set}
        ranked_meta_scores = sorted(meta_propagated.items(), key=lambda item: item[1], reverse=True)
        ranked_fused_scores = rrf_ranker([ranked_vector_scores, ranked_content_scores, ranked_meta_scores])
        
        NUM_CANDIDATES_TO_RERANK = top_k * 5
        candidates_for_reranking = [key for key, score in ranked_fused_scores[:NUM_CANDIDATES_TO_RERANK]]
        
        logger.info(f"3/3: Re-ranking top {len(candidates_for_reranking)} candidates...")
        reranked_results = self.reranker.rerank(
            text_query=query,
            candidate_frames=candidates_for_reranking,
            image_loader_func=self._load_keyframe_image 
        )

        fused_scores = {key: score for key, score in ranked_fused_scores}

        results = []
        # reranked_results is a sorted list of [((vid, idx), rerank_score), ...]
        for (video_id, keyframe_index), rerank_score in reranked_results[:top_k]:
            key = (video_id, keyframe_index)
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_index,
                "vector_score": vector_scores.get(key),
                "content_score": content_scores.get(key),
                "metadata_score": meta_scores.get(video_id),
                "rrf_score": fused_scores.get(key),
                "rerank_score": rerank_score
            })
            
        logger.info(f"Search complete. {results}")
        return results