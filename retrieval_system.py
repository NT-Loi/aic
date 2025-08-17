import logging
from pymilvus import connections, Collection
from elasticsearch import Elasticsearch

import config
from utils.text_encoder import TextEncoder
from utils.ranking import fuse_results_rrf
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
        
        # Initialize the text encoder
        self.encoder = TextEncoder()

    def search(self, query_data: dict, top_k: int = 20):
        """
        Performs a hybrid search using a dictionary of query components from the UI.
        """
        logger.info(f"--- Starting search with data: {query_data} ---")
        
        query = query_data.get("query", "")
        object_list = query_data.get("objects")
        text = query_data.get("text", "")
        metadata = query_data.get("metadata", "")

        # Determine the text to use for vector encoding
        vector_query_text = query

        if not vector_query_text:
            object_query = ""
            if object_list:
                # Extract just the labels from the list of tuples
                temp = [str(label) + str(count) for label, count in object_list]
                object_query = " ".join(temp)
            vector_query_text = ' '.join(filter(None, [object_query, text, metadata]))

        if not vector_query_text:
            logger.warning("Search initiated with no query data.")
            return []

        logger.info("1/4: Encoding query...")
        query_vector = self.encoder.encode(vector_query_text)
        
        logger.info("2/4: Searching vector database...")
        vector_scores = milvus_retriever.search_keyframes(self.keyframes_collection, query_vector)
        
        logger.info("3/4: Searching frame/metadata indices...")
        meta_scores = es_retriever.search_metadata(self.es, metadata or query)
        content_scores = es_retriever.search_keyframes(self.es, text, object_list)

        logger.info("4/4: Fusing results...")
        # List 1: Milvus Keyframes (convert to frame-level scores)
        ranked_vector_scores = sorted(vector_scores.items(), key=lambda item: item[1])

        # List 2: ES Frames
        ranked_content_scores = sorted(content_scores.items(), key=lambda item: item[1], reverse=True)

        # List 3: ES Metadata (propagated to all frames)
        candidate_frames = set(vector_scores.keys()) | set(content_scores.keys())
        meta_propagated = {frame: meta_scores.get(frame[0], 0) for frame in candidate_frames}
        ranked_meta_scores = sorted(meta_propagated.items(), key=lambda item: item[1], reverse=True)

        final_reranked_frames = fuse_results_rrf([ranked_vector_scores, ranked_content_scores, ranked_meta_scores])

        # Format and return results
        results = []
        for (video_id, keyframe_index), rrf_score in final_reranked_frames[:top_k]:
            results.append({
                "video_id": video_id,
                "keyframe_index": keyframe_index,
                "rrf_score": rrf_score,
                "vector_score": vector_scores.get((video_id, keyframe_index)),
                "content_score": content_scores.get((video_id, keyframe_index)),
                "metadata_score": meta_scores.get(video_id)
            })
            
        logger.info(f"Search complete. Returning {len(results)} results.")
        return results