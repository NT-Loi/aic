from pymilvus import Collection
import logging

logger = logging.getLogger(__name__)

def search_keyframes(collection: Collection, query_vector, limit=500) -> dict:
    """Searches the keyframe collection in Milvus."""
    logger.info("Searching Milvus keyframe collection...")
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    
    results = collection.search(
        data=query_vector,
        anns_field="keyframe_vector", 
        param=search_params,
        limit=limit,
        output_fields=["video_id", "keyframe_index"]
    )
    
    keyframe_scores = {}
    if results:
        for hit in results[0]:
            vid = hit.entity.get('video_id')
            frame_idx = hit.entity.get('keyframe_index')
            keyframe_scores[(vid, frame_idx)] = hit.distance
            
    logger.info(f"Found {len(results[0])} potential keyframes from Milvus.")
    return keyframe_scores