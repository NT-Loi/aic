from elasticsearch import Elasticsearch
import logging
import config

logger = logging.getLogger(__name__)

def search_metadata(es_client: Elasticsearch, text_query: str, limit=500) -> dict:
    """
    Searches the metadata index in Elasticsearch.
    Handles empty queries by returning all documents with a neutral score.
    """
    # --- Input Validation ---
    if not (text_query and text_query.strip()):
        logger.info("No metadata query provided. Returning all videos with a neutral score.")
        query = {
            "match_all": {}
        }
    else:
        logger.info(f"Searching metadata index for: '{text_query}'")
        query = {
            "multi_match": {
                "query": text_query,
                "fields": ["title^2", "description", "keywords^1.5"]
            }
        }
    
    try:
        resp = es_client.search(
            index=config.METADATA_INDEX_NAME,
            size=limit,
            query=query,
            request_cache=True 
        )
        return {hit['_id']: hit['_score'] for hit in resp['hits']['hits']}

    except Exception as e:
        logger.error(f"Error searching metadata in Elasticsearch: {e}")
        return {}

def search_keyframes(es_client: Elasticsearch, text_query: str, objects: list, limit=1000) -> dict:
    """Searches the frames index in Elasticsearch for OCR text and detected objects."""
    logger.info(f"Searching ES frames with text='{text_query}' and objects={objects}")
    
    must_clauses = []
    should_clauses = []
    if objects:
        # 'objects' is a list of (label, count) tuples, e.g., [('Person', 2), ('Car', 1)]
        for obj_label, obj_count in objects:
            must_clauses.append({
            "nested": {
                "path": "detected_objects",
                "query": {
                    "bool": {
                        # --- The Hard Requirement ---
                        # The object MUST have the correct label.
                        "must": [
                            {
                                "match": {
                                    "detected_objects.label": obj_label
                                }
                            }
                        ],
                        
                        # --- The Conditional Score Booster ---
                        # This part is OPTIONAL but will increase the score if met.
                        # It will only be evaluated for nested documents that already matched the 'must' clause.
                        "should": [
                            {
                                "range": {
                                    "detected_objects.count": {
                                        "gte": obj_count,
                                        "boost": 1.5 # Apply a boost ONLY to this range condition
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        })
            
    # Only add the OCR text clause if the user provided text, and boost its importance.
    if text_query:
        should_clauses.append({
            "match": {
                "ocr_text": {
                    "query": text_query,
                    "boost": 2.0  # Adjust this value to control the weight
                }
            }
        })
            
    query = {
        "bool": {
            "must": must_clauses,
            "should": should_clauses
        }
    }

    if not must_clauses:
        if should_clauses:
            # If there are no objects, just search for the OCR text
            query = {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1
                }
            }
        else:
            # If there is no query at all
            query = {"match_all": {}}

    resp = es_client.search(index=config.ES_FRAMES_INDEX_NAME, size=limit, query=query, request_cache=True)
    
    frame_scores = {}
    for hit in resp['hits']['hits']:
        source = hit['_source']
        frame_scores[(source['video_id'], source['keyframe_index'])] = hit['_score']
    
    logger.info(f"Found {len(frame_scores)} frames from ES frames search.")
    return frame_scores