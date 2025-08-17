import logging
import json
from pathlib import Path
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from collections import Counter
import config

logger = logging.getLogger(__name__)

def setup_milvus_collection(collection_name, schema, index_field, index_params):
    if utility.has_collection(collection_name):
        logger.warning(f"Collection '{collection_name}' already exists. Dropping.")
        utility.drop_collection(collection_name)
    
    collection = Collection(collection_name, schema)
    logger.info(f"Collection '{collection_name}' created.")
    
    logger.info(f"Creating index for field '{index_field}'...")
    collection.create_index(field_name=index_field, index_params=index_params)
    collection.flush()
    logger.info("Index created and data flushed.")
    return collection

def ingest_keyframe_data(collection: Collection):
    logger.info("Ingesting keyframe data into Milvus...")
    for npy_file in Path(config.CLIP_FEATURES_DIR).glob("*.npy"):
        video_id = npy_file.stem
        vectors = np.load(npy_file).astype(np.float32)
        entities = [[video_id] * len(vectors), list(range(len(vectors))), vectors]
        collection.insert(entities)
    collection.flush()
    logger.info("Keyframe data ingestion complete.")

def setup_es_index(es_client, index_name, mappings=None, actions_generator=None):
    if es_client.indices.exists(index=index_name):
        logger.warning(f"Index '{index_name}' already exists. Dropping.")
        es_client.indices.delete(index=index_name)
    
    if mappings:
        es_client.indices.create(index=index_name, body={"mappings": mappings})
    else:
        es_client.indices.create(index=index_name)
    logger.info(f"Index '{index_name}' created.")

    if actions_generator:
        logger.info(f"Ingesting data into '{index_name}'...")
        bulk(es_client, actions_generator())
        logger.info("Data ingestion complete.")

def generate_metadata_actions():
    for metadata_file in Path(config.METADATA_DIR).glob("*.json"):
        video_id = metadata_file.stem
        with open(metadata_file, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        yield {"_index": config.METADATA_INDEX_NAME, "_id": video_id, "_source": doc}

def load_json(path):
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def load_od_data(video_dir: Path, threshold: float = 0.5) -> dict:
    """
    Loads object detection JSON files, filters by score, and counts object occurrences.

    Returns:
        dict: A dictionary mapping frame indices to a dictionary of object counts.
              e.g., {"001": {"Person": 2, "Car": 1}}
    """
    all_frames_data = {}
    if not video_dir.exists():
        logger.warning(f"Object detection directory not found: {video_dir}")
        return all_frames_data

    for json_file in video_dir.glob("*.json"):
        try:
            frame_idx = json_file.stem
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            scores = data.get("detection_scores", [])
            entities = data.get("detection_class_entities", [])

            if len(scores) != len(entities):
                logger.warning(f"Mismatch between scores and entities in {json_file}. Skipping.")
                continue

            filtered_entities = []
            for score_str, entity in zip(scores, entities):
                try:
                    if float(score_str) > threshold:
                        filtered_entities.append(entity)
                except (ValueError, TypeError):
                    continue
            
            if filtered_entities:
                object_counts = Counter(filtered_entities)
                all_frames_data[frame_idx] = dict(object_counts)

        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {json_file}")
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            
    return all_frames_data

def generate_frames_actions():
    all_video_ids = {p.stem for p in Path(config.METADATA_DIR).glob("*.json")}

    for video_id in all_video_ids:
        ocr_data = load_json(Path(config.OCR_DIR) / f"{video_id}.json")

        obj_dir = Path(config.OBJECT_DETECTION_DIR) / video_id
        obj_data = load_od_data(obj_dir, threshold=0.5)
        
        all_frame_indices = set(ocr_data.keys()) | set(obj_data.keys())

        for frame_idx_str in all_frame_indices:
            frame_idx = int(frame_idx_str)
            
            # Get the dictionary of object counts for the frame. Default to an empty dict.
            object_counts = obj_data.get(frame_idx_str, {})

            # Directly create the nested structure for Elasticsearch from the counts.
            nested_objects = [{"label": label, "count": count} for label, count in object_counts.items()]

            doc = {
                "video_id": video_id,
                "keyframe_index": frame_idx,
                "ocr_text": ocr_data.get(frame_idx_str, ""),
                "detected_objects": nested_objects
            }
            yield {"_index": config.ES_FRAMES_INDEX_NAME, "_id": f"{video_id}_{frame_idx}", "_source": doc}

def main():
    # Connect to services
    connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    es = Elasticsearch(f"http://{config.ES_HOST}:{config.ES_PORT}")

    # --- Milvus Ingestion ---
    kf_fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="keyframe_index", dtype=DataType.INT64),
        FieldSchema(name="keyframe_vector", dtype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIMENSION)
    ]
    kf_schema = CollectionSchema(kf_fields, "Keyframe vectors")
    kf_index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    
    kf_collection = setup_milvus_collection(config.KEYFRAME_COLLECTION_NAME, kf_schema, "keyframe_vector", kf_index_params)
    ingest_keyframe_data(kf_collection)

    # --- Elasticsearch Ingestion ---
    setup_es_index(es, config.METADATA_INDEX_NAME, actions_generator=generate_metadata_actions)
    
    frames_mappings = {
                "properties": {
                    "video_id": {"type": "keyword"},
                    "keyframe_index": {"type": "integer"},
                    "ocr_text": {"type": "text"},
                    "detected_objects": {
                        "type": "nested",
                        "properties": {
                            "label": {"type": "keyword"},
                            "count": {"type": "integer"}
                        }
                    }
                }
            }
    setup_es_index(es, config.ES_FRAMES_INDEX_NAME, mappings=frames_mappings, actions_generator=generate_frames_actions)

    logger.info("--- DATA INGESTION COMPLETE ---")