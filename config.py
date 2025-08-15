# Necessary directories
DATA_ROOT_DIR = "data"
CLIP_FEATURES_DIR = "data/clip-features-32"
VIDEOS_DIR = "data/videos"
KEYFRAMES_DIR = "data/key_frames"
METADATA_DIR = "data/media-info"

# Models
# MODEL_NAME = "ViT-B/32"
MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"

# Milvus config
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "video_keyframes"
VECTOR_DIMENSION = 512

# Elasticsearch config
ES_HOST = "localhost"
ES_PORT = 9200
ES_INDEX_NAME = "video_metadata"