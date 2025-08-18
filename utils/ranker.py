from collections import defaultdict
import config
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

def rrf_ranker(ranked_lists: list, k: int = config.RRF_K) -> dict:
    fused_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list):
            fused_scores[item_id] += 1 / (k + rank + 1)

    return fused_scores

class CrossModalReRanker:
    """
    A re-ranker that uses a cross-encoder model to re-score candidates
    based on the fine-grained similarity between an image and a text query.
    """
    def __init__(self, model_name: str = 'sentence-transformers/clip-ViT-B-32-multilingual-v1', device: str = 'cuda'):
        logger.info(f"Loading Cross-Encoder model: {model_name} onto device: {device}")
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info("✅ Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"💥 Failed to load Cross-Encoder model: {e}")
            self.model = None

    def rerank(self, text_query: str, candidate_frames: list, image_loader_func) -> list:
        """
        Re-ranks a list of candidate frames against a text query.
        """
        if not self.model or not candidate_frames:
            return []

        logger.info(f"Re-ranking {len(candidate_frames)} candidates with the Cross-Encoder...")

        model_input_pairs = []
        loaded_frames_keys = [] # Track successfully loaded frames
        clean_query = str(text_query).strip()
        for video_id, keyframe_index in candidate_frames:
            try:
                image = image_loader_func(video_id, keyframe_index)
                if image:
                    model_input_pairs.append((clean_query, image))
                    loaded_frames_keys.append((video_id, keyframe_index))
            except Exception as e:
                logger.warning(f"Could not load image for {video_id}/{keyframe_index}: {e}")
        
        if not model_input_pairs:
            logger.warning("No images could be loaded for re-ranking.")
            return []
        # logger.info(f"{model_input_pairs}")
        scores = self.model.predict(model_input_pairs, show_progress_bar=True) # Turn on progress bar for long tasks

        # The score at scores[i] corresponds to the key at loaded_frames_keys[i]
        reranked_scores = {key: score for key, score in zip(loaded_frames_keys, scores)}

        # For frames that failed to load, we don't include them, or we could assign a low score
        # Let's add them back with a very low score so they don't get lost
        for frame_key in candidate_frames:
            if frame_key not in reranked_scores:
                reranked_scores[frame_key] = -999.0

        return reranked_scores