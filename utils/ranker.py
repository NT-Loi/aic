from collections import defaultdict
import config
import logging
from sentence_transformers import CrossEncoder
from sentence_transformers import util

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

    def rerank(self, text_query: str, candidate_frames: list, image_loader_func) -> dict:
        """
        Re-ranks a list of candidate frames against a text query using CLIP bi-encoder.
        
        Args:
            text_query (str): The search query.
            candidate_frames (list): List of (video_id, keyframe_index) tuples.
            image_loader_func (func): Function that returns a PIL.Image for a (video_id, keyframe_index).

        Returns:
            dict: Mapping of (video_id, keyframe_index) -> score
        """
        if not self.model or not candidate_frames:
            return {}

        logger.info(f"Re-ranking {len(candidate_frames)} candidates with CLIP bi-encoder...")

        # Clean the text query
        clean_query = str(text_query).strip()

        # Load all images
        loaded_frames_keys = []
        loaded_images = []

        for video_id, keyframe_index in candidate_frames:
            try:
                image = image_loader_func(video_id, keyframe_index)
                if image:
                    loaded_frames_keys.append((video_id, keyframe_index))
                    loaded_images.append(image)
            except Exception as e:
                logger.warning(f"Could not load image for {video_id}/{keyframe_index}: {e}")

        if not loaded_images:
            logger.warning("No images could be loaded for re-ranking.")
            return {}

        # Encode query and images
        query_emb = self.model.encode(clean_query, convert_to_tensor=True, show_progress_bar=False)
        image_embs = self.model.encode(loaded_images, convert_to_tensor=True, show_progress_bar=True)

        # Compute cosine similarity
        scores = util.cos_sim(query_emb, image_embs)[0].cpu().tolist()

        # Build the result dictionary
        reranked_scores = {key: score for key, score in zip(loaded_frames_keys, scores)}

        # Assign very low score to frames that failed to load
        for frame_key in candidate_frames:
            if frame_key not in reranked_scores:
                reranked_scores[frame_key] = -999.0

        return reranked_scores