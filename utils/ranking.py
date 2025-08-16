from collections import defaultdict
import config

def fuse_results_rrf(ranked_lists: list, k: int = config.RRF_K) -> list:
    fused_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, (item_id, _) in enumerate(ranked_list):
            fused_scores[item_id] += 1 / (k + rank + 1)

    # Sort results by the final RRF score
    reranked_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return reranked_results