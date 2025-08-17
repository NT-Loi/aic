import torch
from multilingual_clip import pt_multilingual_clip
import transformers
import logging
import config

logger = logging.getLogger(__name__)

class TextEncoder:
    def __init__(self, device: str = 'cuda'):
        self.device = device
        logger.info(f"Loading multilingual model '{config.MODEL_NAME}' to device '{self.device}'...")
        self.model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(config.MODEL_NAME)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        logger.info("TextEncoder initialized successfully.")

    def encode(self, text_query: str):
        with torch.no_grad():
            text_features = self.model.forward([text_query], self.tokenizer)
        return text_features.float().cpu().numpy()