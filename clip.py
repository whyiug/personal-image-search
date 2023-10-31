import torch
import numpy as np
import os
from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# your models cache will be stored here

# base_path = "/Users/forrest/blue/code/github/personal-image-search"
# os.environ['HUGGINGFACE_HUB_CACHE'] = base_path + '/model_cache'

model_id = "OFA-Sys/chinese-clip-vit-base-patch16"

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPSearcher:
    def __init__(self, model_id, device):
        self.model: ChineseCLIPModel = ChineseCLIPModel.from_pretrained(model_id).to(device)
        self.processor: ChineseCLIPProcessor = ChineseCLIPProcessor.from_pretrained(model_id)

    def get_text_features(self, text):
        inputs = self.processor(text=text, return_tensors = "pt").to(device)
        return self.model.get_text_features(**inputs).cpu().detach().numpy()

    def get_image_features(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        return self.model.get_image_features(**inputs).cpu().detach().numpy()


clip_searcher = CLIPSearcher(model_id, device)
