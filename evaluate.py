import models
import numpy as np
import models

class Evaluate:
    def __init__(self, model, device, embeddings, targets, threshold=0.8):
        self.model = model
        self.device = device
        self.embeddings = embeddings
        self.targets = targets
        self.threshold = threshold
        
    def evaluate(self, images):
        embs = []
        for image in images:
            embs.append(models.get_embedding(self.model, image, self.device))
        v = models.average_embeddings(embs)
        cosine = []
        
        for emb in self.embeddings:
            cosine.append(models.cosine(v, emb))
        if len(cosine) == 0:
            return (False, "UNK")
        idx = np.argmax(cosine)
        if cosine[idx] < self.threshold:
            return (False, "UNK")
        
        return cosine[idx], self.targets[idx]