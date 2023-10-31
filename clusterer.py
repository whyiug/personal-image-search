import faiss
import numpy as np


class ImageIndexer:
    def __init__(self):
        self.index = None
        self.store_file = 'embed_data/faiss_clusters.index'

    def fit(self, image_embeds, num_clusters = None, m = 32):
        if type(image_embeds) != np.ndarray:
            image_embeds = np.array(image_embeds)

        # dim, measure = image_embeds.shape[1], faiss.METRIC_L2 
        # param = 'HNSW' 
        # self.index = faiss.index_factory(dim, param, measure)
        # self.index.train(image_embeds)
        # self.index.add(image_embeds)


        if num_clusters is None:
            num_clusters = min(round(np.sqrt(image_embeds.shape[0])), image_embeds.shape[0] // 39) # Emperically determined
        n_dims = image_embeds.shape[1]

        quantizer = faiss.IndexHNSWFlat(n_dims, m)
        self.index = faiss.IndexIVFFlat(quantizer, n_dims, num_clusters, faiss.METRIC_INNER_PRODUCT) #METRIC_INNER_PRODUCT: the higher this value, the better; #METRIC_L2: other way round
        self.index.train(image_embeds)
        self.index.add(image_embeds)

        faiss.write_index(self.index, self.store_file)

        return self.index

    def predict(self, embed, k):
        if self.index is None:
            self.index = faiss.read_index(self.store_file)

        D, I = self.index.search(embed, k)
        return D[0], I[0]


image_indexer = ImageIndexer()