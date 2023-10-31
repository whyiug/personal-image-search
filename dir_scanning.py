
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from clip import clip_searcher
from clusterer import image_indexer
from PIL import Image


# path = '/Users/forrest/blue/code/github/personal-image-search/images'

def walk_directory(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')) and not f.startswith('.'):
                fullname = os.path.join(root, f)
                yield fullname


def scan_directory(path):
    df = pd.DataFrame(columns=['image_path'])
    df_image_embeds = []
    i = 0
    image_paths = []
    for img in walk_directory(path):
        image_paths.append(img)
    
    print("## Length of image: ", len(image_paths))
    for i, v in tqdm(enumerate(image_paths), total=len(image_paths)):
        df.loc[i, 'image_path'] = v
        df_image_embeds.append(clip_searcher.get_image_features(Image.open(v)).flatten())
        i += 1

    df.to_csv('embed_data/df.csv',  sep='\t')
    image_indexer.fit(df_image_embeds)

    return df, df_image_embeds

def compute_similarity(embeds: str, df: pd.DataFrame, top_k: int = 5):
    score, ids = image_indexer.predict(embeds, top_k)

    ids = [id for id in ids if id != -1]
    score = score[:len(ids)]

    df = df.loc[ids]
    df['score'] = score

    df = df.sort_values(by='score', ascending=False)
    # df = df.sort_values(by='score', ascending=True)
    
    return df.reset_index()


def get_top_k_text_similarities(text: str, df: pd.DataFrame, top_k: int = 5):

    text_embeds = clip_searcher.get_text_features(text)
    
    return compute_similarity(text_embeds, df, top_k)

def get_top_k_image_similarities(image, df: pd.DataFrame, top_k: int = 5):

    if type(image) == str:
        image = Image.open(image)

    image_embeds = clip_searcher.get_image_features(image)
    
    return compute_similarity(image_embeds, df, top_k)
        

# scan_directory(path)

# df = pd.read_csv('df.csv', sep='\t')
# df_image_embeds = np.load('embed_data/df_image_embeds.npy')
# df_image_embeds = [x.flatten() for x in df_image_embeds]


# top_k_df = get_top_k_text_similarities("guy with smartphone in hand", df, df_image_embeds)[['image_path', 'cos_sim']]
# top_k_df.to_csv('top_k.csv',  sep='\t')

# top_k_df = get_top_k_image_similarities("D:\\Code\Diffusion_Images\\2023-05-11 17-43-18.281968.jpg", df, df_image_embeds)[['image_path', 'cos_sim']]
# top_k_df.to_csv('top_k.csv',  sep='\t')