import gradio as gr
import pandas as pd
import numpy as np
import os

from PIL import Image
from dir_scanning import get_top_k_text_similarities, get_top_k_image_similarities, scan_directory


if not os.path.exists('embed_data'):
    os.mkdir('embed_data')
  
try:
    df = pd.read_csv('embed_data/df.csv', sep='\t')
except:
    df = None

def search_by_text(text, top_k):
    if df is None:
        return

    top_k_df = get_top_k_text_similarities(text, df, top_k)

    images = []
    for i, row in top_k_df.iterrows():
        images.append(Image.open(row['image_path']))

    return list(zip(images, top_k_df.image_path.to_numpy()))

def search_by_image(image, top_k):
    if df is None:
        return

    top_k_df = get_top_k_image_similarities(image, df, top_k)

    images = []
    for i, row in top_k_df.iterrows():
        images.append(Image.open(row['image_path']))

    return list(zip(images, top_k_df.image_path.to_numpy()))

def scan_dir(path):
    if path is None or not os.path.exists(path):
        return

    scan_directory(path)

    global df

    df = pd.read_csv('embed_data/df.csv', sep='\t')


with gr.Blocks() as webui:
    gr.Markdown("CLIP Searcher")

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label = "Text", info = "Text to search")

            with gr.Row():
                image = gr.Image(label = "Image", info = "Image to search")

                with gr.Column():
                    top_k_slider = gr.Slider(label="Top K", minimum=1, maximum=50, step=1, value=5, info = "Top K closest results to the query")
                    search_by_text_btn = gr.Button("Search by text")
                    search_by_image_btn = gr.Button("Search by image")

            path = gr.Textbox(label = "Path", info = "Path with images to scan")
            scan_dir_btn = gr.Button("Scan directory", variant="primary")

  
        gallery = gr.Gallery(label = "Gallery", show_label=False).style(columns=3, rows = 2, height="auto", preview = False)

    search_by_text_btn.click(
        search_by_text, 
        inputs = [text, top_k_slider], 
        outputs = gallery
    )

    search_by_image_btn.click(
        search_by_image,
        inputs = [image, top_k_slider],
        outputs = gallery
    )

    scan_dir_btn.click(
        scan_dir,
        inputs = [path],
        outputs = path
    )

webui.queue()
webui.launch(share=True)