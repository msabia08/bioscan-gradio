import gradio as gr
import torch
import numpy as np
import h5py
import faiss
from PIL import Image
import io 
import pickle



def searchEmbeddings(id):
    # variable and index initialization
    dim = 768
    count = 0
    num_neighbors = 10

    image_index = faiss.IndexFlatIP(dim)

    # get index
    image_index = faiss.read_index("big_image_index.index")

    # search for query
    query = id_to_emb_dict[id]
    query = query.astype(np.float32)
    D, I = image_index.search(query, num_neighbors)

    id_list = []
    i = 1
    for indx in I[0]:
        id = indx_to_id_dict[indx]
        id_list.append(id)
        
    return id_list

with gr.Blocks() as demo:

    with open("dataset_processid_list.pickle", "rb") as f:
        dataset_processid_list = pickle.load(f)
    with open("dataset_image_mask.pickle", "rb") as f:
        dataset_image_mask = pickle.load(f)
    with open("processid_to_index.pickle", "rb") as f:
        processid_to_index = pickle.load(f)
    with open("big_id_to_emb_dict.pickle", "rb") as f:
        id_to_emb_dict = pickle.load(f)
    with open("big_indx_to_id_dict.pickle", "rb") as f:
        indx_to_id_dict = pickle.load(f)

    with gr.Column():
        process_id = gr.Textbox(label="ID:", info="Enter a sample ID to search for")
        process_id_list = gr.Textbox(label="Closest 10 matches:" )
        search_btn = gr.Button("Search") 

    search_btn.click(fn=searchEmbeddings, inputs=process_id, 
                     outputs=[process_id_list])
    
    

# ARONZ671-20
demo.launch()