import gradio as gr
import torch
import numpy as np
import h5py
import faiss
from PIL import Image
import io 
import pickle

# add random index retreival


def get_image(file, dataset_image_mask, processid_to_index, idx):
    image_enc_padded = file[idx].astype(np.uint8)
    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

def searchEmbeddings(id, mod1, mod2):
    # variable and index initialization
    dim = 768
    num_neighbors = 10

    # get index
    index = faiss.IndexFlatIP(dim)
    if (mod2 == "Image"):
        index = faiss.read_index("image_index.index")
    elif (mod2 == "DNA"):
        index = faiss.read_index("dna_index.index")


    # search index
    if (mod1 == "Image"):
        query = id_to_image_emb_dict[id]
    elif(mod1 == "DNA"):
        query = id_to_dna_emb_dict[id]
    query = query.astype(np.float32)
    D, I = index.search(query, num_neighbors)

    id_list = []
    for indx in I[0]:
        id = indx_to_id_dict[indx]
        id_list.append(id)
    
    image1 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][0])
    image2 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][1])
    image3 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][2])
    image4 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][3])
    image5 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][4])
    image6 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][5])
    image7 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][6])
    image8 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][7])
    image9 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][8])
    image10 = get_image(dataset_hdf5_all_key, dataset_image_mask, processid_to_index, I[0][9])
        
    return id_list, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10

with gr.Blocks() as demo:
    # open general files
    dataset_hdf5_all_key = h5py.File('full5m/BIOSCAN_5M.hdf5', "r", libver="latest")['all_keys']['image']
    with open("dataset_processid_list.pickle", "rb") as f: # not finding certain ids
        dataset_processid_list = pickle.load(f)
    with open("dataset_image_mask.pickle", "rb") as f: 
        dataset_image_mask = pickle.load(f)
    with open("processid_to_index.pickle", "rb") as f: 
        processid_to_index = pickle.load(f)

    with open("indx_to_dna_id_dict.pickle", "rb") as f: 
        indx_to_id_dict = pickle.load(f)

    # open image files
    with open("id_to_image_emb_dict.pickle", "rb") as f: 
        id_to_image_emb_dict = pickle.load(f)

    # open dna files
    with open("id_to_dna_emb_dict.pickle", "rb") as f: 
        id_to_dna_emb_dict = pickle.load(f)

    

    with gr.Column():
        process_id = gr.Textbox(label="ID:", info="Enter a sample ID to search for")
        process_id_list = gr.Textbox(label="Closest 10 matches:" )
        mod1 = gr.Radio(choices=["DNA", "Image"], label="Search From:")
        mod2 = gr.Radio(choices=["DNA", "Image"], label="Search To:")
        search_btn = gr.Button("Search")

    with gr.Row():
        image1 = gr.Image(label=1)
        image2 = gr.Image(label=2)
        image3 = gr.Image(label=3)
        image4 = gr.Image(label=4)
        image5 = gr.Image(label=5)
    with gr.Row():
        image6 = gr.Image(label=6)
        image7 = gr.Image(label=7)
        image8 = gr.Image(label=8)
        image9 = gr.Image(label=9)
        image10 = gr.Image(label=10)
    
    search_btn.click(fn=searchEmbeddings, inputs=[process_id, mod1, mod2], 
                     outputs=[process_id_list, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10])
    examples = gr.Examples(
        examples=[["ABOTH966-22", "DNA", "DNA"],
                  ["CRTOB8472-22", "DNA", "Image"],
                  ["PLOAD050-20", "Image", "DNA"],
                  ["HELAC26711-21", "Image", "Image"]],
        inputs=[process_id, mod1, mod2],)

# ARONZ671-20
demo.launch()