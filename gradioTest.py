import gradio as gr
import torch
import numpy as np
import h5py
import faiss
import json
from PIL import Image
import io 
import pickle

def get_image(file, dataset_image_mask, processid_to_index, idx):
    image_enc_padded = file["image"][idx].astype(np.uint8)
    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

def searchEmbeddings(id):
    # variable and index initialization
    dim = 768
    count = 0
    num_neighbors = 10

    image_index = faiss.IndexFlatIP(dim)

    # load dictionaries
    with open("id_to_emb_dict.pickle", "rb") as f:
        id_to_emb_dict = pickle.load(f)
    with open("indx_to_id.pickle", "rb") as f:
        indx_to_id_dict = pickle.load(f)

    # get index
    image_index = faiss.read_index("image_index.index")

    # search for query
    query = id_to_emb_dict[id]
    query = query.astype(np.float32)
    D, I = image_index.search(query, num_neighbors)

    id_list = []
    i = 0
    for indx in I[0]:
        # id_list[i] = indx_to_id_dict[0]
        id = indx_to_id_dict[indx]
        id_list.append(id)
    
    # get image data
    dataset_hdf5_all_key = h5py.File('BIOSCAN_5M.hdf5', "r", libver="latest")['all_keys']
    dataset_processid_list = [item.decode("utf-8") for item in dataset_hdf5_all_key["processid"][:]]
    dataset_image_mask = dataset_hdf5_all_key["image_mask"][:]
    processid_to_index = {pid: idx for idx, pid in enumerate(dataset_processid_list)}

    # get images
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

demo = gr.Interface(
    fn=searchEmbeddings,
    inputs=[gr.Textbox(label="ID:", info="Enter a sample ID to search for")],
    outputs=[gr.Textbox(label="Results:"),
             gr.Image(label="1"),
             gr.Image(label="2"),
             gr.Image(label="3"),
             gr.Image(label="4"),
             gr.Image(label="5"),
             gr.Image(label="6"),
             gr.Image(label="7"),
             gr.Image(label="8"),
             gr.Image(label="9"),
             gr.Image(label="10")],
    title="Bioscan-Clip",
    description="Taxonomic classification tool for insects",
)

demo.launch()