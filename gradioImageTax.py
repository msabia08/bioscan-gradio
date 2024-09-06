import gradio as gr
import torch
import numpy as np
import h5py
import faiss
from PIL import Image
import io 
import pickle
import random

def get_image(file, dataset_image_mask, processid_to_index, idx):
    image_enc_padded = file[idx].astype(np.uint8)
    enc_length = dataset_image_mask[idx]
    image_enc = image_enc_padded[:enc_length]
    image = Image.open(io.BytesIO(image_enc))
    return image

def searchEmbeddings(id, mod1, mod2):
    # variable and index initialization
    original_indx = processid_to_index[id]    
    dim = 768
    num_neighbors = 10

    # get index
    index = faiss.IndexFlatIP(dim)
    if (mod2 == "Image"):
        index = faiss.read_index("index/image_index.index")
    elif (mod2 == "DNA"):
        index = faiss.read_index("index/dna_index.index")

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
    
    # get images
    image0 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, original_indx)
    image1 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][0])
    image2 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][1])
    image3 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][2])
    image4 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][3])
    image5 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][4])
    image6 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][5])
    image7 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][6])
    image8 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][7])
    image9 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][8])
    image10 = get_image(dataset_hdf5_all_key['image'], dataset_image_mask, processid_to_index, I[0][9])

    # get taxonomic information
    s0 = getTax(original_indx)
    s1 = getTax(I[0][0])
    s2 = getTax(I[0][1])
    s3 = getTax(I[0][2])
    s4 = getTax(I[0][3])
    s5 = getTax(I[0][4])
    s6 = getTax(I[0][5])
    s7 = getTax(I[0][6])
    s8 = getTax(I[0][7])
    s9 = getTax(I[0][8])
    s10 = getTax(I[0][9])
        
    return id_list, image0, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10

def getRandID():
    indx = random.randrange(0, 325667)
    return indx_to_id_dict[indx], indx

def getTax(indx):
    s = species[indx]
    g = genus[indx]
    f = family[indx]
    str = "Species: " + s + "\nGenus: " + g + "\nFamily: " + f
    return str

with gr.Blocks(title="Bioscan-Clip") as demo:
    # open general files
    dataset_hdf5_all_key = h5py.File('full5m/BIOSCAN_5M.hdf5', "r", libver="latest")['all_keys']
    with open("pickle/dataset_processid_list.pickle", "rb") as f:
        dataset_processid_list = pickle.load(f)
    with open("pickle/dataset_image_mask.pickle", "rb") as f: 
        dataset_image_mask = pickle.load(f)
    with open("pickle/processid_to_index.pickle", "rb") as f: 
        processid_to_index = pickle.load(f)
    with open("pickle/indx_to_id_dict.pickle", "rb") as f: 
        indx_to_id_dict = pickle.load(f)

    # open image files
    with open("pickle/id_to_image_emb_dict.pickle", "rb") as f: 
        id_to_image_emb_dict = pickle.load(f)

    # open dna files
    with open("pickle/id_to_dna_emb_dict.pickle", "rb") as f: 
        id_to_dna_emb_dict = pickle.load(f)

    # open taxonomy files
    family = [item.decode("utf-8") for item in dataset_hdf5_all_key["family"][:]]
    genus = [item.decode("utf-8") for item in dataset_hdf5_all_key["genus"][:]]
    species = [item.decode("utf-8") for item in dataset_hdf5_all_key["species"][:]]

    with gr.Column():
        process_id = gr.Textbox(label="ID:", info="Enter a sample ID to search for")
        process_id_list = gr.Textbox(label="Closest 10 matches:" )
        mod1 = gr.Radio(choices=["DNA", "Image"], label="Search From:")
        mod2 = gr.Radio(choices=["DNA", "Image"], label="Search To:")
        search_btn = gr.Button("Search")

        with gr.Row():
            with gr.Column():
                image0 = gr.Image(label="Original", height=550)
                tax0 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                rand_id = gr.Textbox(label="Random ID:")
                rand_id_indx = gr.Textbox(label="Index:")
                id_btn = gr.Button("Get Random ID")

        with gr.Row():
            with gr.Column():
                image1 = gr.Image(label=1)
                tax1 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image2 = gr.Image(label=2)
                tax2 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image3 = gr.Image(label=3)
                tax3 = gr.Textbox(label="Taxonomy")

        with gr.Row():
            with gr.Column():
                image4 = gr.Image(label=4)
                tax4 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image5 = gr.Image(label=5)
                tax5 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image6 = gr.Image(label=6)
                tax6 = gr.Textbox(label="Taxonomy")

        with gr.Row():   
            with gr.Column():
                image7 = gr.Image(label=7)
                tax7 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image8 = gr.Image(label=8)
                tax8 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image9 = gr.Image(label=9)
                tax9 = gr.Textbox(label="Taxonomy")
            with gr.Column():
                image10 = gr.Image(label=10)
                tax10 = gr.Textbox(label="Taxonomy")

    id_btn.click(fn=getRandID, inputs=[], outputs=[rand_id, rand_id_indx])
    search_btn.click(fn=searchEmbeddings, inputs=[process_id, mod1, mod2], 
                     outputs=[process_id_list, image0, image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, 
                              tax0, tax1, tax2, tax3, tax4, tax5, tax6, tax7, tax8, tax9, tax10])
    examples = gr.Examples(
        examples=[["ABOTH966-22", "DNA", "DNA"],
                  ["CRTOB8472-22", "DNA", "Image"],
                  ["PLOAD050-20", "Image", "DNA"],
                  ["HELAC26711-21", "Image", "Image"]],
        inputs=[process_id, mod1, mod2],)

demo.launch()