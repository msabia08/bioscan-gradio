import io
import os
import gradio as gr
import cv2
import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm
from bioscanclip.model.simple_clip import load_clip_model
import faiss
import pickle
import random


def getRandID():
    indx = random.randrange(0, 396503)
    return index_to_id_dict[indx], indx

# broken
def searchEmbeddingsID(id, mod1, mod2):
    # variable and index initialization
    dim = 768
    count = 0
    num_neighbors = 10
    # index = faiss.IndexFlatIP(dim)

    # get index
    if (mod2 == "Image"):
        index = image_index_IP
    elif (mod2 == "DNA"):
        index = dna_index_IP
    
    # search for query
    if (mod1 == "Image"):
        query = image_index_IP.reconstruct(id)
    elif (mod1 == "DNA"):
        query = dna_index_IP.reconstruct(id)
    query = query.astype(np.float32)
    print("ID Query: \n\n", query[:4])
    D, I = index.search(query, num_neighbors)

    id_list = []
    i = 1
    for indx in I[0]:
        id = index_to_id_dict[indx]
        id_list.append(id)
        
    return id_list

def searchEmbeddingsImage(image, mod2):
    dim = 768
    count = 0
    num_neighbors = 10
    # index = faiss.IndexFlatIP(dim)

    # get index
    if (mod2 == "Image"):
        index = image_index_IP
    elif (mod2 == "DNA"):
        index = dna_index_IP
    
    query = getQuery(image)
    query = query.astype(np.float32)
    print("Image Query: \n\n", query[:4])
    temp = index.reconstruct(processid_to_index["GMSPA11989-21"])
    print("Diff:", query-temp)
    D, I = index.search(query, num_neighbors)

    id_list = []
    i = 1
    for indx in I[0]:
        id = index_to_id_dict[indx]
        id_list.append(id)
        
    return id_list

def encode_image(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    image_output = F.normalize(model(image), p=2, dim=-1)
    feature = image_output.cpu().detach().numpy()
    return feature

def get_image_encoder(model, device):
    image_encoder = model.image_encoder
    image_encoder.eval()
    image_encoder.to(device)
    return image_encoder


def load_image():
        print(hdf5["val_seen"]["processid"][0])
        image_enc_padded = hdf5['val_seen']["image"][0].astype(np.uint8)
        enc_length = hdf5['val_seen']["image_mask"][0]
        image_enc = image_enc_padded[:enc_length]
        curr_image = Image.open(io.BytesIO(image_enc))
        return curr_image

def wrapperFunc(args: DictConfig):
    def getQuery(im):
        # print(args)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        # Init transform
        transform = transforms.Compose(
                        [
                            transforms.ToTensor(),
                            transforms.Resize(size=256, antialias=True),
                            transforms.CenterCrop(224),
                        ]
                    )

        model = load_clip_model(args, device)
        if hasattr(args.model_config, "load_ckpt") and args.model_config.load_ckpt is False:
            pass
        else:
            checkpoint = torch.load(args.model_config.ckpt_path, map_location="cuda:0")
            model.load_state_dict(checkpoint)
        model.eval()
        # Get the image encoder
        image_encoder = get_image_encoder(model, device)
        # Encode all the images
        # im = load_image()
        encoded_feature = encode_image(im, image_encoder, transform, device)

        return encoded_feature
    
    return getQuery

with gr.Blocks() as demo:
    # hdf5 = h5py.File('/localhome/cs3dlgv/bioscan-clip/bioscan-clip-scripts/extracted_features_for_all_5m_data.hdf5', "r", libver="latest")
    hdf5 = h5py.File('/localhome/cs3dlgv/bioscan-clip/data/BIOSCAN_5M/BIOSCAN_5M.hdf5', "r", libver="latest")

    image_index_IP = faiss.read_index("bioscan-clip-scripts/bioscan_5m_image_IndexFlatIP.index")
    # image_index_IP = faiss.read_index("bioscan-clip-scripts/index/bioscan_5m_3_3.index")
    dna_index_IP = faiss.read_index("bioscan-clip-scripts/bioscan_5m_dna_IndexFlatIP.index")

    # with open("bioscan-clip-scripts/pickle/processid_to_indx_3_5.pickle", "rb") as f:
    #     processid_to_index = pickle.load(f)
    with open("bioscan-clip-scripts/big_indx_to_id_dict.pickle", "rb") as f:
        index_to_id_dict = pickle.load(f)
    processid_to_index = {v: k for k, v in index_to_id_dict.items()}

    print(image_index_IP.ntotal)
    print(len(processid_to_index))
    print(len(index_to_id_dict))

    print(np.unique(list(index_to_id_dict.values()), return_counts=True))

    # np.unique(list(index_to_id_dict.values()), return_counts=True)
    

    with gr.Column():
        with gr.Row():
            with gr.Column():
                rand_id = gr.Textbox(label="Random ID:")
                rand_id_indx = gr.Textbox(label="Index:")
                id_btn = gr.Button("Get Random ID")
            with gr.Column():
                mod1 = gr.Radio(choices=["DNA", "Image"], label="Search From:")
                mod2 = gr.Radio(choices=["DNA", "Image"], label="Search To:")

        indexType = gr.Radio(choices=["FlatIP(default)"], label="Index:", value="FlatIP(default)")
        process_id = gr.Textbox(label="ID:", info="Enter a sample ID to search for")
        process_id_list_ids = gr.Textbox(label="Closest 10 matches:")
        search_id_btn = gr.Button("Search")
        id_btn.click(fn=getRandID, inputs=[], outputs=[rand_id, rand_id_indx])

        image_input = gr.Image(type="numpy")
        process_id_list_images = gr.Textbox(label="Closest 10 matches:")
        with gr.Row():
            search_image_btn = gr.Button("Search")

        
    search_image_btn.click(fn=searchEmbeddingsImage, inputs=[image_input, mod2], outputs=[process_id_list_images])
    search_id_btn.click(fn=searchEmbeddingsID, inputs=[process_id, mod1, mod2], 
                     outputs=[process_id_list_ids])


# @hydra.main(config_path="../bioscanclip/config", config_name="global_config", version_base="1.1")
hydra.initialize(config_path="../bioscanclip/config", version_base="1.1")
args = hydra.compose(config_name="global_config", overrides=["model_config=for_bioscan_5m/final_experiments/image_dna_text_seed_42.yaml"])
# args = hydra.compose(config_name="global_config", return_hydra_config=True)


getQuery = wrapperFunc(args)
demo.launch()

# test image: GMGKA4995-21, BIOUG56844-C09

# mkl-fft==1.3.1
# torch==1.12.0+cu116
# torchaudio==0.12.0+cu116
# torchvision==0.13.0+cu116
# gdown==4.7.1+cu116

# -3.72611322e-02  1.92827024e-02 -1.35646798e-02  2.36576665e-02
# 2.25652531e-02 -2.84570977e-02  1.53016858e-02  4.66185901e-03

# cropping, index, or checkpoint
# test post cropped image first through hdf5
# make branch, push/merge file, ask about pull request
