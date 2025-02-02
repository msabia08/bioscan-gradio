import gradio as gr
import torch
import numpy as np
import h5py
import faiss
import json
import random
import pickle

def getFlatIP():
    test_index = faiss.IndexFlatIP(768)
    return test_index

def getFlatL2():
    test_index = faiss.IndexFlatL2(768)
    return test_index

def getIVFFlat(all_keys, seen_test, unseen_test, seen_val, unseen_val):
    quantizer = faiss.IndexFlatIP(768)
    test_index = faiss.IndexIVFFlat(quantizer, 768, 128)
    test_index.train(all_keys)
    test_index.train(seen_test)
    test_index.train(unseen_test)
    test_index.train(seen_val)
    test_index.train(unseen_val)
    return test_index

def getHNSW():
    # 16: connections for each vertex. efSearch: depth of search during search. efConstruction: depth of search during build
    test_index = faiss.IndexHNSWFlat(768, 16)
    test_index.hnsw.efSearch = 32
    test_index.hnsw.efConstruction = 64
    return test_index

def getLSH():
    test_index = faiss.IndexLSH(768, 768*2)
    return test_index

def getIdToEmbedding(allid, stid, utid, svalid, uvalid, all_keys, seen_test, unseen_test, seen_val, unseen_val):
    id_to_emb_dict = dict()
    i = 0
    for id in allid:
        id_to_emb_dict[id] = np.array([all_keys[i]])
        i += 1
    for id in stid:
        id_to_emb_dict[id] = np.array([seen_test[i]])
        i += 1
    for id in utid:
        id_to_emb_dict[id] = np.array([unseen_test[i]])
        i += 1
    for id in svalid:
        id_to_emb_dict[id] = np.array([seen_val[i]])
        i += 1
    for id in uvalid:
        id_to_emb_dict[id] = np.array([unseen_val[i]])
        i += 1
    
    return id_to_emb_dict

def main():
    # initialize data
    all_keys_dna = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_all_keys.hdf5', "r", libver="latest")['encoded_dna_feature'][:]
    seen_test_dna = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_seen_test.hdf5', "r", libver="latest")['encoded_dna_feature'][:]
    unseen_test_dna = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_unseen_test.hdf5', "r", libver="latest")['encoded_dna_feature'][:]
    seen_val_dna = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_seen_val.hdf5', "r", libver="latest")['encoded_dna_feature'][:]
    unseen_val_dna = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_unseen_val.hdf5', "r", libver="latest")['encoded_dna_feature'][:]

    all_keys_im = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_all_keys.hdf5', "r", libver="latest")['encoded_image_feature'][:]
    seen_test_im = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_seen_test.hdf5', "r", libver="latest")['encoded_image_feature'][:]
    unseen_test_im = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_unseen_test.hdf5', "r", libver="latest")['encoded_image_feature'][:]
    seen_val_im = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_seen_val.hdf5', "r", libver="latest")['encoded_image_feature'][:]
    unseen_val_im = h5py.File('bioscan-clip-scripts/extracted_features/extracted_features_of_unseen_val.hdf5', "r", libver="latest")['encoded_image_feature'][:]

    dataset = h5py.File('data/BIOSCAN_5M/BIOSCAN_5M.hdf5', "r", libver="latest")
    allid = [item.decode("utf-8") for item in dataset["all_keys"]["processid"][:]]
    stid = [item.decode("utf-8") for item in dataset["test_seen"]["processid"][:]]
    utid = [item.decode("utf-8") for item in dataset["test_unseen"]["processid"][:]]
    svalid = [item.decode("utf-8") for item in dataset["val_seen"]["processid"][:]]
    uvalid = [item.decode("utf-8") for item in dataset["val_unseen"]["processid"][:]]

    all_keys = dataset["all_keys"]
    seen_test = dataset["test_seen"]
    unseen_test = dataset["test_unseen"]
    seen_val = dataset["val_seen"]
    unseen_val = dataset["val_unseen"]

    # d = getIdToEmbedding(allid, stid, utid, svalid, uvalid, all_keys_dna, seen_test_dna, unseen_test_dna, seen_val_dna, unseen_val_dna)
    # d = getIdToEmbedding(allid, stid, utid, svalid, uvalid, all_keys_im, seen_test_im, unseen_test_im, seen_val_im, unseen_val_im)

    big_id_to_image_emb_dict = dict()
    i = 0
    for object in allid:
        big_id_to_image_emb_dict[object] = np.array([all_keys_im[i]])
        i += 1
    i = 0
    for object in stid:
        big_id_to_image_emb_dict[object] = np.array([seen_test_im[i]])
        i += 1
    i = 0
    for object in utid:
        big_id_to_image_emb_dict[object] = np.array([unseen_test_im[i]])
        i += 1
    i = 0
    for object in svalid:
        big_id_to_image_emb_dict[object] = np.array([seen_val_im[i]])
        i += 1
    i = 0
    for object in uvalid:
        big_id_to_image_emb_dict[object] = np.array([unseen_val_im[i]])
        i += 1

    ###

    big_id_to_dna_emb_dict = dict()
    i = 0
    for object in allid:
        big_id_to_dna_emb_dict[object] = np.array([all_keys_dna[i]])
        i += 1
    i = 0
    for object in stid:
        big_id_to_dna_emb_dict[object] = np.array([seen_test_dna[i]])
        i += 1
    i = 0
    for object in utid:
        big_id_to_dna_emb_dict[object] = np.array([unseen_test_dna[i]])
        i += 1
    i = 0
    for object in svalid:
        big_id_to_dna_emb_dict[object] = np.array([seen_val_dna[i]])
        i += 1
    i = 0
    for object in uvalid:
        big_id_to_dna_emb_dict[object] = np.array([unseen_val_dna[i]])
        i += 1

    ###

    processid_to_indx = dict()
    big_indx_to_id_dict = dict()
    i = 0
    for object in allid:
        big_indx_to_id_dict[i] = object
        processid_to_indx[object] = i
        i += 1

    for object in stid:
        big_indx_to_id_dict[i] = object
        processid_to_indx[object] = i
        i += 1
 
    for object in utid:
        big_indx_to_id_dict[i] = object
        processid_to_indx[object] = i
        i += 1

    for object in svalid:
        big_indx_to_id_dict[i] = object
        processid_to_indx[object] = i
        i += 1

    for object in uvalid:
        big_indx_to_id_dict[i] = object
        processid_to_indx[object] = i
        i += 1

    ###


    # with open('bioscan-clip-scripts/pickle/big_id_to_image_emb_dict.pickle', 'wb') as f:
    #     pickle.dump(big_id_to_emb_dict, f)

    # test_index.add(all_keys)
    # test_index.add(seen_test)
    # test_index.add(unseen_test)
    # test_index.add(seen_val)
    # test_index.add(unseen_val)
    # faiss.write_index(test_index, "bioscan-clip-scripts/index/big_image_index_LSH.index")


main()