import faiss
import time
import numpy as np
import os
import uuid
import json
import random
import torch
import torch.nn as nn
import argparse
import shutil
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import torchaudio
import warnings
from eval import get_index, load_memmap_data
from test_fp import create_fp_db
from util import \
create_fp_dir, load_config, \
query_len_from_seconds, seconds_from_query_len, \
load_augmentation_index
from modules.data import NeuralfpDataset
from encoder.graph_encoder import GraphEncoder
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformNeuralfp
from eval import get_index, load_memmap_data

root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--test_dir', default='data/fma_small.json', type=str,
                    help='path to test data')
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--query_lens', default=None, type=str)
parser.add_argument('--encoder', default='grafp', type=str)
parser.add_argument('--small_test', action='store_true', default=False)
parser.add_argument('--text', default='test', type=str)
parser.add_argument('--test_snr', default=None, type=int)
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--k', default=3, type=int)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--test_ids', default='2000', type=str)
parser.add_argument('--shuffle', action='store_true', default=False)
parser.add_argument('--test_config', default='config/test_config.yaml', type=str)


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def predict_query(emb_dir, 
                emb_dummy_dir=None,
                index_type='ivfpq',
                nogpu=False,
                max_train=1e7,
                test_ids='icassp',
                test_seq_len='1 3 5 9 11 19',
                k_probe=20,
                n_centroids=64):

    if type(test_seq_len) == str:
        test_seq_len = np.asarray(
            list(map(int, test_seq_len.split())))  # '1 3 5' --> [1, 3, 5]

    # we might either have a single or multiple queries, so put them in a folder
    query, query_shape, query_metadata = load_memmap_data(emb_dir, 'query')
    if emb_dummy_dir is None:
        emb_dummy_dir = emb_dir
    dummy_db, dummy_db_shape, dummy_db_metadata = load_memmap_data(emb_dummy_dir, 'dummy_db')


    index = get_index(index_type, dummy_db, dummy_db.shape, (not nogpu),
                      max_train, n_centroids=n_centroids)
    index.add(dummy_db); print(f'{len(dummy_db)} items from dummy DB')
    del dummy_db

    # TODO: check if fake_recon_index is necessary or we can keep dummy_db
    fake_recon_index, index_shape, _ = load_memmap_data(
        emb_dummy_dir, 'dummy_db',
        display=False)
    fake_recon_index.flush()

    # Get test_ids
    print(f'test_id: \033[93m{test_ids}\033[0m,  ', end='')
    if test_ids.lower() == 'all':
        # test every possible starting position
        test_ids = np.arange(0, len(query) - max(test_seq_len), 1) # will test all segments in query/db set
    elif test_ids.isnumeric():
        # randomly sample N positions
        np.random.seed(42)
        test_ids = np.random.permutation(len(query) - max(test_seq_len))[:int(test_ids)]
    else:
        # load from file
        test_ids = np.load(test_ids)

    # test_ids = starting segment index in thet query array to test 
    # n_test = #queries to run
    n_test = len(test_ids)
    print(f'n_test: \033[93m{n_test:n}\033[0m')

    pred_id_results = np.zeros((n_test, len(test_seq_len))).astype(int)  # shape: (n_test, n_seq_len)

    for ti, test_id in enumerate(test_ids):
        for si, sl in enumerate(test_seq_len):
            assert test_id <= len(query)
            q = query[test_id:(test_id + sl), :] # shape(q) = (length, dim)

            # segment-level top k search for each segment
            _, I = index.search(
                q, k_probe) # _: distance, I: result IDs matrix

            # offset compensation to get the start IDs of candidate sequences
            for offset in range(len(I)):
                I[offset, :] -= offset

            # unique candidates
            candidates = np.unique(I[np.where(I >= 0)])   # ignore id < 0

            """ Sequence match score """
            _scores = np.zeros(len(candidates))
            for ci, cid in enumerate(candidates):
                _scores[ci] = np.mean(
                    np.diag(
                        # np.dot(q, index.reconstruct_n(cid, (cid + l)).T)
                        np.dot(q, fake_recon_index[cid:cid + sl, :].T)
                        )
                    )

            """ Evaluate """
            pred_ids = candidates[np.argsort(-_scores)[:10]]
            # pred_id = candidates[np.argmax(_scores)] <-- only top1-hit
            
            # ? change to 3, 5 or 10 
            pred_id_results[ti, si] = pred_ids[0]

    del fake_recon_index, query
            
    pred_metadata_results = [[(dummy_db_metadata[idx].tolist(),query_metadata[i]) for idx in row] for i,row in enumerate(pred_id_results)]
    print(pred_metadata_results)

def main():
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_dir = cfg['data_dir']
    if args.test_snr is not None:
        cfg['val_snr'] = [int(args.test_snr), int(args.test_snr)]

    # Hyperparameters
    random_seed = 42

    ############# ablation experimental setup #################
    if args.model is not None:
        if args.model == 'tc_27':
            cfg['offset'] = 0.2
            cfg['overlap'] = 0.5
        elif args.model == 'tc_29':
            cfg['offset'] = 0.05
            cfg['overlap'] = 0.9
        elif args.model == 'tc_30':
            cfg['offset'] = 0.1
            cfg['overlap'] = 0.8
        elif args.model == 'tc_31':
            cfg['offset'] = 0.125
            cfg['overlap'] = 0.75
    ###########################################################

    print("Creating new model...")
    if args.encoder == 'resnet':
        # TODO: Add support for resnet encoder (deprecated)
        raise NotImplementedError
    elif args.encoder == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoder(cfg=cfg, in_channels=cfg['n_filters'], k=args.k))
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # model = DataParallel(model).to(device)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model = model.to(device)

    print("Creating dataloaders ...")
    
    augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=None, 
                                        noise_dir=None, 
                                        train=False).to(device)

    dataset = NeuralfpDataset(cfg, path=args.test_dir, train=False, inference=True)
    
    query_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False)

    if args.small_test:
        index_type = 'l2'
    else:
        index_type = 'ivfpq'

    if args.query_lens is not None:
        args.query_lens = [int(q) for q in args.query_lens.split(',')]
        test_seq_len = [query_len_from_seconds(q, cfg['overlap'], dur=cfg['dur'])
                        for q in args.query_lens]
    
    ckp_name = 'tc_27'
    epochs = 'best'
    if not type(epochs) == list:
        epochs = [epochs]
    writer = SummaryWriter(f'runs/{ckp_name}')

    for epoch in epochs:
        ckp = os.path.join(model_folder, f'model_{ckp_name}_{str(epoch)}.pth')
        if os.path.isfile(ckp):
            print("=> loading checkpoint '{}'".format(ckp))
            checkpoint = torch.load(ckp)
            # Check for DataParallel
            if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
                checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(ckp))
            continue

        if not os.path.exists('inference'):
            os.makedirs('inference')   
        fp_dir = f'inference/{int(time.time())}'
        os.makedirs(fp_dir, exist_ok=True)

        create_fp_db(query_db_loader, augment=augment, 
                        model=model, output_root_dir=fp_dir, verbose=False)
        
        text = f'{args.text}_{str(epoch)}'
        label = epoch if type(epoch) == int else 0

        if args.query_lens is not None:
            predict_query(emb_dir=fp_dir,
                                emb_dummy_dir='/content/drive/MyDrive/GraFP/logs/store/small/model_tc_27_best',
                                test_ids=args.test_ids, 
                                test_seq_len=test_seq_len, 
                                index_type=index_type,
                                nogpu=True) 
        else:
            predict_query(emb_dir=fp_dir, 
                                emb_dummy_dir='/content/drive/MyDrive/GraFP/logs/store/small/model_tc_27_best',
                                test_ids=args.test_ids, 
                                index_type=index_type,
                                nogpu=True)



if __name__ == '__main__':
    main()