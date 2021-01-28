import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
from stylegan2_pytorch import Trainer, NanException

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
from pathlib import Path
import urllib


def download_checkpoints(checkpoint_dir, model_name):

    checkpoint_url = "https://github.com/vlbthambawita/deepsynth-gitract/releases/download/checkpoints-v1.0/model_1000.pt"

    download_path = Path(Path(checkpoint_dir) / model_name)

    checkpoint_path = Path(download_path / "model_1000.pt")

    if os.path.exists(download_path):
        print("checkpoint is already available at:", checkpoint_path)
        return 

    else:

        try:
            os.makedirs(download_path, exist_ok=True)    
            print("Downloading checkpoints...")
            #wget.download(checkpoint_url, str(Path(download_path / "model_1000.pt")))
            urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
            print("Download checkpoints to:", str(checkpoint_path))

        except:

            print("Downloading error. Check your connections...!")


def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def train_from_folder(
    data = './data',
    results_dir = './results',
    models_dir = ' ',
    name = 'test', # Vajira
    new = False,
    load_from = -1,
    image_size = 128,
    network_capacity = 16,
    fmap_max = 512,
    transparent = False,
    batch_size = 5,
    gradient_accumulate_every = 1,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False, # changed to 
    num_generate = 1,
    generate_interpolation = True,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 1,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    fp16 = False,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [1,2],
    no_const = False,
    aug_prob = 0.,
    aug_types = ['translation', 'cutout'],
    top_k_training = False,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dataset_aug_prob = 0.,
    multi_gpus = False,
    calculate_fid_every = None,
    seed = 42,
    log = False
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dataset_aug_prob = dataset_aug_prob,
        calculate_fid_every = calculate_fid_every,
        mixed_prob = mixed_prob,
        log = log
    )

    if generate:
        model = Trainer(**model_args)
        #model.models_dir = Path("/work/vajira/DL/checkpoints")
        model.load(load_from)
        samples_name = timestamped_filename()
        for num in tqdm(range(num_generate)):
            model.evaluate(f'{samples_name}-{num}', num_image_tiles)
        print(f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    if generate_interpolation:
        model = Trainer(**model_args)
        #model.models_dir = Path("/work/vajira/DL/checkpoints")
        model.load(load_from)
        samples_name = timestamped_filename()
        model.generate_interpolation(samples_name, num_image_tiles, num_steps = interpolation_num_steps, save_frames = save_frames)
        print(f'interpolation generated at {results_dir}/{name}/{samples_name}')
        return

    

def generate(name, result_dir, checkpoint_dir, num_img_per_tile, num_of_outputs, trunc_psi=0.75, **kwargs):
    """ Generate deepfake Gastrointestinal tract images.

    Keyword arguments:
    name -- Any name to keep trac of generations
    result_dir -- A directory to save output
    checkpoint_dir -- A directory to download pre-trained checkpoints
    num_img_per_tile -- Number of images per dimenstion of the grid
    num_of_outputs -- Number of outputs to generate
    trunc_psi -- value between 0.5 and 1.0 (default 0.75)
    """

    # Download checkpoints
    download_checkpoints(checkpoint_dir, name)

    # Generate  data to folder
    train_from_folder(name=name, models_dir=checkpoint_dir, results_dir=result_dir, 
                    num_image_tiles=num_img_per_tile, num_generate=num_of_outputs,
                    generate=True, generate_interpolation=False, trunc_psi=trunc_psi, **kwargs)


def generate_interpolation(name, result_dir, checkpoint_dir, num_img_per_tile, num_of_outputs, num_of_steps_to_interpolate, save_frames, trunc_psi=0.75, **kwargs):
    """ Generate deepfake Gastrointestinal tract images.

    Keyword arguments:
    name -- Any name to keep trac of generations
    result_dir -- A directory to save output
    checkpoint_dir -- A directory to download pre-trained checkpoints
    num_img_per_tile -- Number of images per dimenstion of the grid
    num_of_outputs -- Number of outputs to generate
    num_of_steps_to_interpolate -- Number of step between two random points
    save_frames -- True if you want frame by frame, otherwise .gif will be generated
    trunc_psi -- value between 0.5 and 1.0 (default 0.75)
    """
    # Download checkpoints
    download_checkpoints(checkpoint_dir, name)

    # Generate  data to folder
    train_from_folder(name=name, models_dir=checkpoint_dir, results_dir=result_dir, 
                    num_image_tiles=num_img_per_tile, num_generate=num_of_outputs,
                    generate=False, generate_interpolation=True, interpolation_num_steps=num_of_steps_to_interpolate, 
                    save_frames=save_frames, trunc_psi=trunc_psi, **kwargs)


