import torch
import tensorstore # needed for bfloat16 on zarr
import zarr
import numpy as np
from pathlib import Path
from safetensors.torch import save_file
import gc
from tqdm import tqdm
from collections import OrderedDict

layer_mappings = {
        'layers.mlp.linear_fc1.layer_norm_bias': 'model.layers.{lnum}.mlp.input_layernorm.weight',
        'layers.mlp.linear_fc1.layer_norm_weight': 'model.layers.{lnum}.mlp.input_layernorm.bias',
        'layers.mlp.linear_fc1.weight': 'model.layers.{lnum}.mlp.up_proj.weight',
        'layers.mlp.linear_fc2.weight': 'model.layers.{lnum}.mlp.down_proj.weight',
        'layers.self_attention.linear_qkv.weight': 'model.layers.{lnum}.self_attn.qkv_proj.weight',
        'layers.self_attention.linear_proj.weight': 'model.layers.{lnum}.self_attn.o_proj.weight',
        'layers.self_attention.linear_qkv.layer_norm_bias': 'model.layers.{lnum}.post_attention_layernorm.bias',
        'layers.self_attention.linear_qkv.layer_norm_weight': 'model.layers.{lnum}.post_attention_layernorm.weight',
        'embedding.word_embeddings.weight': 'model.embed_tokens.weight',
        'final_layernorm.weight': 'model.norm.weight',
        'final_layernorm.bias': 'model.norm.bias',
        'output_layer.weight': 'lm_head.weight'
}

def convert_to_torch(tensor):
    if "bfloat16" in tensor.dtype.name:
        # bfloat16 isn't properly supported by numpy, so gotta convert to a different format then back
        tensor = torch.from_numpy(tensor.view(np.int16)).view(torch.bfloat16)
    else:
        tensor = torch.from_numpy(tensor)
    return tensor


def convert_nemo(path: Path):
    model_map = {}
    layer_count = 0
    special_layers = {}

    for subdir in path.iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        sharded_state_dict = {}
        key = subdir.name

        arr = zarr.convenience.open(subdir,'r')

        key = key.split('.')
        while key[0] in ('model','decoder'):
            key.pop(0)

        multi_layered = key[0] == 'layers'
        key = '.'.join(key)

        if not multi_layered:
            arr = np.expand_dims(arr,0)
            special_layers[key] = arr
        else:
            if layer_count < arr.shape[0]:
                layer_count = arr.shape[0]
            model_map[key] = arr

    print("Exporting", layer_count, "layers")

    # have the index ordered mostly for readability's sake
    index = OrderedDict()

    # we store the output layer at the end in its own file, and keep it at top of index
    index['lm_head.weight'] = f"model-{layer_count+1:05}-of-{layer_count+1:05}"
    output_layer = convert_to_torch(special_layers['output_layer.weight'])
    save_file({'lm_head.weight':output_layer},f"model-{layer_count+1:05}-of-{layer_count+1:05}")

    # now that we have instances to each, let's store things by order of layers for better loading
    for layer in range(layer_count):
        # hacky way of positioning standalone layers:
        if layer == 0:
            model_map['embedding.word_embeddings.weight'] = special_layers['embedding.word_embeddings.weight']
        elif layer == layer_count-1:
            model_map['final_layernorm.weight'] = special_layers['final_layernorm.weight']
            model_map['final_layernorm.bias'] = special_layers['final_layernorm.bias']

        sharded_state_dict = dict()
        fname = f"model-{layer+1:05}-of-{layer_count+1:05}.safetensors"

        for key,arr in tqdm(model_map.items()):
            lnum = layer
            if arr.shape[0] < layer:
                lnum = 0
            k = layer_mappings[key].replace("{lnum}",str(layer))
            shared_state_dict[k] = convert_to_torch(arr[lnum,:])
            index[k] = fname

        save_file(sharded_state_dict,fname)

        # cleanup to save RAM
        del sharded_state_dict
        gc.collect()

        print("saved",fname)
        if layer == 0:
            del model_map['embedding.word_embeddings.weight']

if __name__ == "__main__":
    convert_nemo(Path.cwd())
