import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

BLOCK_SIZE = 128


def weight_dequant(weight, scale, block_size=BLOCK_SIZE):
    """
    Dequantize a block-wise FP8 weight tensor into the current default dtype.

    DeepSeek-V3.2 stores one scale value per `block_size x block_size` tile. Some
    weight matrices are not exact multiples of the block size, so the final tiles
    along each dimension may be partial. This helper pads to the scale grid,
    applies the block scales, and trims the padded region afterward.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(weight.shape)}")
    if scale.dim() != 2:
        raise ValueError(f"Expected a 2D scale tensor, got shape {tuple(scale.shape)}")

    rows, cols = weight.shape
    scale_rows = (rows + block_size - 1) // block_size
    scale_cols = (cols + block_size - 1) // block_size
    expected_scale_shape = (scale_rows, scale_cols)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"Scale shape mismatch for weight {tuple(weight.shape)}: "
            f"expected {expected_scale_shape}, got {tuple(scale.shape)}"
        )

    padded_rows = scale_rows * block_size
    padded_cols = scale_cols * block_size
    if padded_rows != rows or padded_cols != cols:
        padded_weight = weight.new_zeros((padded_rows, padded_cols))
        padded_weight[:rows, :cols] = weight
    else:
        padded_weight = weight

    dequant = padded_weight.view(scale_rows, block_size, scale_cols, block_size)
    dequant = dequant.transpose(1, 2).contiguous()
    dequant = (
        dequant.float() * scale.view(scale_rows, scale_cols, 1, 1).float()
    ).to(torch.get_default_dtype())
    dequant = dequant.transpose(1, 2).contiguous().view(padded_rows, padded_cols)
    return dequant[:rows, :cols].contiguous()


def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()

    resume_index = 0
    while resume_index < len(safetensor_files):
        output_shard = os.path.join(
            bf16_path, os.path.basename(safetensor_files[resume_index])
        )
        if not os.path.exists(output_shard):
            break
        resume_index += 1

    if resume_index == len(safetensor_files):
        print("All shards already converted, refreshing model index only.")
        safetensor_files = []
    else:
        if resume_index > 0:
            print(
                "Resuming from",
                os.path.basename(safetensor_files[resume_index]),
            )
        safetensor_files = safetensor_files[resume_index:]

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
                
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file({k: v.cpu() for k, v in new_state_dict.items()}, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump(
            {
                "metadata": model_index.get("metadata", {}),
                "weight_map": {
                    name: file_name
                    for name, file_name in weight_map.items()
                    if not name.endswith("_scale_inv")
                },
            },
            f,
            indent=2,
        )
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
    
