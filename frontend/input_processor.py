"""
Usage Examples:
  1. Standard FP32 Dummy Data:
     python input_processor.py --target_dir ./out
     
  2. FP32 Real Image (Batch=4):
     python input_processor.py --target_dir ./out --image_path ./cat.jpg --batch 4
     
  3. INT8 Quantized Input from CIFAR-10:
     python input_processor.py --target_dir ./out --dataset cifar10 --batch 128 \
         --quantize --scale 0.015 --zero_point 0
         
  4. Large Scale Dataset with Disk Streaming (e.g., 50000 ImageNet):
     python input_processor.py --target_dir ./out --dataset imagenet --batch 50000 \
         --data_root ./frontend_dataset
"""

import argparse
import numpy as np
from pathlib import Path
import torch
import json
import os
import sys

try:
    from PIL import Image
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

def process_and_save_input(args):
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Search upwards up to 3 directory levels for metadata.json
    meta_path = None
    current_search_dir = target_dir.resolve()
    for _ in range(3):
        candidate = current_search_dir / "metadata.json"
        if candidate.exists():
            meta_path = candidate
            break
        current_search_dir = current_search_dir.parent

    if meta_path:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            args.batch = meta.get("N", args.batch)
            args.channels = meta.get("C", args.channels)
            args.height = meta.get("H", args.height)
            args.width = meta.get("W", args.width)
        print(f"\033[1;36m[Input Processor]\033[0m Auto-loaded dimensions from metadata: {args.batch}x{args.channels}x{args.height}x{args.width}")
    else:
        print(f"\033[33m[WARNING] metadata.json not found in parents of {target_dir}. Using default arguments.\033[0m")

    tensor_shape = (args.batch, args.channels, args.height, args.width)

    # Define unified output paths and types
    bin_path = target_dir / "input_0.bin"
    npy_path = target_dir / "input_0.npy"
    labels_npy_path = target_dir / "labels_0.npy"
    labels_bin_path = target_dir / "labels_0.bin"

    dtype_np = np.int8 if args.quantize else np.float32
    dtype_str = "INT8" if args.quantize else "FP32"

    # Flow control flags
    data_already_saved = False
    has_labels = False

    # Mode C: Load from standard torchvision datasets (Disk Streaming)

    if args.dataset:
        if not HAS_VISION:
            raise ImportError("[Error] torchvision is required to load standard datasets.")

        print(f"[Input Processor] Fetching {args.batch} samples from dataset: {args.dataset.upper()}")

        dataset_name = args.dataset.lower()
        root_dir = args.data_root

        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
            dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
        elif dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.Resize((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
            dataset = torchvision.datasets.MNIST(root=root_dir, train=False, download=True, transform=transform)
        elif dataset_name == 'imagenet':
            imagenet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((args.height, args.width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_dir = os.path.join(root_dir, 'imagenet', 'val')
            if not os.path.exists(val_dir):
                raise FileNotFoundError(f"[Error] ImageNet directory not found at: {val_dir}\n"
                                        "Please create the directory and download ILSVRC2012_img_val.tar into it.")

            subdirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
            if len(subdirs) < 1000:
                print(f"\n\033[1;31m[CRITICAL ERROR] ImageNet Validation Set is NOT properly organized!\033[0m")
                print(f"Please run the following commands in your terminal:")
                print(f"  cd {val_dir}")
                print(f"  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash\n")
                raise RuntimeError("ImageNet folder structure invalid.")

            dataset = torchvision.datasets.ImageFolder(root=val_dir, transform=imagenet_transform)
        else:
            raise ValueError(f"[Error] Unsupported dataset: {dataset_name}.")

        print(f"  -> Initializing disk storage (Memory Map) to prevent OOM...")
        # Directly allocate .npy space on disk to prevent physical memory overflow
        fp_inputs = np.lib.format.open_memmap(str(npy_path), mode='w+', dtype=dtype_np, shape=tensor_shape)
        fp_labels = np.lib.format.open_memmap(str(labels_npy_path), mode='w+', dtype=np.int32, shape=(args.batch,))

        dataloader = DataLoader(dataset, batch_size=250, shuffle=False, num_workers=4)
        loaded_count = 0

        with open(bin_path, 'wb') as f_bin, open(labels_bin_path, 'wb') as f_labels_bin:
            for inp, lbl in dataloader:
                current_batch = inp.size(0)
                if loaded_count + current_batch > args.batch:
                    current_batch = args.batch - loaded_count
                    inp = inp[:current_batch]
                    lbl = lbl[:current_batch]

                if args.quantize:
                    inp = torch.round(inp / args.scale) + args.zero_point
                    inp = torch.clamp(inp, -128, 127).to(torch.int8)

                inp_np = inp.detach().cpu().numpy().astype(dtype_np)
                lbl_np = lbl.detach().cpu().numpy().astype(np.int32)

                fp_inputs[loaded_count : loaded_count + current_batch] = inp_np
                fp_labels[loaded_count : loaded_count + current_batch] = lbl_np
                f_bin.write(inp_np.tobytes())
                f_labels_bin.write(lbl_np.tobytes())

                loaded_count += current_batch
                print(f"  -> Streamed chunk to disk... {loaded_count}/{args.batch}", end='\r')

                if loaded_count >= args.batch:
                    break

        print() # Print a newline after carriage return progress
        fp_inputs.flush()
        fp_labels.flush()
        del fp_inputs, fp_labels

        if loaded_count < args.batch:
            raise RuntimeError(f"[FATAL] Requested {args.batch} images but only found {loaded_count}!")

        data_already_saved = True
        has_labels = True

    # Mode A: Process a real local image (In-Memory)

    elif args.image_path:
        if not HAS_VISION:
            raise ImportError("[Error] Pillow and torchvision are required to process real images.")

        print(f"[Input Processor] Processing local image: {args.image_path}")
        img = Image.open(args.image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.height),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img).unsqueeze(0)
        if args.batch > 1:
            input_tensor = input_tensor.repeat(args.batch, 1, 1, 1)
        print(f"  -> Applied vision transforms. Final shape: {list(input_tensor.shape)}")

    # Mode B: Generate dummy test data (In-Memory)

    else:
        print(f"[Input Processor] Generating dummy input tensor with shape {tensor_shape}...")
        torch.manual_seed(42)
        input_tensor = torch.rand(*tensor_shape) - 0.5
        print("  -> Dummy data generated.")

    # In-Memory Export (Skipped if streamed to disk)

    if not data_already_saved:
        if args.quantize:
            print(f"\n[Input Processor] Applying INT8 Quantization...")
            quantized_tensor = torch.round(input_tensor / args.scale) + args.zero_point
            quantized_tensor = torch.clamp(quantized_tensor, -128, 127).to(torch.int8)
            np_array = quantized_tensor.detach().cpu().numpy().astype(np.int8)
        else:
            np_array = input_tensor.detach().cpu().numpy().astype(np.float32)

        np_array.tofile(str(bin_path))
        np.save(str(npy_path), np_array)

    # Final Unified Console Output

    print(f"\n[SUCCESS] Input processing complete!")
    print(f"  - Input C++ Binary ({dtype_str}) saved to : {bin_path.resolve()}")
    print(f"  - Input Python Numpy ({dtype_str}) saved to: {npy_path.resolve()}")

    if has_labels:
        print(f"  - Labels Python Numpy saved to: {labels_npy_path.resolve()}")
        print(f"  - Labels C++ Binary saved to: {labels_bin_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or process input data for ZK-ML testing.")
    parser.add_argument('--target_dir', type=str, required=True, help="Directory to save the processed input files")
    parser.add_argument('--batch', type=int, default=1, help="Batch size (can be > 1)")
    parser.add_argument('--channels', type=int, default=3, help="Number of channels")
    parser.add_argument('--height', type=int, default=224, help="Image height")
    parser.add_argument('--width', type=int, default=224, help="Image width")
    parser.add_argument('--image_path', type=str, default=None, help="Path to a single real image file")
    parser.add_argument('--dataset', type=str, default=None, help="Name of torchvision dataset (e.g., cifar10, mnist)")
    parser.add_argument('--data_root', type=str, default='./frontend_dataset', help="Root directory for datasets")
    parser.add_argument('--quantize', action='store_true', help="Enable INT8 quantization for the output tensor")
    parser.add_argument('--scale', type=float, default=1.0, help="Quantization scale factor")
    parser.add_argument('--zero_point', type=int, default=0, help="Quantization zero point")
    args = parser.parse_args()

    process_and_save_input(args)