"""
Dependencies:
    pip install torch torchvision numpy

Arguments:
    --out_dir       : (Optional) Destination directory path for the exported model.
                      If not provided, it defaults to:
                      ../generated_model/<model_name>_<timestamp>/
    --vision_model  : (Mode 1) Name of a standard built-in torchvision model 
                      (e.g., vgg16_bn, resnet18).
    --custom_script : (Mode 2) Path to the user-defined .py file containing 
                      the model architecture.
    --model_class   : (Mode 2) Name of the core model class inside the 
                      user-defined .py file.
    --weights       : (Optional) Path to the user's pre-trained .pth weights 
                      file. If not provided, random initialized weights are used.

Usage Examples:

    Scenario 1: Export a standard torchvision model (Output to default timestamped dir)
        python extract.py --vision_model vgg16_bn

    Scenario 2: Export a custom model (Output to default timestamped dir)
        python extract.py \
            --custom_script my_awesome_net.py \
            --model_class SuperNet

    Scenario 3: Export a custom model with weights to a specific custom directory
        python extract.py \
            --custom_script /path/to/my_awesome_net.py \
            --model_class SuperNet \
            --weights /path/to/epoch_100.pth \
            --out_dir /absolute/path/to/custom_dir
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, Node, GraphModule
import json
import numpy as np
import argparse
import importlib.util
import sys
import ast
from pathlib import Path
from datetime import datetime
import copy

# 1. AST Security Checker

def is_script_safe_to_import(filepath: Path) -> tuple[bool, str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error in script: {e}"

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Pass)):
            continue
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        elif isinstance(node, ast.If):
            try:
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__' and
                    isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == '__main__'):
                    continue
            except Exception:
                pass
            return False, f"Unsafe top-level 'if' statement detected at line {node.lineno}. Only 'if __name__ == \"__main__\":' is allowed."
        else:
            return False, f"Unsafe top-level execution detected ({type(node).__name__} at line {node.lineno}). Only definitions and assignments are allowed at the module level."

    return True, "Safe"

# 2. Offline Graph Optimization Passes

def fuse_conv_bn_eval(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    assert not (conv.training or bn.training), "Fusion only works in eval mode."
    gamma = bn.weight
    beta = bn.bias
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    scale = gamma / torch.sqrt(var + eps)

    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
    )
    w_conv = conv.weight.clone()
    scale_reshaped = scale.view(-1, 1, 1, 1)
    fused_conv.weight.data = w_conv * scale_reshaped
    b_conv = conv.bias.clone() if conv.bias is not None else torch.zeros(conv.out_channels, device=w_conv.device)
    fused_conv.bias.data = (b_conv - mu) * scale + beta
    return fused_conv

def auto_fold_adaptive_pool_and_linear(model: nn.Module, dummy_input: torch.Tensor):
    """
    Adapt VGG/AlexNet-style models for static graph extraction.
    This folds AdaptiveAvgPool2d into the following linear layer when possible.
    """
    avgpool_module = None
    linear_module = None
    linear_parent = None
    linear_idx = -1
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.AdaptiveAvgPool2d):
            avgpool_module = module
            break

    linear_attr_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_module = module
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            linear_attr_name = name.rsplit('.', 1)[1] if '.' in name else name
            if parent_name:
                linear_parent = model.get_submodule(parent_name)
            break

    if avgpool_module is None or linear_module is None:
        return

    features_before_pool = None
    def hook_fn(m, inp, out):
        nonlocal features_before_pool
        features_before_pool = inp[0].shape

    handle = avgpool_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(dummy_input)
    handle.remove()

    _, C, H_in, W_in = features_before_pool
    target_size = avgpool_module.output_size
    if isinstance(target_size, int): target_size = (target_size, target_size)

    if target_size[0] == 1:
        return

    if H_in == 1 and W_in == 1 and target_size[0] > 1:
        print(f"\n[\033[1;35mZK-Adapter\033[0m] Detected Architecture Mismatch (1x1 -> {target_size[0]}x{target_size[0]}). Triggering SILKY WEIGHT FOLDING...")

        old_weight = linear_module.weight.data
        old_bias = linear_module.bias.data if linear_module.bias is not None else None
        out_features, in_features = old_weight.shape
        S = target_size[0]

        if in_features == C * S * S:
            print(f"  > Compressing Pretrained Weights: [{out_features}, {in_features}] -> [{out_features}, {C}]")
            reshaped_weight = old_weight.view(out_features, C, S, S)
            folded_weight = reshaped_weight.sum(dim=(2, 3))

            new_linear = nn.Linear(C, out_features, bias=(old_bias is not None))
            new_linear.weight.data = folded_weight
            if old_bias is not None:
                new_linear.bias.data = old_bias

            if linear_parent is not None:
                if isinstance(linear_parent, nn.Sequential) and linear_attr_name.isdigit():
                    linear_parent[int(linear_attr_name)] = new_linear
                else:
                    setattr(linear_parent, linear_attr_name, new_linear)
            else:
                setattr(model, linear_attr_name, new_linear)

    elif H_in == target_size[0] and W_in == target_size[1]:
        print(f"\n[\033[1;35mZK-Adapter\033[0m] Redundant AdaptiveAvgPool2d ({target_size[0]}x{target_size[0]}) detected. Stripping it from graph...")

    else:
        return

    for name, module in model.named_modules():
        if module is avgpool_module:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                attr_name = name
                parent = model

            setattr(parent, attr_name, nn.Identity())
            break

def fx_fold_batchnorm(model: nn.Module) -> GraphModule:
    model.eval()

    try:
        traced_model = symbolic_trace(model)
    except Exception as e:
        print("\n" + "="*60)
        print("[CRITICAL ERROR] Model Tracing Failed!")
        print("="*60)
        print("We detected dynamic control flow inside your model's forward() function.")
        print("This is usually caused by data-dependent 'if' statements or 'for' loops.")
        print("\nOur ZK Exporter strictly requires a STATIC computational graph.")
        print("Please refactor your forward() pass or ensure the model architecture is static.")
        print(f"\n[Detailed PyTorch Trace Error]:\n{e}")
        print("="*60)
        sys.exit(1)

    modules = dict(traced_model.named_modules())

    for node in traced_model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], nn.BatchNorm2d):
            prev_node = node.args[0]
            if prev_node.op == 'call_module' and isinstance(modules[prev_node.target], nn.Conv2d):
                conv_name = prev_node.target
                bn_name = node.target
                print(f"[Optimizer] Fusing BN: {conv_name} -> {bn_name}")

                fused_conv = fuse_conv_bn_eval(modules[conv_name], modules[bn_name])
                setattr(traced_model, conv_name, fused_conv)
                modules[conv_name] = fused_conv

                node.replace_all_uses_with(prev_node)
                traced_model.graph.erase_node(node)

    traced_model.recompile()
    return traced_model

def fx_fold_scale_avgpool_linear(model: GraphModule) -> GraphModule:
    modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if node.op == 'call_module' and isinstance(modules[node.target], nn.AdaptiveAvgPool2d):
            target_mod = modules[node.target]
            out_size = target_mod.output_size

            is_gap = (out_size == (1, 1) or out_size == 1 or out_size == [1, 1])
            if not is_gap:
                continue

            can_fold = False
            linear_node = None
            curr_nodes = list(node.users.keys())

            while len(curr_nodes) == 1:
                nxt = curr_nodes[0]
                is_shape_op = False

                if nxt.op == 'call_function':
                    fname = getattr(nxt.target, '__name__', str(nxt.target))
                    if fname in ['flatten', 'reshape', 'squeeze']: is_shape_op = True
                elif nxt.op == 'call_method':
                    mname = str(nxt.target)
                    if mname in ['flatten', 'view', 'reshape', 'squeeze']: is_shape_op = True
                elif nxt.op == 'call_module' and isinstance(modules.get(nxt.target), nn.Flatten):
                    is_shape_op = True

                if is_shape_op:
                    curr_nodes = list(nxt.users.keys())
                elif nxt.op == 'call_module' and isinstance(modules.get(nxt.target), nn.Linear):
                    linear_node = nxt
                    can_fold = True
                    break
                else:
                    break

            if can_fold:
                area = 1.0
                prev_node = node.args[0]
                tensor_meta = prev_node.meta.get('tensor_meta')

                if tensor_meta is not None:
                    if hasattr(tensor_meta, 'shape') and not isinstance(tensor_meta, torch.Size):
                        in_shape = tensor_meta.shape
                    elif isinstance(tensor_meta, (tuple, list)) and len(tensor_meta) > 0 and hasattr(tensor_meta[0], 'shape'):
                        in_shape = tensor_meta[0].shape
                    else:
                        in_shape = tensor_meta

                    if hasattr(in_shape, '__len__') and len(in_shape) >= 4:
                        area = float(in_shape[2] * in_shape[3])

                if area > 1.0:
                    linear_mod = modules[linear_node.target]
                    if not hasattr(linear_mod, '_folded_area'):
                        print(f"[\033[1;32mScale Folding\033[0m] Scaling Bias of Linear '{linear_node.name}' by Area: {area}")

                        if getattr(linear_mod, 'bias', None) is not None:
                            linear_mod.bias.data = linear_mod.bias.data * area
                        linear_mod._folded_area = area

                        target_mod._folded_to_sum = True
                        target_mod._pool_area = area

    return model

# 3. Graph Topology & Weights Exporter

def export_zk_model(model: GraphModule, export_dir_str: str):
    export_dir = Path(export_dir_str)
    export_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = export_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    modules = dict(model.named_modules())
    zk_graph_ir = []

    for node in model.graph.nodes:
        if node.op == 'placeholder':
            zk_graph_ir.append({"name": node.name, "type": "input", "inputs": []})

        elif node.op == 'call_module':
            target_mod = modules[node.target]

            if isinstance(target_mod, nn.ReLU):
                users = list(node.users.keys())
                if len(users) == 1 and users[0].op == 'call_module' and isinstance(modules[users[0].target], nn.MaxPool2d):
                    print(f"[Optimizer] Skipping ReLU '{node.name}', merging with MaxPool2d.")
                    continue
                else:
                    zk_graph_ir.append({"name": node.name, "type": "relu", "inputs": [str(node.args[0])]})

            elif isinstance(target_mod, nn.MaxPool2d):
                prev_node = node.args[0]
                is_fused = False

                if prev_node.op == 'call_module' and isinstance(modules[prev_node.target], nn.ReLU) and len(prev_node.users) == 1:
                    is_fused = True
                    prev_prev_node = prev_node.args[0]
                    print(f"[Optimizer] Exporting fused '{node.name}' as integrated_nl.")
                    zk_graph_ir.append({
                        "name": node.name, "type": "integrated_nl",
                        "inputs": [str(prev_prev_node)],
                        "kernel_size": target_mod.kernel_size,
                        "stride": target_mod.stride,
                        "padding": target_mod.padding
                    })

                if not is_fused:
                    zk_graph_ir.append({
                        "name": node.name, "type": "max_pool2d", "inputs": [str(node.args[0])],
                        "kernel_size": target_mod.kernel_size, "stride": target_mod.stride
                    })

            elif isinstance(target_mod, (nn.Conv2d, nn.Linear)):
                node_type = type(target_mod).__name__.lower()
                node_info = {"name": node.name, "type": node_type, "inputs": [str(node.args[0])]}

                if isinstance(target_mod, nn.Conv2d):
                    node_info["stride"] = target_mod.stride
                    node_info["padding"] = target_mod.padding

                if hasattr(target_mod, 'weight') and target_mod.weight is not None:
                    node_info["weight_shape"] = list(target_mod.weight.shape)
                    w_path = weights_dir / f"{node.name}_weight.bin"
                    target_mod.weight.detach().cpu().numpy().astype(np.float32).tofile(str(w_path))

                if hasattr(target_mod, 'bias') and target_mod.bias is not None:
                    node_info["bias_shape"] = list(target_mod.bias.shape)
                    b_path = weights_dir / f"{node.name}_bias.bin"
                    target_mod.bias.detach().cpu().numpy().astype(np.float32).tofile(str(b_path))

                zk_graph_ir.append(node_info)

            elif isinstance(target_mod, nn.AdaptiveAvgPool2d):
                out_size = target_mod.output_size
                is_gap = (out_size == (1, 1) or out_size == 1 or out_size == [1, 1])

                if is_gap:
                    if getattr(target_mod, '_folded_to_sum', False):
                        area_val = getattr(target_mod, '_pool_area', 1.0)

                        node_info = {"name": node.name, "type": "global_sum_pool2d", "inputs": [str(node.args[0])], "pool_area": area_val}
                    else:
                        node_info = {"name": node.name, "type": "global_avg_pool2d", "inputs": [str(node.args[0])]}
                    zk_graph_ir.append(node_info)
                else:
                    print(f"\n[CRITICAL ERROR] AdaptiveAvgPool2d with output_size={out_size} is detected!")
                    print("The ZK C++ backend currently ONLY supports AdaptiveAvgPool2d when used as a Global Average Pooling (output_size=1).")
                    print("Please modify your PyTorch model architecture.")
                    sys.exit(1)

            else:
                module_type = type(target_mod).__name__.lower()
                print(f"[Warning] Exporting generic/unsupported module: '{node.name}' (Type: {module_type})")

                node_info = {
                    "name": node.name,
                    "type": f"module_{module_type}",
                    "inputs": [str(arg) for arg in node.args if isinstance(arg, Node)],
                    "status": "UNCHECKED",
                    "param_shapes": {}
                }

                for param_name, param_tensor in target_mod.named_parameters():
                    if param_tensor is not None:
                        node_info["param_shapes"][param_name] = list(param_tensor.shape)
                        bin_filename = f"{node.name}_{param_name.replace('.', '_')}.bin"
                        bin_path = weights_dir / bin_filename
                        param_tensor.detach().cpu().numpy().astype(np.float32).tofile(str(bin_path))

                zk_graph_ir.append(node_info)

        elif node.op == 'call_function':
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            if func_name == "flatten":
                zk_graph_ir.append({"name": node.name, "type": "flatten", "inputs": [str(node.args[0])]})
            else:
                zk_graph_ir.append({
                    "name": node.name, "type": f"func_{func_name}",
                    "inputs": [str(arg) for arg in node.args if isinstance(arg, Node)],
                    "args_const": [str(arg) for arg in node.args if not isinstance(arg, Node)],
                    "kwargs": {k: str(v) for k, v in node.kwargs.items()}
                })

        elif node.op == 'call_method':
            method_name = str(node.target)
            tensor_input = node.args[0]
            other_args = [str(arg) for arg in node.args[1:] if not isinstance(arg, Node)]
            zk_graph_ir.append({
                "name": node.name, "type": f"method_{method_name}",
                "inputs": [str(tensor_input)] if isinstance(tensor_input, Node) else [],
                "method_args": other_args
            })

        elif node.op == 'output':
            out_input = node.args[0][0] if isinstance(node.args[0], tuple) else node.args[0]
            zk_graph_ir.append({"name": node.name, "type": "output", "inputs": [str(out_input)]})

    json_path = export_dir / "graph.json"
    with open(json_path, "w") as f:
        json.dump(zk_graph_ir, f, indent=4)

    print(f"\n[Success] Model exported successfully to directory: {export_dir.resolve()}")

# 4. Main CLI Entry Point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZK-ML Model Exporter (AST-Secured)")
    parser.add_argument('--vision_model', type=str, help="Name of torchvision model (e.g., resnet18)")
    parser.add_argument('--custom_script', type=str, help="Path to your python model script (e.g., my_net.py)")
    parser.add_argument('--model_class', type=str, help="Class name in your script (e.g., MyCustomNet)")
    parser.add_argument('--weights', type=str, help="Path to the .pth state_dict file", default=None)
    parser.add_argument('--out_dir', type=str, default=None, help="Optional specific output directory")
    parser.add_argument('--batch', type=int, default=1, help="Input batch size (Default: 1)")
    parser.add_argument('--channels', type=int, default=3, help="Input channels (Default: 3)")
    parser.add_argument('--height', type=int, default=32, help="Input height (Default: 32)")
    parser.add_argument('--width', type=int, default=32, help="Input width (Default: 32)")

    args = parser.parse_args()
    model = None
    model_name = "unknown_model"

    if args.vision_model:
        import torchvision.models as models
        print(f"Loading standard torchvision model: {args.vision_model}")
        model_fn = getattr(models, args.vision_model)
        model = model_fn(weights='DEFAULT')
        model_name = args.vision_model

    elif args.custom_script and args.model_class:
        script_path = Path(args.custom_script).resolve()
        is_safe, msg = is_script_safe_to_import(script_path)
        if not is_safe:
            print(f"\n[SECURITY ALERT] Import Rejected!")
            sys.exit(1)

        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        spec = importlib.util.spec_from_file_location("user_module", str(script_path))
        user_module = importlib.util.module_from_spec(spec)
        sys.modules["user_module"] = user_module
        spec.loader.exec_module(user_module)

        if sys.path[0] == script_dir:
            sys.path.pop(0)

        model_class = getattr(user_module, args.model_class)
        model = model_class()
        model_name = args.model_class
    else:
        print("[Error] Invalid arguments. Specify either --vision_model OR (--custom_script AND --model_class)")
        sys.exit(1)

    if args.weights:
        print(f"Loading weights from {args.weights}...")
        model.load_state_dict(torch.load(args.weights, map_location='cpu'))

    print("\n[Step 1] Initializing Graph Optimization...")
    # We change the batchsize to 1

    dummy_in = torch.randn(1, args.channels, args.height, args.width)
    auto_fold_adaptive_pool_and_linear(model, dummy_in)
    fused_net = fx_fold_batchnorm(model)

    print("[Step 1.5] Propagating shapes for Scale Folding Engine...")
    from torch.fx.passes.shape_prop import ShapeProp
    try:
        # We change the batchsize to 1

        dummy_in = torch.randn(1, args.channels, args.height, args.width)
        ShapeProp(fused_net).propagate(dummy_in)
    except Exception as e:
        print(f"[Warning] Shape propagation failed: {e}")

    original_state_dict = copy.deepcopy(model.state_dict())

    print("[Step 1.6] Applying Scale Folding (AvgPool -> Linear)...")
    fused_net = fx_fold_scale_avgpool_linear(fused_net)

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(__file__).resolve().parent.parent / "generated_model"
        final_out_dir = str(base_dir / f"{model_name}_{timestamp}")
    else:
        final_out_dir = args.out_dir

    print(f"\n[Step 2] Exporting JSON graph and .bin weights to {final_out_dir}...")
    export_zk_model(fused_net, export_dir_str=final_out_dir)

    pth_path = Path(final_out_dir) / "pytorch_weights.pth"
    torch.save(original_state_dict, pth_path)
    print(f"  - PyTorch Native Weights saved to: {pth_path}")

    meta_data = {
        "model_name": model_name,
        "N": args.batch,
        "C": args.channels,
        "H": args.height,
        "W": args.width
    }
    meta_path = Path(final_out_dir) / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=4)
    print(f"  - \033[1;36mMetadata SSOT\033[0m saved to: {meta_path}")
    print("\033[1;32m[SUCCESS] Exporter Pipeline Completed Perfectly!\033[0m\n")
