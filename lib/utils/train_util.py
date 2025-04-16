import importlib

import os

from pytorch_lightning.utilities import rank_zero_only
@rank_zero_only
def main_print(*args):
    print(*args)



def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        main_print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    main_print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_class_from_config(config):
    class_path = config.get("target")
    if not class_path:
        raise KeyError("Expected key `target` to instantiate.")

    if "." not in class_path:
        raise ValueError(f"Invalid class path: '{class_path}'. Expected format 'module.submodule.ClassName'")
    
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        raise ImportError(f"Module '{module_path}' not found") from e
    
    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
    
    return getattr(module, class_name)  # 返回类对象
