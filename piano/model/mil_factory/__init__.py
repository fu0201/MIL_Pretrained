"""
MIL Factory - Multiple Instance Learning Models Factory

This module provides a unified interface for creating MIL baseline models.
Uses lazy loading to avoid import failures affecting other models.
"""

import torch
import torch.nn as nn

# ============================================================================
# Part 1: Available Models Registry (Lazy Loading)
# ============================================================================

# Available model names for lazy loading
AVAILABLE_MODELS = [
    'abmil', 'gated_abmil', 'abmil_distill', 'gated_abmil_distill',
    'clam_sb', 'clam_mb', 'clam_sb_distill', 'clam_mb_distill',
    'transmil', 'transmil_distill', 'wikg', 'wikg_distill',
    'amdmil', 'amdmil_distill',
    '2dmamba', '2dmamba_distill',
    'aemmil', 'aemmil_distill', 
    'dagmil', 'dagmil_distill', 'gdfmil', 'gdfmil_distill',
]

def _lazy_load_model(model_name):
    """
    Lazy load model class based on model name
    
    Args:
        model_name (str): Name of the model to load
        
    Returns:
        class: Model class
        
    Raises:
        ImportError: If model cannot be imported
        ValueError: If model name is not recognized
    """
    if model_name == 'abmil':
        from .abmil.abmil import ABMIL
        return ABMIL
    elif model_name == 'gated_abmil':
        from .abmil.abmil import GatedABMIL
        return GatedABMIL
    elif model_name == 'abmil_distill':
        from .abmil.abmil_distill import ABMILDistill
        return ABMILDistill
    elif model_name == 'gated_abmil_distill':
        from .abmil.abmil_distill import GatedABMILDistill
        return GatedABMILDistill
    elif model_name == 'clam_sb':
        from .clam.clam import CLAM_SB
        return CLAM_SB
    elif model_name == 'clam_mb':
        from .clam.clam import CLAM_MB
        return CLAM_MB
    elif model_name in ('clam_distill', 'clam_sb_distill'):
        from .clam.clam_distill import CLAMDistill
        return CLAMDistill
    elif model_name == 'clam_mb_distill':
        from .clam.clam_distill import CLAMMBDistill
        return CLAMMBDistill
    elif model_name == 'transmil':
        from .transmil.transmil import TransMIL
        return TransMIL
    elif model_name == 'transmil_distill':
        from .transmil.transmil_distill import TransMILDistill
        return TransMILDistill
    elif model_name == 'wikg':
        from .wikg.wikg import WiKG
        return WiKG
    elif model_name == 'wikg_distill':
        from .wikg.wikg_distill import WiKGDistill
        return WiKGDistill
    elif model_name == 'amdmil':
        from .amdmil.amdmil import AMD_MIL
        return AMD_MIL
    elif model_name == 'amdmil_distill':
        from .amdmil.amdmil_distill import AMDMILDistill
        return AMDMILDistill
    elif model_name == '2dmamba':
        from .mamba_2d.mamba_2d import MambaMIL_2D
        return MambaMIL_2D
    elif model_name == 'aemmil':
        from .aemmil.aemmil import AEM_MIL
        return AEM_MIL
    elif model_name == 'aemmil_distill':
        from .aemmil.aemmil_distill import AEMMILDistill
        return AEMMILDistill
    elif model_name == '2dmamba_distill':
        from .mamba_2d.mamba_2d_distill import MambaMIL2DDistill
        return MambaMIL2DDistill
    elif model_name == 'dagmil':
        from .dagmil.dagmil import DeformableGraphGNN
        return DeformableGraphGNN
    elif model_name == 'dagmil_distill':
        from .dagmil.dagmil_distill import DAGMILDistill
        return DAGMILDistill
    elif model_name == 'gdfmil':
        from .gdfmil.gdfmil import GDF_MIL
        return GDF_MIL
    elif model_name == 'gdfmil_distill':
        from .gdfmil.gdfmil_distill import GDFMILDistill
        return GDFMILDistill
    else:
        raise ValueError(f"Unknown model: {model_name}. Available models: {AVAILABLE_MODELS}")

# ============================================================================
# Part 2: Model Registry (For compatibility)
# ============================================================================

class LazyModelRegistry:
    """Lazy model registry that loads models on demand"""
    
    def __contains__(self, key):
        return key in AVAILABLE_MODELS
    
    def __getitem__(self, key):
        return _lazy_load_model(key)
    
    def keys(self):
        return AVAILABLE_MODELS

# Create registry instance
MIL_MODEL_REGISTRY = LazyModelRegistry()

# ============================================================================
# Part 3: Default Parameters for each model
# ============================================================================

MIL_DEFAULT_PARAMS = {
    'abmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'gated_abmil': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'abmil_distill': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'gated_abmil_distill': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 2, 'dropout': 0.25, 'survival': False},
    'clam_sb': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 2, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False},
    'clam_mb': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 2, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False},
    'clam_sb_distill': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 0, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False},
    'clam_mb_distill': {'dim_in': 1024, 'dim_hidden': 512, 'dropout': 0.25, 'num_classes': 0, 'k_sample': 8, 'instance_loss_fn': None, 'subtyping': False, 'survival': False, 'distill_branches': 2},
    'transmil': {'dim_in': 1024, 'dim_hidden': 1024, 'num_classes': 2, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.25, 'survival': False},
    'transmil_distill': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 0, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.25, 'survival': False},
    'wikg': {'dim_in': 1024, 'dim_hidden': 1024, 'num_classes': 2, 'topk': 6, 'agg_type': 'bi-interaction', 'dropout': 0.3, 'pool': 'attn', 'survival': False},
    'wikg_distill': {'dim_in': 1024, 'dim_hidden': 512, 'num_classes': 0, 'topk': 6, 'agg_type': 'bi-interaction', 'dropout': 0.3, 'pool': 'attn', 'survival': False},
    'amdmil': {'dim_in': 1024, 'embed_dim': 512, 'num_classes': 10, 'agent_num': 256, 'survival': False}, 
    'amdmil_distill': {'dim_in': 1024, 'embed_dim': 512, 'num_classes': 0, 'agent_num': 256, 'dropout': 0.25, 'survival': False},
    '2dmamba': {'dim_in': 1024, 'drop_out': 0.25, 'num_classes': 2, 'survival': False, 'pos_emb_type': None},
    'aemmil': {'dim_in': 1024, 'L': 512, 'D': 128, 'num_classes': 2, 'dropout': 0.1, 'temperature': 1.0, 'lambda_entropy': 0.1},
    'aemmil_distill': {'dim_in': 1024, 'L': 512, 'D': 128, 'num_classes': 0, 'dropout': 0.1, 'temperature': 1.0, 'lambda_entropy': 0.1},
    '2dmamba_distill': {'dim_in': 1024, 'drop_out': 0.25, 'num_classes': 0, 'survival': False, 'pos_emb_type': None},
    'dagmil': {'dim_in': 1024, 'dim_hidden': 256, 'n_classes': 2, 'topk': 6, 'stride': 512, 'agg_type': 'bi-interaction'},
    'dagmil_distill': {'dim_in': 1024, 'dim_hidden': 256, 'num_classes': 0, 'topk': 6, 'stride': 512, 'agg_type': 'bi-interaction', 'dropout': 0.25},
    'gdfmil': {'dim_in': 1024, 'num_classes': 2, 'hid_dim': 256, 'out_dim': 128, 'k_components': 10, 'k_neighbors': 10, 'dropout': 0.1, 'act': 'leaky_relu'},
    'gdfmil_distill': {'dim_in': 1024, 'num_classes': 0, 'hid_dim': 256, 'out_dim': 128, 'k_components': 10, 'k_neighbors': 10, 'dropout': 0.1, 'act': 'leaky_relu'},
}

# ============================================================================
# Part 4: Model Creation Function
# ============================================================================

def create_mil_model(model_name, **kwargs):
    """
    Create MIL model using lazy loading
    
    Args:
        model_name (str): Model name
        **kwargs: Model parameters that will override default parameters
        
    Returns:
        torch.nn.Module: Created model instance
        
    Raises:
        ValueError: If model_name is not recognized
        ImportError: If model cannot be imported
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {AVAILABLE_MODELS}")
    
    try:
        # Lazy load the model class
        model_class = _lazy_load_model(model_name)
    except ImportError as e:
        raise ImportError(f"Failed to import model '{model_name}': {e}")
    
    # Get default parameters for the model
    if model_name in MIL_DEFAULT_PARAMS:
        default_kwargs = MIL_DEFAULT_PARAMS[model_name].copy()
    else:
        # Fallback for any models not in default params
        default_kwargs = {'dim_in': 1024, 'num_classes': 2, 'survival': False}
    
    # Update default parameters with passed kwargs
    default_kwargs.update(kwargs)
    
    return model_class(**default_kwargs)


def get_mil_model_names():
    """
    Get all available MIL model names
    
    Returns:
        list: List of available model names
    """
    return AVAILABLE_MODELS.copy()


def get_mil_default_params(mil_name):
    """
    Get default parameters for a specific MIL model
    
    Args:
        mil_name (str): Model name
        
    Returns:
        dict: Default parameters for the model
        
    Raises:
        ValueError: If mil_name is not recognized
    """
    if mil_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {mil_name}. Available models: {AVAILABLE_MODELS}")
    
    if mil_name in MIL_DEFAULT_PARAMS:
        return MIL_DEFAULT_PARAMS[mil_name].copy()
    else:
        return {'dim_in': 1024, 'num_classes': 2, 'survival': False}


# ============================================================================
# Part 5: Export
# ============================================================================

__all__ = [
    'create_mil_model',
    'get_mil_model_names', 
    'get_mil_default_params',
    'MIL_MODEL_REGISTRY',
    'MIL_DEFAULT_PARAMS'
] 
