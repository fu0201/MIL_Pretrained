import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Import base model and registry
from .base_model import BaseModel, MODEL_REGISTRY
from .model_registry import get_model_output_dim, get_model_hf_path

# Import all model modules to register them
from .conch.conch_v1_model import CONCHModel
from .conch.conch_v1_5_model import CONCHV1_5Model

def create_patch_encoder(model_name, checkpoint_path=None, local_dir=False):
    """
    Create a pathology foundation model.

    Args:
        model_name (str): Name of the model to create
        checkpoint_path (str, optional): Path to model checkpoint. Defaults to None.
        local_dir (bool, optional): Whether checkpoint_path is a local directory. Defaults to False.

    Returns:
        BaseModel: Initialized model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(checkpoint_path=checkpoint_path, local_dir=local_dir)

    return model


# Export key functions and classes
__all__ = [
    'create_patch_encoder',
    'get_model_output_dim', 
    'get_model_hf_path',
    'BaseModel',
    'MODEL_REGISTRY'
] 