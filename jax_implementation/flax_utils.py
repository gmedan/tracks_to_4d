from typing import Any, Optional
from einops.layers.flax import EinMix as Mix

def EinMix(pattern: str, 
           weight_shape: str, 
           bias_shape: Optional[str] = None, 
           **axes_lengths: Any) -> Mix:
    
    return Mix(pattern=pattern,
               weight_shape=weight_shape,
               bias_shape=bias_shape,
               sizes=axes_lengths)
