import torch 

along_dim = lambda shp, dim: tuple(1 if d==dim%len(shp) else s 
                                   for d,s in enumerate(shp))

def pad_val_after(x: torch.Tensor, dim: int = -1, val: float = 1.0):
    return torch.cat(
        [
            x, 
            torch.tensor(val).broadcast_to(along_dim(shp=x.shape, dim=dim))
        ], 
        dim=dim)