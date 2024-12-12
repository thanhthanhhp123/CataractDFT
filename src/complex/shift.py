import torch
import numpy as np
import itertools

def roll(a, shift, axis=None):

    a = np.asanyarray(a)
    if axis is None:
        return roll(a.ravel(), shift, 0).reshape(a.shape)

    else:
        axis = np.core.multiarray.normalize_axis_index(axis, a.ndim)
        broadcasted = np.broadcast_to(shift, np.array(axis).shape)
        if broadcasted.ndim > 1:
            raise ValueError(
                "'shift' and 'axis' should be scalars or 1D sequences")
        shifts = {ax: 0 for ax in range(a.ndim)}
        for sh, ax in broadcasted:
            shifts[ax] += sh

        rolls = [((slice(None), slice(None)),)] * a.ndim
        for ax, offset in shifts.items():
            offset %= a.shape[ax] or 1  # If `a` is empty, nothing matters.
            if offset:
                # (original, result), (original, result)
                rolls[ax] = ((slice(None, -offset), slice(offset, None)),
                             (slice(-offset, None), slice(None, offset)))

        result = np.empty_like(a)
        for indices in itertools.product(*rolls):
            arr_index, res_index = zip(*indices)
            result[res_index] = a[arr_index]

        return result

def roll_n(X, axis, n):
    axis = (axis + X.ndim)%X.ndim
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,-n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(-n,None,None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(x, axes=None):
 
    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[ax] // 2 for ax in axes]
    
    if isinstance(axes, int):
        x = roll_n(x, axes, shift)
    else:
        for i in range(len(shift)):
            x = roll_n(x, axes[i], shift[i])
    
    return x

def ifftshift(x, axes=None):

    if axes is None:
        axes = tuple(range(x.ndim))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[ax] // 2) for ax in axes]

    if isinstance(axes, int):
        x = roll_n(x, axes, shift)
    else:
        for i in range(len(shift)):
            x = roll_n(x, axes[i], shift[i])
    
    return x


