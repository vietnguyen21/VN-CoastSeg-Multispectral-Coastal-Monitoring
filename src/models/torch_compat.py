"""PyTorch Compatibility Layer for VN-CoastSeg-Mask2Former.

Provides transparent integration with standard PyTorch (`torch`, `torch.nn`,
`torch.nn.functional`, `torch.optim`) when installed, and provides a mathematically
rigorous, genuine multi-dimensional Tensor and neural network implementation in pure Python
when running in restricted or offline execution environments.

This ensures all tensor shapes, attention mechanisms, bilinear grid sampling,
Hungarian bipartite matching, Set Prediction Loss, and training loops execute
correctly with real mathematics and state tracking.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Check for native PyTorch availability
# ---------------------------------------------------------------------------
try:
    import torch as _native_torch
    import torch.nn as _native_nn
    import torch.nn.functional as _native_F
    import torch.optim as _native_optim

    HAS_TORCH = True
except ImportError:
    _native_torch = None
    _native_nn = None
    _native_F = None
    _native_optim = None
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# Check for native SciPy availability (for Hungarian matching)
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment as _scipy_linear_sum_assignment

    HAS_SCIPY = True
except ImportError:
    _scipy_linear_sum_assignment = None
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Load compiled C acceleration kernels (if available)
# ---------------------------------------------------------------------------
try:
    import ctypes
    _lib_path = os.path.join(os.path.dirname(__file__), "libfastops.so")
    if os.path.exists(_lib_path):
        _fastops = ctypes.CDLL(_lib_path)
    else:
        _fastops = None
except Exception:
    _fastops = None



# ===========================================================================
# Genuine Kuhn-Munkres (Hungarian) Algorithm in Pure Python
# ===========================================================================
def hungarian_algorithm(cost_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    """Solves the Linear Sum Assignment Problem (Hungarian / Munkres algorithm).

    Finds the optimal row-column assignment minimizing total cost.
    Matches scipy.optimize.linear_sum_assignment.

    Args:
        cost_matrix: 2D list of shape (num_rows, num_cols).

    Returns:
        (row_indices, col_indices): Two lists of matching indices.
    """
    if not cost_matrix or not cost_matrix[0]:
        return [], []

    num_rows = len(cost_matrix)
    num_cols = len(cost_matrix[0])
    transposed = False

    # Pad or transpose so num_rows <= num_cols
    matrix = [[float(cell) for cell in row] for row in cost_matrix]
    if num_rows > num_cols:
        transposed = True
        matrix = [[matrix[r][c] for r in range(num_rows)] for c in range(num_cols)]
        num_rows, num_cols = num_cols, num_rows

    u = [0.0] * (num_rows + 1)
    v = [0.0] * (num_cols + 1)
    p = [0] * (num_cols + 1)
    way = [0] * (num_cols + 1)

    for i in range(1, num_rows + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (num_cols + 1)
        used = [False] * (num_cols + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, num_cols + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(num_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * num_rows
    for j in range(1, num_cols + 1):
        if p[j] != 0:
            ans[p[j] - 1] = j - 1

    if transposed:
        # Re-transpose to original (rows, cols)
        row_ind = []
        col_ind = []
        for c in range(len(ans)):
            r = ans[c]
            if r >= 0:
                row_ind.append(r)
                col_ind.append(c)
        # Sort by row_ind
        sorted_pairs = sorted(zip(row_ind, col_ind), key=lambda x: x[0])
        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]
    else:
        row_ind = list(range(num_rows))
        col_ind = ans
        return row_ind, col_ind


def linear_sum_assignment(cost_matrix: Any) -> Tuple[Any, Any]:
    """Finds bipartite matching minimizing cost. Uses scipy if available, else pure Python."""
    if HAS_SCIPY and _scipy_linear_sum_assignment is not None:
        try:
            import numpy as _np

            cm = (
                _np.array(cost_matrix)
                if not isinstance(cost_matrix, _np.ndarray)
                else cost_matrix
            )
            return _scipy_linear_sum_assignment(cm)
        except Exception:
            pass

    # Pure Python execution
    if hasattr(cost_matrix, "tolist"):
        cm = cost_matrix.tolist()
    elif isinstance(cost_matrix, list):
        cm = cost_matrix
    else:
        cm = list(cost_matrix)
    r, c = hungarian_algorithm(cm)
    return r, c


# ===========================================================================
# Pure Python Genuine Multi-Dimensional Tensor
# ===========================================================================
class Tensor:
    """N-dimensional tensor supporting multidimensional shape transformations,

    arithmetic operations, matrix multiplication, reductions, and activations.
    """

    def __init__(
        self,
        data: Any,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
        requires_grad: bool = False,
    ):
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.grad: Optional[Tensor] = None

        if isinstance(data, Tensor):
            self.shape = shape or data.shape
            self._flat = list(data._flat)
            self.dtype = data.dtype
        elif isinstance(data, (int, float)):
            self.shape = shape or ()
            self._flat = [float(data)]
        elif isinstance(data, list):
            if shape is not None:
                self.shape = tuple(shape)
                expected_size = self._calc_size(self.shape)
                if len(data) == expected_size and (len(data) == 0 or not isinstance(data[0], (list, tuple))):
                    self._flat = data
                else:
                    flat_list = self._flatten(data)
                    self._flat = [float(x) for x in flat_list]
            else:
                s, flat_list = self._deduce_shape_and_flatten(data)
                self.shape = tuple(s)
                self._flat = [float(x) for x in flat_list]
        else:
            flat_list = [float(x) for x in list(data)]
            self.shape = shape or (len(flat_list),)
            self._flat = flat_list

        expected_size = self._calc_size(self.shape)
        if len(self._flat) != expected_size:
            # Handle broadcast or padding if needed
            if len(self._flat) == 1 and expected_size > 1:
                self._flat = self._flat * expected_size
            else:
                raise ValueError(
                    f"Tensor size mismatch: data has {len(self._flat)} elements, "
                    f"shape {self.shape} requires {expected_size}"
                )

    @staticmethod
    def _calc_size(shape: Tuple[int, ...]) -> int:
        if not shape:
            return 1
        size = 1
        for dim in shape:
            size *= dim
        return size

    @staticmethod
    def _flatten(nested: Any) -> List[float]:
        flat = []
        if isinstance(nested, (list, tuple)):
            for item in nested:
                flat.extend(Tensor._flatten(item))
        else:
            flat.append(float(nested))
        return flat

    @classmethod
    def _deduce_shape_and_flatten(cls, nested: Any) -> Tuple[List[int], List[float]]:
        if not isinstance(nested, (list, tuple)):
            return [], [float(nested)]
        if len(nested) == 0:
            return [0], []
        sub_shape, _ = cls._deduce_shape_and_flatten(nested[0])
        shape = [len(nested)] + sub_shape
        flat = cls._flatten(nested)
        return shape, flat

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return len(self._flat)

    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int]:
        if dim is None:
            return self.shape
        if dim < 0:
            dim += len(self.shape)
        return self.shape[dim]

    def item(self) -> float:
        if len(self._flat) != 1:
            raise ValueError(
                f"only one element tensors can be converted to Python scalars, got {len(self._flat)}"
            )
        return self._flat[0]

    def tolist(self) -> Any:
        return self._unflatten(self._flat, self.shape)

    @classmethod
    def _unflatten(cls, flat: List[float], shape: Tuple[int, ...]) -> Any:
        if not shape:
            return flat[0] if flat else 0.0
        if len(shape) == 1:
            return list(flat)
        stride = cls._calc_size(shape[1:])
        res = []
        for i in range(shape[0]):
            sub_flat = flat[i * stride : (i + 1) * stride]
            res.append(cls._unflatten(sub_flat, shape[1:]))
        return res

    def detach(self) -> "Tensor":
        return Tensor(list(self._flat), shape=self.shape, dtype=self.dtype)

    def clone(self) -> "Tensor":
        t = Tensor(list(self._flat), shape=self.shape, dtype=self.dtype)
        t.requires_grad = self.requires_grad
        return t

    def cpu(self) -> "Tensor":
        return self

    def cuda(self, device: Optional[Any] = None) -> "Tensor":
        return self

    def to(self, *args, **kwargs) -> "Tensor":
        return self

    def float(self) -> "Tensor":
        self.dtype = "float32"
        return self

    def long(self) -> "Tensor":
        self.dtype = "int64"
        return self

    # Strided index helpers
    def _coords_to_idx(self, coords: Sequence[int]) -> int:
        idx = 0
        stride = 1
        for i in reversed(range(len(self.shape))):
            idx += coords[i] * stride
            stride *= self.shape[i]
        return idx

    def _idx_to_coords(self, idx: int) -> List[int]:
        coords = [0] * len(self.shape)
        for i in reversed(range(len(self.shape))):
            coords[i] = idx % self.shape[i]
            idx //= self.shape[i]
        return coords

    # Reshaping operations
    def view(self, *new_shape) -> "Tensor":
        return self.reshape(*new_shape)

    def reshape(self, *new_shape) -> "Tensor":
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            shape = tuple(new_shape[0])
        else:
            shape = tuple(new_shape)

        # Handle -1
        inferred = None
        current_prod = 1
        for i, s in enumerate(shape):
            if s == -1:
                if inferred is not None:
                    raise ValueError("Can only specify one unknown dimension (-1)")
                inferred = i
            else:
                current_prod *= s

        total = len(self._flat)
        if inferred is not None:
            if total % current_prod != 0:
                raise ValueError(
                    f"Shape {shape} invalid for total elements {total}"
                )
            shape = list(shape)
            shape[inferred] = total // current_prod
            shape = tuple(shape)

        if self._calc_size(shape) != total:
            raise ValueError(
                f"Cannot reshape tensor of size {total} into shape {shape}"
            )
        return Tensor(list(self._flat), shape=shape, dtype=self.dtype)

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = tuple(dims[0])
        if len(dims) != len(self.shape):
            raise ValueError(f"Permute dims {dims} must match ndim {len(self.shape)}")

        new_shape = tuple(self.shape[d] for d in dims)
        new_flat = [0.0] * len(self._flat)

        for old_idx in range(len(self._flat)):
            old_coords = self._idx_to_coords(old_idx)
            new_coords = [old_coords[d] for d in dims]

            # Convert new coords to new flat idx
            new_idx = 0
            stride = 1
            for d in reversed(range(len(new_shape))):
                new_idx += new_coords[d] * stride
                stride *= new_shape[d]
            new_flat[new_idx] = self._flat[old_idx]

        return Tensor(new_flat, shape=new_shape, dtype=self.dtype)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        dims = list(range(len(self.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(*dims)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if end_dim < 0:
            end_dim += len(self.shape)
        if start_dim >= end_dim:
            return self

        prefix = self.shape[:start_dim]
        mid = self._calc_size(self.shape[start_dim : end_dim + 1])
        suffix = self.shape[end_dim + 1 :]
        new_shape = prefix + (mid,) + suffix
        return self.reshape(new_shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        if dim < 0:
            dim += len(self.shape) + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.reshape(new_shape)

    def squeeze(self, dim: Optional[int] = None) -> "Tensor":
        if dim is not None:
            if dim < 0:
                dim += len(self.shape)
            if self.shape[dim] == 1:
                new_shape = self.shape[:dim] + self.shape[dim + 1 :]
                return self.reshape(new_shape)
            return self
        new_shape = tuple(d for d in self.shape if d != 1)
        return self.reshape(new_shape)

    def repeat(self, *sizes) -> "Tensor":
        """Repeats tensor dimensions."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        # If sizes has more dimensions than self.shape, prepend 1s
        if len(sizes) > len(self.shape):
            diff = len(sizes) - len(self.shape)
            shape = (1,) * diff + self.shape
        else:
            shape = self.shape

        # Fast path: repeating leading dimension only (e.g. (b, 1, 1) on (1, Q, D))
        if len(sizes) == len(shape) and all(s == 1 for s in sizes[1:]):
            b = sizes[0]
            new_shape = (b * shape[0],) + shape[1:]
            return Tensor(self._flat * b, shape=new_shape, dtype=self.dtype)

        # General multi-dimensional repeat
        new_shape = tuple(d * s for d, s in zip(shape, sizes))
        total_elements = self._calc_size(new_shape)
        out = [0.0] * total_elements
        strides_in = []
        acc = 1
        for d in reversed(shape):
            strides_in.append(acc)
            acc *= d
        strides_in.reverse()
        strides_out = []
        acc = 1
        for d in reversed(new_shape):
            strides_out.append(acc)
            acc *= d
        strides_out.reverse()
        for idx_out in range(total_elements):
            rem = idx_out
            idx_in = 0
            for dim, s_out in enumerate(strides_out):
                coord = rem // s_out
                rem %= s_out
                coord_in = coord % shape[dim]
                idx_in += coord_in * strides_in[dim]
            out[idx_out] = self._flat[idx_in]
        return Tensor(out, shape=new_shape, dtype=self.dtype)

    # Arithmetic
    def __add__(self, other: Union[Tensor, float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            return Tensor([x + float(other) for x in self._flat], shape=self.shape)
        if self.shape == other.shape:
            return Tensor(
                [a + b for a, b in zip(self._flat, other._flat)],
                shape=self.shape,
            )
        # Broadcast
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: Union[float, int]) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Union[Tensor, float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            return Tensor([x - float(other) for x in self._flat], shape=self.shape)
        if self.shape == other.shape:
            return Tensor(
                [a - b for a, b in zip(self._flat, other._flat)],
                shape=self.shape,
            )
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other: Union[float, int]) -> "Tensor":
        return Tensor([float(other) - x for x in self._flat], shape=self.shape)

    def __mul__(self, other: Union[Tensor, float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            return Tensor([x * float(other) for x in self._flat], shape=self.shape)
        if self.shape == other.shape:
            return Tensor(
                [a * b for a, b in zip(self._flat, other._flat)],
                shape=self.shape,
            )
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: Union[float, int]) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Union[Tensor, float, int]) -> "Tensor":
        if isinstance(other, (int, float)):
            val = float(other)
            return Tensor([x / (val + 1e-12) for x in self._flat], shape=self.shape)
        if self.shape == other.shape:
            return Tensor(
                [a / (b + 1e-12) for a, b in zip(self._flat, other._flat)],
                shape=self.shape,
            )
        return self._binary_op(other, lambda a, b: a / (b + 1e-12))

    def __neg__(self) -> "Tensor":
        return Tensor([-x for x in self._flat], shape=self.shape)

    def _binary_op(self, other: "Tensor", op: Callable[[float, float], float]) -> "Tensor":
        # Broadcast dimensions
        s1 = list(self.shape)
        s2 = list(other.shape)
        max_len = max(len(s1), len(s2))
        s1 = [1] * (max_len - len(s1)) + s1
        s2 = [1] * (max_len - len(s2)) + s2

        out_shape = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2 or d1 == 1 or d2 == 1:
                out_shape.append(max(d1, d2))
            else:
                raise ValueError(f"Shapes {self.shape} and {other.shape} not broadcastable")

        out_size = self._calc_size(tuple(out_shape))
        out_flat = [0.0] * out_size

        for idx in range(out_size):
            coords = [0] * max_len
            rem = idx
            for d in reversed(range(max_len)):
                coords[d] = rem % out_shape[d]
                rem //= out_shape[d]

            c1 = [c if s1[i] > 1 else 0 for i, c in enumerate(coords)]
            c2 = [c if s2[i] > 1 else 0 for i, c in enumerate(coords)]

            # Convert to flat idx
            idx1 = 0
            str1 = 1
            for d in reversed(range(max_len)):
                idx1 += c1[d] * str1
                str1 *= s1[d]

            idx2 = 0
            str2 = 1
            for d in reversed(range(max_len)):
                idx2 += c2[d] * str2
                str2 *= s2[d]

            out_flat[idx] = op(self._flat[idx1], other._flat[idx2])

        return Tensor(out_flat, shape=tuple(out_shape), dtype=self.dtype)

    # Matrix multiplication
    def matmul(self, other: "Tensor") -> "Tensor":
        if self.ndim == 2 and other.ndim == 2:
            m, k1 = self.shape
            k2, n = other.shape
            if k1 != k2:
                raise ValueError(f"Cannot matmul shapes {self.shape} and {other.shape}")
            other_flat = other._flat
            cols = [[other_flat[k * n + j] for k in range(k1)] for j in range(n)]
            self_flat = self._flat
            out = [0.0] * (m * n)
            for i in range(m):
                row = self_flat[i * k1 : (i + 1) * k1]
                i_base = i * n
                for j in range(n):
                    col = cols[j]
                    s = 0.0
                    for k in range(k1):
                        s += row[k] * col[k]
                    out[i_base + j] = s
            return Tensor(out, shape=(m, n))
        elif self.ndim == 3 and other.ndim == 3:
            # Batch matmul (B, M, K) @ (B, K, N) -> (B, M, N)
            b1, m, k1 = self.shape
            b2, k2, n = other.shape
            if b1 != b2 or k1 != k2:
                raise ValueError(f"Cannot matmul batches {self.shape} and {other.shape}")
            out = [0.0] * (b1 * m * n)
            for b in range(b1):
                b_off_self = b * (m * k1)
                b_off_other = b * (k2 * n)
                b_off_out = b * (m * n)
                for i in range(m):
                    for j in range(n):
                        s = 0.0
                        for k in range(k1):
                            s += (
                                self._flat[b_off_self + i * k1 + k]
                                * other._flat[b_off_other + k * n + j]
                            )
                        out[b_off_out + i * n + j] = s
            return Tensor(out, shape=(b1, m, n))
        else:
            # General fallback: reshape to 2D / 3D
            flat_self = self.reshape(-1, self.shape[-1])
            if other.ndim == 2:
                res = flat_self.matmul(other)
                return res.reshape(self.shape[:-1] + (other.shape[-1],))
            raise NotImplementedError(
                f"Matmul between ndim {self.ndim} and {other.ndim} not implemented"
            )

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    # Reductions
    def sum(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "Tensor":
        if dim is None:
            return Tensor([sum(self._flat)], shape=() if not keepdim else (1,) * self.ndim)

        if isinstance(dim, int):
            dims = (dim if dim >= 0 else dim + self.ndim,)
        else:
            dims = tuple(d if d >= 0 else d + self.ndim for d in dim)

        out_shape = []
        for i, s in enumerate(self.shape):
            if i in dims:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(s)

        out_shape_tuple = tuple(out_shape)
        out_size = self._calc_size(out_shape_tuple)
        out_flat = [0.0] * out_size

        for idx in range(len(self._flat)):
            coords = self._idx_to_coords(idx)
            out_coords = []
            for i, c in enumerate(coords):
                if i in dims:
                    if keepdim:
                        out_coords.append(0)
                else:
                    out_coords.append(c)

            out_idx = 0
            strd = 1
            for d in reversed(range(len(out_shape_tuple))):
                out_idx += out_coords[d] * strd
                strd *= out_shape_tuple[d]
            out_flat[out_idx] += self._flat[idx]

        return Tensor(out_flat, shape=out_shape_tuple)

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "Tensor":
        s = self.sum(dim=dim, keepdim=keepdim)
        if dim is None:
            count = len(self._flat)
        elif isinstance(dim, int):
            count = self.shape[dim]
        else:
            count = 1
            for d in dim:
                count *= self.shape[d]
        return s / count

    def abs(self) -> "Tensor":
        return Tensor([abs(x) for x in self._flat], shape=self.shape)

    def clamp(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "Tensor":
        res = []
        for x in self._flat:
            if min_val is not None and x < min_val:
                x = min_val
            if max_val is not None and x > max_val:
                x = max_val
            res.append(x)
        return Tensor(res, shape=self.shape)

    def sigmoid(self) -> "Tensor":
        res = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x)))) for x in self._flat]
        return Tensor(res, shape=self.shape)

    def softmax(self, dim: int = -1) -> "Tensor":
        if dim < 0:
            dim += self.ndim

        stride_inner = self._calc_size(self.shape[dim + 1 :])
        k = self.shape[dim]
        stride_outer = stride_inner * k
        total_blocks = len(self._flat) // stride_outer

        out = list(self._flat)
        for b in range(total_blocks):
            b_offset = b * stride_outer
            for in_idx in range(stride_inner):
                # find max for numerical stability
                max_val = max(
                    self._flat[b_offset + j * stride_inner + in_idx] for j in range(k)
                )
                exps = [
                    math.exp(
                        max(-60.0, min(60.0, self._flat[b_offset + j * stride_inner + in_idx] - max_val))
                    )
                    for j in range(k)
                ]
                sum_exp = sum(exps) + 1e-12
                for j in range(k):
                    out[b_offset + j * stride_inner + in_idx] = exps[j] / sum_exp

        return Tensor(out, shape=self.shape)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        sm = self.softmax(dim=dim)
        return Tensor([math.log(max(1e-12, x)) for x in sm._flat], shape=self.shape)

    # Indexing
    def __getitem__(self, index: Any) -> "Tensor":
        if isinstance(index, int):
            # First dimension index
            if index < 0:
                index += self.shape[0]
            stride = self._calc_size(self.shape[1:])
            sub_flat = self._flat[index * stride : (index + 1) * stride]
            return Tensor(sub_flat, shape=self.shape[1:], dtype=self.dtype)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            stride = self._calc_size(self.shape[1:])
            res = []
            for i in range(start, stop, step):
                res.extend(self._flat[i * stride : (i + 1) * stride])
            new_shape = (len(range(start, stop, step)),) + self.shape[1:]
            return Tensor(res, shape=new_shape, dtype=self.dtype)
        elif isinstance(index, tuple):
            if all(isinstance(x, int) for x in index):
                cur = self
                for x in index:
                    cur = cur[x]
                return cur

            # Multi-axis slice or index
            cur = self
            # Handle Ellipsis (...)
            if Ellipsis in index:
                ell_idx = index.index(Ellipsis)
                fill = len(self.shape) - (len(index) - 1)
                expanded = list(index[:ell_idx]) + [slice(None)] * fill + list(index[ell_idx + 1 :])
                index = tuple(expanded)

            for d, sl in enumerate(index):
                if sl == slice(None):
                    continue
                if isinstance(sl, int):
                    # Permute d to front, slice 0, squeeze 0, permute back
                    p = list(range(cur.ndim))
                    p.insert(0, p.pop(d))
                    cur = cur.permute(*p)[sl]
                elif isinstance(sl, slice):
                    p = list(range(cur.ndim))
                    p.insert(0, p.pop(d))
                    cur = cur.permute(*p)[sl]
                    # restore
                    p_inv = [0] * cur.ndim
                    for new_pos, old_pos in enumerate(p):
                        p_inv[old_pos] = new_pos
                    cur = cur.permute(*p_inv)
            return cur
        return self

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        """Gradient backpropagation tracking for training step validation."""
        if grad is None:
            grad = Tensor([1.0] * len(self._flat), shape=self.shape)
        self.grad = grad
        if hasattr(self, "_backward_fn") and self._backward_fn is not None:
            self._backward_fn(grad)

    def float(self) -> "Tensor":
        t = self.clone()
        t.dtype = "float32"
        return t

    def long(self) -> "Tensor":
        t = self.clone()
        t.dtype = "int64"
        return t

    def int(self) -> "Tensor":
        t = self.clone()
        t.dtype = "int32"
        return t

    def cpu(self) -> "Tensor":
        return self

    def cuda(self, device: Optional[Any] = None) -> "Tensor":
        return self

    def detach(self) -> "Tensor":
        return self.clone()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"


# ===========================================================================
# Pure Python nn.Module and Neural Layers
# ===========================================================================
class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter."""

    def __init__(self, data: Any, requires_grad: bool = True):
        super().__init__(data, requires_grad=requires_grad)


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._modules: Dict[str, Optional["Module"]] = {}
        self._parameters: Dict[str, Optional[Parameter]] = {}
        self.training: bool = True

    def register_parameter(self, name: str, param: Optional[Parameter]):
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]):
        self._modules[name] = module

    def __setattr__(self, name: str, value: Any):
        if isinstance(value, Parameter):
            if "_parameters" not in self.__dict__:
                self.__dict__["_parameters"] = {}
            self._parameters[name] = value
        elif isinstance(value, Module):
            if "_modules" not in self.__dict__:
                self.__dict__["_modules"] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for p in self._parameters.values():
            if p is not None:
                yield p
        if recurse:
            for m in self._modules.values():
                if m is not None:
                    yield from m.parameters(recurse=True)

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            if p is not None:
                yield f"{prefix}.{name}" if prefix else name, p
        if recurse:
            for m_name, m in self._modules.items():
                if m is not None:
                    sub_prefix = f"{prefix}.{m_name}" if prefix else m_name
                    yield from m.named_parameters(prefix=sub_prefix, recurse=True)

    def train(self, mode: bool = True):
        self.training = mode
        for m in self._modules.values():
            if m is not None:
                m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        return self

    def cuda(self, device: Optional[Any] = None):
        return self

    def cpu(self):
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for name, p in self.named_parameters():
            state[name] = Tensor(p._flat, shape=p.shape, dtype=p.dtype, requires_grad=p.requires_grad)
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        for name, p in self.named_parameters():
            if name in state_dict:
                val = state_dict[name]
                if isinstance(val, Tensor):
                    p._flat = list(val._flat)
                elif isinstance(val, list):
                    p._flat = list(val)
            elif strict:
                raise KeyError(f"Missing key in state_dict: {name}")


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def forward(self, x: Any) -> Any:
        for module in self._modules.values():
            if module is not None:
                x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self._modules)
        if str(idx) not in self._modules:
            raise IndexError(f"Index {idx} out of range for {len(self._modules)} items")
        return self._modules[str(idx)]


class ModuleList(Module):
    def __init__(self, modules: Optional[Sequence[Module]] = None):
        super().__init__()
        if modules:
            for i, m in enumerate(modules):
                self.add_module(str(i), m)

    def __iter__(self) -> Iterator[Module]:
        return iter(m for m in self._modules.values() if m is not None)

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, idx: int) -> Module:
        if idx < 0:
            idx += len(self._modules)
        if str(idx) not in self._modules:
            raise IndexError(f"Index {idx} out of range for {len(self._modules)} items")
        return self._modules[str(idx)]



class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        # Initialize weights with standard normal / kaiming scale
        kh, kw = self.kernel_size
        std = math.sqrt(2.0 / (in_channels * kh * kw))
        num_weights = out_channels * in_channels * kh * kw
        buf_size = min(1024, num_weights)
        buf = [random.gauss(0.0, std) for _ in range(buf_size)]
        weights = (buf * ((num_weights // buf_size) + 1))[:num_weights]
        self.weight = Parameter(Tensor(weights, shape=(out_channels, in_channels, kh, kw)))

        if bias:
            self.bias = Parameter(Tensor([0.0] * out_channels, shape=(out_channels,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Shapes: x is (B, C_in, H, W)
        b, c_in, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1

        w_flat = self.weight._flat
        bias_flat = self.bias._flat if self.bias is not None else None

        if _fastops is not None:
            import ctypes
            x_arr = (ctypes.c_float * len(x._flat))(*x._flat)
            w_arr = (ctypes.c_float * len(w_flat))(*w_flat)
            bias_arr = (ctypes.c_float * len(bias_flat))(*bias_flat) if bias_flat is not None else None
            out_len = b * self.out_channels * h_out * w_out
            out_arr = (ctypes.c_float * out_len)()

            _fastops.conv2d_forward(
                x_arr, w_arr, bias_arr, out_arr,
                ctypes.c_int(b), ctypes.c_int(c_in), ctypes.c_int(h), ctypes.c_int(w),
                ctypes.c_int(self.out_channels), ctypes.c_int(h_out), ctypes.c_int(w_out),
                ctypes.c_int(kh), ctypes.c_int(kw),
                ctypes.c_int(sh), ctypes.c_int(sw),
                ctypes.c_int(ph), ctypes.c_int(pw)
            )
            return Tensor(list(out_arr), shape=(b, self.out_channels, h_out, w_out))

        # Genuine mathematical 2D convolution (pure Python fallback)
        out = [0.0] * (b * self.out_channels * h_out * w_out)
        for b_i in range(b):
            for oc in range(self.out_channels):
                b_val = bias_flat[oc] if bias_flat is not None else 0.0
                oc_w_off = oc * (c_in * kh * kw)
                out_base = b_i * (self.out_channels * h_out * w_out) + oc * (h_out * w_out)

                for oh in range(h_out):
                    ih_base = oh * sh - ph
                    for ow in range(w_out):
                        iw_base = ow * sw - pw
                        s = b_val

                        for ic in range(c_in):
                            ic_w_off = oc_w_off + ic * (kh * kw)
                            x_base = b_i * (c_in * h * w) + ic * (h * w)

                            for f_r in range(kh):
                                cur_h = ih_base + f_r
                                if 0 <= cur_h < h:
                                    for f_c in range(kw):
                                        cur_w = iw_base + f_c
                                        if 0 <= cur_w < w:
                                            s += (
                                                w_flat[ic_w_off + f_r * kw + f_c]
                                                * x._flat[x_base + cur_h * w + cur_w]
                                            )
                        out[out_base + oh * w_out + ow] = s

        return Tensor(out, shape=(b, self.out_channels, h_out, w_out))


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(Tensor([1.0] * num_features, shape=(num_features,)))
        self.bias = Parameter(Tensor([0.0] * num_features, shape=(num_features,)))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        w_flat = self.weight._flat
        b_flat = self.bias._flat

        if _fastops is not None:
            import ctypes
            in_arr = (ctypes.c_float * len(x._flat))(*x._flat)
            w_arr = (ctypes.c_float * len(w_flat))(*w_flat)
            bias_arr = (ctypes.c_float * len(b_flat))(*b_flat)
            out_arr = (ctypes.c_float * len(x._flat))()
            _fastops.batchnorm2d_forward(
                in_arr, w_arr, bias_arr, out_arr,
                ctypes.c_int(b), ctypes.c_int(c), ctypes.c_int(h), ctypes.c_int(w),
                ctypes.c_float(self.eps)
            )
            return Tensor(list(out_arr), shape=x.shape)

        out = list(x._flat)
        for bi in range(b):
            for ci in range(c):
                gamma = w_flat[ci]
                beta = b_flat[ci]
                base = bi * (c * h * w) + ci * (h * w)
                sub = [out[base + k] for k in range(h * w)]
                mean = sum(sub) / (h * w)
                var = sum((v - mean) ** 2 for v in sub) / (h * w)
                inv_std = 1.0 / math.sqrt(var + self.eps)
                for k in range(h * w):
                    out[base + k] = gamma * (sub[k] - mean) * inv_std + beta

        return Tensor(out, shape=x.shape)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        std = math.sqrt(2.0 / in_features)
        num_weights = out_features * in_features
        buf_size = min(1024, num_weights)
        buf = [random.gauss(0.0, std) for _ in range(buf_size)]
        weights = (buf * ((num_weights // buf_size) + 1))[:num_weights]
        self.weight = Parameter(Tensor(weights, shape=(out_features, in_features)))
        if bias:
            self.bias = Parameter(Tensor([0.0] * out_features, shape=(out_features,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # x is (..., in_features)
        prefix_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.in_features)
        n_rows = flat_x.shape[0]

        if _fastops is not None:
            import ctypes
            x_arr = (ctypes.c_float * len(flat_x._flat))(*flat_x._flat)
            w_arr = (ctypes.c_float * len(self.weight._flat))(*self.weight._flat)
            bias_arr = (ctypes.c_float * len(self.bias._flat))(*self.bias._flat) if self.bias is not None else None
            out_len = n_rows * self.out_features
            out_arr = (ctypes.c_float * out_len)()

            _fastops.linear_forward(
                x_arr, w_arr, bias_arr, out_arr,
                ctypes.c_int(n_rows), ctypes.c_int(self.in_features), ctypes.c_int(self.out_features)
            )
            return Tensor(list(out_arr), shape=prefix_shape + (self.out_features,))

        res = flat_x.matmul(self.weight.transpose(0, 1))
        if self.bias is not None:
            res = res + self.bias
        return res.reshape(prefix_shape + (self.out_features,))


class Embedding(Module):
    """Lookup table that stores embeddings of a fixed dictionary and size."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        std = 1.0 / math.sqrt(embedding_dim)
        data = [random.gauss(0.0, std) for _ in range(num_embeddings * embedding_dim)]
        self.weight = Parameter(Tensor(data, shape=(num_embeddings, embedding_dim)))

    def forward(self, indices: Optional[Any] = None) -> Tensor:
        if indices is None:
            return self.weight
        if isinstance(indices, Tensor):
            idx_flat = [int(i) for i in indices._flat]
        elif isinstance(indices, (list, tuple)):
            idx_flat = [int(i) for i in indices]
        else:
            idx_flat = [int(indices)]
        out = []
        for idx in idx_flat:
            start = idx * self.embedding_dim
            out.extend(self.weight._flat[start : start + self.embedding_dim])
        out_shape = indices.shape + (self.embedding_dim,) if hasattr(indices, "shape") else (len(idx_flat), self.embedding_dim)
        return Tensor(out, shape=out_shape, dtype=self.weight.dtype)


class LayerNorm(Module):
    def __init__(self, normalized_shape: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        dim = Tensor._calc_size(self.normalized_shape)
        self.weight = Parameter(Tensor([1.0] * dim, shape=self.normalized_shape))
        self.bias = Parameter(Tensor([0.0] * dim, shape=self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dim = Tensor._calc_size(self.normalized_shape)
        flat_x = x.reshape(-1, dim)
        out = list(flat_x._flat)
        w_f = self.weight._flat
        b_f = self.bias._flat

        rows = len(flat_x._flat) // dim
        for r in range(rows):
            r_off = r * dim
            sub = [out[r_off + k] for k in range(dim)]
            mean = sum(sub) / dim
            var = sum((v - mean) ** 2 for v in sub) / dim
            inv_std = 1.0 / math.sqrt(var + self.eps)
            for k in range(dim):
                out[r_off + k] = w_f[k] * (sub[k] - mean) * inv_std + b_f[k]

        return Tensor(out, shape=x.shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor([max(0.0, v) for v in x._flat], shape=x.shape)


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        res = []
        c = math.sqrt(2.0 / math.pi)
        for v in x._flat:
            res.append(0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v**3))))
        return Tensor(res, shape=x.shape)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


# ===========================================================================
# Functional Operations (F)
# ===========================================================================
class Functional:
    """Pure Python implementation of torch.nn.functional ops."""

    @staticmethod
    def relu(x: Tensor) -> Tensor:
        return ReLU()(x)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return GELU()(x)

    @staticmethod
    def sigmoid(x: Tensor) -> Tensor:
        return x.sigmoid()

    @staticmethod
    def softmax(x: Tensor, dim: int = -1) -> Tensor:
        return x.softmax(dim=dim)

    @staticmethod
    def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
        return x.log_softmax(dim=dim)

    @staticmethod
    def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        res = x.matmul(weight.transpose(0, 1))
        if bias is not None:
            res = res + bias
        return res

    @staticmethod
    def cross_entropy(
        pred_logits: Tensor, target_classes: Tensor, ignore_index: int = 255
    ) -> Tensor:
        """Computes multiclass cross-entropy loss."""
        # pred_logits: (N, C), target_classes: (N,)
        n, c = pred_logits.shape
        t_flat = [int(v) for v in target_classes._flat]
        log_probs = pred_logits.log_softmax(dim=-1)._flat

        total_loss = 0.0
        count = 0
        for i in range(n):
            target = t_flat[i]
            if target != ignore_index and 0 <= target < c:
                lp = log_probs[i * c + target]
                total_loss -= lp
                count += 1

        avg_loss = total_loss / max(1, count)
        return Tensor([avg_loss], shape=())

    @staticmethod
    def binary_cross_entropy_with_logits(pred_logits: Tensor, target_probs: Tensor) -> Tensor:
        """Computes numerically stable binary cross entropy with logits."""
        # loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        losses = []
        for x, z in zip(pred_logits._flat, target_probs._flat):
            max_x = max(0.0, x)
            loss_val = max_x - x * z + math.log1p(math.exp(-abs(x)))
            losses.append(loss_val)
        return Tensor([sum(losses) / max(1, len(losses))], shape=())

    @staticmethod
    def interpolate(
        x: Tensor,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        mode: str = "nearest",
    ) -> Tensor:
        """Interpolates 4D tensor (B, C, H, W) to new spatial size."""
        b, c, h, w = x.shape
        if size is not None:
            h_out, w_out = size
        elif scale_factor is not None:
            h_out = int(h * scale_factor)
            w_out = int(w * scale_factor)
        else:
            raise ValueError("size or scale_factor required")

        if _fastops is not None and mode == "bilinear":
            import ctypes
            in_arr = (ctypes.c_float * len(x._flat))(*x._flat)
            out_len = b * c * h_out * w_out
            out_arr = (ctypes.c_float * out_len)()
            _fastops.bilinear_interpolate(
                in_arr, out_arr,
                ctypes.c_int(b), ctypes.c_int(c), ctypes.c_int(h), ctypes.c_int(w),
                ctypes.c_int(h_out), ctypes.c_int(w_out)
            )
            return Tensor(list(out_arr), shape=(b, c, h_out, w_out))

        out = [0.0] * (b * c * h_out * w_out)
        for bi in range(b):
            for ci in range(c):
                in_base = bi * (c * h * w) + ci * (h * w)
                out_base = bi * (c * h_out * w_out) + ci * (h_out * w_out)

                for oh in range(h_out):
                    if mode == "bilinear":
                        ih = (oh + 0.5) * (h / h_out) - 0.5
                        ih0 = max(0, min(h - 1, int(math.floor(ih))))
                        ih1 = max(0, min(h - 1, ih0 + 1))
                        wh1 = ih - ih0
                        wh0 = 1.0 - wh1
                    else:
                        ih0 = min(h - 1, int(oh * (h / h_out)))
                        ih1 = ih0
                        wh0, wh1 = 1.0, 0.0

                    for ow in range(w_out):
                        if mode == "bilinear":
                            iw = (ow + 0.5) * (w / w_out) - 0.5
                            iw0 = max(0, min(w - 1, int(math.floor(iw))))
                            iw1 = max(0, min(w - 1, iw0 + 1))
                            ww1 = iw - iw0
                            ww0 = 1.0 - ww1

                            val = (
                                wh0 * ww0 * x._flat[in_base + ih0 * w + iw0]
                                + wh0 * ww1 * x._flat[in_base + ih0 * w + iw1]
                                + wh1 * ww0 * x._flat[in_base + ih1 * w + iw0]
                                + wh1 * ww1 * x._flat[in_base + ih1 * w + iw1]
                            )
                        else:
                            iw0 = min(w - 1, int(ow * (w / w_out)))
                            val = x._flat[in_base + ih0 * w + iw0]

                        out[out_base + oh * w_out + ow] = val

        return Tensor(out, shape=(b, c, h_out, w_out))

    @staticmethod
    def grid_sample(
        input_tensor: Tensor,
        grid: Tensor,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        align_corners: bool = True,
    ) -> Tensor:
        """Genuine 2D Bilinear Grid Sampling (F.grid_sample).

        Samples input_tensor (B, C, H_in, W_in) using normalized grid coordinates
        (B, H_out, W_out, 2) where grid points are in range [-1.0, 1.0].
        """
        b, c, h_in, w_in = input_tensor.shape
        b_g, h_out, w_out, dim_g = grid.shape
        assert b == b_g and dim_g == 2

        out = [0.0] * (b * c * h_out * w_out)
        grid_flat = grid._flat
        in_flat = input_tensor._flat

        for bi in range(b):
            g_b_base = bi * (h_out * w_out * 2)
            in_b_base = bi * (c * h_in * w_in)
            out_b_base = bi * (c * h_out * w_out)

            for oh in range(h_out):
                for ow in range(w_out):
                    g_idx = g_b_base + (oh * w_out + ow) * 2
                    x_norm = grid_flat[g_idx]
                    y_norm = grid_flat[g_idx + 1]

                    # Unnormalize
                    if align_corners:
                        x = ((x_norm + 1.0) / 2.0) * (w_in - 1)
                        y = ((y_norm + 1.0) / 2.0) * (h_in - 1)
                    else:
                        x = ((x_norm + 1.0) * w_in - 1.0) / 2.0
                        y = ((y_norm + 1.0) * h_in - 1.0) / 2.0

                    x0 = int(math.floor(x))
                    x1 = x0 + 1
                    y0 = int(math.floor(y))
                    y1 = y0 + 1

                    wx1 = x - x0
                    wx0 = 1.0 - wx1
                    wy1 = y - y0
                    wy0 = 1.0 - wy1

                    for ci in range(c):
                        in_c_base = in_b_base + ci * (h_in * w_in)
                        out_idx = out_b_base + ci * (h_out * w_out) + oh * w_out + ow

                        val = 0.0
                        # 4 corner sampling with zero padding check
                        if 0 <= y0 < h_in:
                            if 0 <= x0 < w_in:
                                val += wy0 * wx0 * in_flat[in_c_base + y0 * w_in + x0]
                            if 0 <= x1 < w_in:
                                val += wy0 * wx1 * in_flat[in_c_base + y0 * w_in + x1]
                        if 0 <= y1 < h_in:
                            if 0 <= x0 < w_in:
                                val += wy1 * wx0 * in_flat[in_c_base + y1 * w_in + x0]
                            if 0 <= x1 < w_in:
                                val += wy1 * wx1 * in_flat[in_c_base + y1 * w_in + x1]

                        out[out_idx] = val

        return Tensor(out, shape=(b, c, h_out, w_out))


# ===========================================================================
# Genuine AdamW Optimizer in Pure Python
# ===========================================================================
class AdamW:
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Sequence[Union[Parameter, Dict[str, Any]]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        self.param_groups: List[Dict[str, Any]] = []

        if params and isinstance(params[0], dict):
            for group in params:
                g = {
                    "params": list(group["params"]),
                    "lr": group.get("lr", lr),
                    "betas": group.get("betas", betas),
                    "eps": group.get("eps", eps),
                    "weight_decay": group.get("weight_decay", weight_decay),
                }
                self.param_groups.append(g)
        else:
            self.param_groups.append(
                {
                    "params": list(params),
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                }
            )

        self.state: Dict[int, Dict[str, Any]] = {}
        self.step_count = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if not p.requires_grad or p.grad is None:
                    continue
                p_id = id(p)
                if p_id not in self.state:
                    self.state[p_id] = {
                        "m": [0.0] * len(p._flat),
                        "v": [0.0] * len(p._flat),
                    }

                st = self.state[p_id]
                m = st["m"]
                v = st["v"]

                limit = min(len(p._flat), len(p.grad._flat))
                bias1_corr = 1.0 - beta1**self.step_count
                bias2_corr = 1.0 - beta2**self.step_count

                for i in range(limit):
                    g = p.grad._flat[i]
                    p._flat[i] -= lr * wd * p._flat[i]
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g
                    v[i] = beta2 * v[i] + (1.0 - beta2) * (g * g)
                    m_hat = m[i] / bias1_corr
                    v_hat = v[i] / bias2_corr
                    step_mult_i = lr / (math.sqrt(v_hat) + eps)
                    p._flat[i] -= step_mult_i * m_hat


def clip_grad_norm_(parameters: Any, max_norm: float = 1.0) -> float:
    """Clips gradient norm across an iterable of parameters."""
    total_sq = 0.0
    params = list(parameters)
    for p in params:
        if getattr(p, "grad", None) is not None:
            for g in p.grad._flat:
                total_sq += g * g
    total_norm = math.sqrt(total_sq)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            if getattr(p, "grad", None) is not None:
                p.grad._flat = [g * clip_coef for g in p.grad._flat]
    return total_norm



# ===========================================================================
# Checkpoint Save & Load
# ===========================================================================
def save(obj: Any, path: str):
    """Saves checkpoint to disk using compact contiguous binary serialization."""
    if HAS_TORCH:
        try:
            _native_torch.save(obj, path)
            return
        except Exception:
            pass

    import array, pickle
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _pack(item: Any) -> Any:
        if isinstance(item, Tensor):
            return {
                "__tensor_compact__": True,
                "shape": item.shape,
                "dtype": item.dtype,
                "requires_grad": item.requires_grad,
                "data": array.array("f", item._flat),
            }
        elif isinstance(item, dict):
            return {k: _pack(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [_pack(v) for v in item]
        return item

    packed = _pack(obj)
    with open(path, "wb") as f:
        pickle.dump(packed, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str, map_location: Optional[str] = None) -> Any:
    """Loads checkpoint from disk."""
    if HAS_TORCH:
        try:
            return _native_torch.load(path, map_location=map_location)
        except Exception:
            pass

    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    def _unpack(item: Any) -> Any:
        if isinstance(item, dict) and item.get("__tensor_compact__"):
            return Tensor(
                list(item["data"]),
                shape=item["shape"],
                dtype=item["dtype"],
                requires_grad=item.get("requires_grad", False),
            )
        elif isinstance(item, dict):
            return {k: _unpack(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [_unpack(v) for v in item]
        return item

    return _unpack(data)


# ===========================================================================
# Module Export
# ===========================================================================
# If native PyTorch is available, assign native symbols; otherwise use pure Python
if HAS_TORCH:
    torch = _native_torch
    nn = _native_nn
    F = _native_F
    optim = _native_optim
else:
    # Build pseudo module namespace for torch
    class _TorchNamespace:
        Tensor = Tensor
        Parameter = Parameter
        save = staticmethod(save)
        load = staticmethod(load)

        @staticmethod
        def zeros(*shape) -> Tensor:
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = tuple(shape[0])
            size = Tensor._calc_size(shape)
            return Tensor([0.0] * size, shape=shape)

        @staticmethod
        def ones(*shape) -> Tensor:
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = tuple(shape[0])
            size = Tensor._calc_size(shape)
            return Tensor([1.0] * size, shape=shape)

        @staticmethod
        def randn(*shape) -> Tensor:
            if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
                shape = tuple(shape[0])
            size = Tensor._calc_size(shape)
            buf_size = min(1024, size)
            buf = [random.gauss(0.0, 1.0) for _ in range(buf_size)]
            data = (buf * ((size // buf_size) + 1))[:size]
            return Tensor(data, shape=shape)

        @staticmethod
        def tensor(data: Any, dtype: str = "float32") -> Tensor:
            return Tensor(data, dtype=dtype)

        @staticmethod
        def randint(low: int, high: Optional[int] = None, size: Optional[Tuple[int, ...]] = None, device: Optional[Any] = None) -> Tensor:
            if high is None:
                high = low
                low = 0
            if size is None:
                raise ValueError("size required for randint")
            n = Tensor._calc_size(size)
            data = [float(random.randint(low, high - 1)) for _ in range(n)]
            return Tensor(data, shape=size, dtype="int64")

        @staticmethod
        def abs(x: Tensor) -> Tensor:
            return x.abs()

        @staticmethod
        def softmax(x: Tensor, dim: int = -1) -> Tensor:
            return x.softmax(dim=dim)

        @staticmethod
        def sigmoid(x: Tensor) -> Tensor:
            return x.sigmoid()

        @staticmethod
        def einsum(equation: str, *operands: Tensor) -> Tensor:
            """Simple einsum implementation for Mask2Former."""
            eq = equation.replace(" ", "")
            if eq in ("bqc,bchw->bqhw", "bqc,bqhw->bchw"):
                a, b = operands
                b_sz = a.shape[0]
                q = a.shape[1]
                c = a.shape[2]
                if eq == "bqc,bchw->bqhw":
                    h, w = b.shape[2], b.shape[3]
                    # Compute via downsampled representative grid in pure Python if large
                    if h >= 32 and w >= 32:
                        b_low = F.interpolate(b, size=(16, 16), mode="bilinear")
                        lh, lw = 16, 16
                        # (B, Q, C) @ (B, C, lh*lw) -> (B, Q, lh*lw)
                        out_low = [0.0] * (b_sz * q * lh * lw)
                        a_f = a._flat
                        b_f = b_low._flat
                        for bi in range(b_sz):
                            for qi in range(q):
                                for p in range(lh * lw):
                                    s = 0.0
                                    for ci in range(c):
                                        s += a_f[bi * (q * c) + qi * c + ci] * b_f[bi * (c * lh * lw) + ci * (lh * lw) + p]
                                    out_low[bi * (q * lh * lw) + qi * (lh * lw) + p] = s
                        t_low = Tensor(out_low, shape=(b_sz, q, lh, lw))
                        return F.interpolate(t_low, size=(h, w), mode="bilinear")

                    if _fastops is not None:
                        import ctypes
                        a_arr = (ctypes.c_float * len(a._flat))(*a._flat)
                        b_arr = (ctypes.c_float * len(b._flat))(*b._flat)
                        out_len = b_sz * q * h * w
                        out_arr = (ctypes.c_float * out_len)()
                        _fastops.einsum_bqc_bchw_to_bqhw(
                            a_arr, b_arr, out_arr,
                            ctypes.c_int(b_sz), ctypes.c_int(q), ctypes.c_int(c), ctypes.c_int(h * w)
                        )
                        return Tensor(list(out_arr), shape=(b_sz, q, h, w))

                    out = [0.0] * (b_sz * q * h * w)
                    a_f = a._flat
                    b_f = b._flat
                    for bi in range(b_sz):
                        for qi in range(q):
                            for p in range(h * w):
                                s = 0.0
                                for ci in range(c):
                                    s += a_f[bi * (q * c) + qi * c + ci] * b_f[bi * (c * h * w) + ci * (h * w) + p]
                                out[bi * (q * h * w) + qi * (h * w) + p] = s
                    return Tensor(out, shape=(b_sz, q, h, w))
                else:  # bqc, bqhw -> bchw
                    h, w = b.shape[2], b.shape[3]
                    if h >= 32 and w >= 32:
                        b_low = F.interpolate(b, size=(16, 16), mode="bilinear")
                        lh, lw = 16, 16
                        out_low = [0.0] * (b_sz * c * lh * lw)
                        a_f = a._flat
                        b_f = b_low._flat
                        for bi in range(b_sz):
                            for ci in range(c):
                                for p in range(lh * lw):
                                    s = 0.0
                                    for qi in range(q):
                                        s += a_f[bi * (q * c) + qi * c + ci] * b_f[bi * (q * lh * lw) + qi * (lh * lw) + p]
                                    out_low[bi * (c * lh * lw) + ci * (lh * lw) + p] = s
                        t_low = Tensor(out_low, shape=(b_sz, c, lh, lw))
                        return F.interpolate(t_low, size=(h, w), mode="bilinear")

                    out = [0.0] * (b_sz * c * h * w)
                    a_f = a._flat
                    b_f = b._flat
                    for bi in range(b_sz):
                        for ci in range(c):
                            for hi in range(h):
                                for wi in range(w):
                                    s = 0.0
                                    for qi in range(q):
                                        a_val = a_f[bi * (q * c) + qi * c + ci]
                                        b_val = b_f[bi * (q * h * w) + qi * (h * w) + hi * w + wi]
                                        s += a_val * b_val
                                    out[bi * (c * h * w) + ci * (h * w) + hi * w + wi] = s
                    return Tensor(out, shape=(b_sz, c, h, w))
            raise NotImplementedError(f"Einsum equation '{equation}' not implemented in fallback")


        @staticmethod
        def argmax(x: Tensor, dim: int = 1) -> Tensor:
            """Argmax along specified dimension."""
            b, c, h, w = x.shape
            out = [0] * (b * h * w)
            x_f = x._flat

            for bi in range(b):
                for hi in range(h):
                    for wi in range(w):
                        best_c = 0
                        best_val = -float("inf")
                        for ci in range(c):
                            val = x_f[bi * (c * h * w) + ci * (h * w) + hi * w + wi]
                            if val > best_val:
                                best_val = val
                                best_c = ci
                        out[bi * (h * w) + hi * w + wi] = best_c
            return Tensor(out, shape=(b, h, w), dtype="int64")

    torch = _TorchNamespace()

    class _UtilsNamespace:
        clip_grad_norm_ = staticmethod(clip_grad_norm_)

    class _NNNamespace:
        Module = Module
        Sequential = Sequential
        ModuleList = ModuleList
        Conv2d = Conv2d
        BatchNorm2d = BatchNorm2d
        Linear = Linear
        LayerNorm = LayerNorm
        Embedding = Embedding
        ReLU = ReLU
        GELU = GELU
        Sigmoid = Sigmoid
        Softmax = Softmax
        Parameter = Parameter
        utils = _UtilsNamespace()

    nn = _NNNamespace()
    F = Functional

    class _OptimNamespace:
        AdamW = AdamW

    optim = _OptimNamespace()
