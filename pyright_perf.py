from __future__ import annotations

import math
from collections import abc
from functools import partial
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union, cast

import numpy as np
import tensorflow as tf
from typing_extensions import TypeGuard

T1 = TypeVar("T1")
T2 = TypeVar("T2")
ContainerGeneric = Union[Mapping[str, "ContainerGeneric[T1]"], Sequence["ContainerGeneric[T1]"], T1]
ContainerArrays = ContainerGeneric[Union[np.ndarray, np.number]]
ContainerTensors = ContainerGeneric[Union[tf.Tensor, tf.SparseTensor]]
ContainerAnyTensors = TypeVar("ContainerAnyTensors", ContainerTensors, ContainerArrays)
ArrayLike = Union[np.ndarray, np.number, tf.Tensor, tf.SparseTensor, float]

ComparisonResultT = Tuple[bool, Optional[List["int | str"]]]
DEFAULT_ABSOLUTE_TOLERANCE = 1e-4


def container_fmap(f: Callable[[T1], T2], elements: ContainerGeneric[T1]) -> ContainerGeneric[T2]:
    if isinstance(elements, abc.Mapping):
        return {key: container_fmap(f, val) for key, val in elements.items()}
    if isinstance(elements, tuple):
        return tuple(map(partial(container_fmap, f), elements))
    if isinstance(elements, abc.Sequence) and not isinstance(elements, (str, bytes)):
        return [container_fmap(f, elem) for elem in elements]
    return f(elements)


def _is_integer_dtype(dtype: np.dtype | tf.dtypes.DType) -> bool:
    if isinstance(dtype, np.dtype):
        return np.issubdtype(dtype, np.integer)
    else:
        return dtype.is_integer


def tensor_equality(
    tensor1: ArrayLike,
    tensor2: ArrayLike,
    atol: float = DEFAULT_ABSOLUTE_TOLERANCE,
    test_env: Optional[tf.test.TestCase] = None,
) -> bool:
    """
    Checks if two tf tensors/numpy array are equal. For integral tensors checks for exact
    equality. For float tensors checks for approximate equality. It expects
    both tensors to have the same dtype and same shape.

    For tf tensors assumes they are eager tensors. Graph tensors need to be evaluated in session
    to numpy array. If you pass in tf.test.TestCase it will work with graph tensors that can
    be evaluated by it.

    It expects the type to be same shape with one exception, being that scalers can be compared against tensors.
    """
    if isinstance(tensor1, tf.SparseTensor) or isinstance(tensor2, tf.SparseTensor):
        tensor1 = tf.sparse.to_dense(tensor1)
        tensor2 = tf.sparse.to_dense(tensor2)

    if isinstance(tensor1, float) and isinstance(tensor2, float):
        return math.isclose(tensor1, tensor2, abs_tol=atol)

    if isinstance(tensor1, (int, float)):
        if isinstance(tensor2, tf.Tensor):
            tensor1 = tf.constant(tensor1)
        else:
            tensor1 = np.array(tensor1)

    if isinstance(tensor2, (int, float)):
        if isinstance(tensor1, tf.Tensor):
            tensor2 = tf.constant(tensor2)
        else:
            tensor2 = np.array(tensor2)

    assert not isinstance(tensor1, (int, float))
    assert not isinstance(tensor2, (int, float))

    if type(tensor1) != type(tensor2):
        return False

    if tensor1.dtype != tensor2.dtype:
        return False

    if tensor1.shape != tensor2.shape:
        return False

    if test_env:
        assert isinstance(tensor1, tf.Tensor)
        array1 = test_env.evaluate(tensor1)
        array2 = test_env.evaluate(tensor2)
    else:
        if isinstance(tensor1, tf.Tensor) and isinstance(tensor2, tf.Tensor):
            array1 = tensor1.numpy()
            array2 = tensor2.numpy()
        else:
            assert isinstance(tensor1, (np.ndarray, np.number))
            assert isinstance(tensor2, (np.ndarray, np.number))
            array1 = tensor1
            array2 = tensor2

    if _is_integer_dtype(array1.dtype):
        return bool(np.all(array1 == array2))
    else:
        comparisons = np.isclose(array1, array2, atol=atol, equal_nan=True)
        return bool(comparisons.all())


def extract_nested_key(data: ContainerGeneric[T1], key_path: Iterable[int | str]) -> T1:
    result = data
    for key in key_path:
        if isinstance(key, int):
            assert isinstance(result, abc.Sequence)
            result = result[key]
        else:
            assert isinstance(result, abc.Mapping)
            result = result[key]

    return cast(T1, result)


def _tensor_collection_equality(
    tensors1: ContainerArrays,
    tensors2: ContainerArrays,
    atol: float = DEFAULT_ABSOLUTE_TOLERANCE,
    debug_path: Optional[List[str | int]] = None,
) -> ComparisonResultT:
    if debug_path is None:
        debug_path = []

    if not (isinstance(tensors1, type(tensors2)) or isinstance(tensors2, type(tensors1))):
        return (False, debug_path)

    # The ors are here for the type checker. While previous line guarantees that the types are the same
    # the type checker is unable to use that.
    if not isinstance(tensors1, (np.ndarray, dict, list, tuple, np.number)) or not isinstance(
        tensors2, (np.ndarray, dict, list, tuple, np.number)
    ):
        raise TypeError(f"Unexpected type for tensors1: {type(tensors1)}")

    if isinstance(tensors1, (np.ndarray, np.number)) or isinstance(tensors2, (np.ndarray, np.number)):
        result = tensor_equality(tensors1, tensors2, atol=atol)
        if result:
            return (result, None)
        else:
            return (result, debug_path)

    if len(tensors1) != len(tensors2):
        return (False, debug_path)

    if isinstance(tensors1, dict):
        assert isinstance(tensors2, dict)

        for key, value in tensors1.items():
            if key not in tensors2:
                return (False, debug_path + [key])
            key_result, key_debug_path = _tensor_collection_equality(
                value, tensors2[key], atol=atol, debug_path=debug_path + [key]
            )
            # Short circuit if possible.
            if not key_result:
                return (key_result, key_debug_path)
        return (True, None)

    assert isinstance(tensors2, (list, tuple))

    for key, (tensor1, tensor2) in enumerate(zip(tensors1, tensors2)):
        debug_extension: List[int | str] = [key]
        key_result, key_debug_path = _tensor_collection_equality(
            tensor1, tensor2, atol=atol, debug_path=debug_path + debug_extension
        )
        if not key_result:
            return (key_result, key_debug_path)

    return (True, None)


def evaluate_tensors(tensors: ContainerTensors, test_env: Optional[tf.test.TestCase]) -> ContainerArrays:
    if test_env:
        return test_env.evaluate(tensors)
    else:
        return container_fmap(lambda tensor: tensor.numpy(), tensors)  # type: ignore


def _log_tensor_equality_mismatch(
    arrays1: ContainerArrays, arrays2: ContainerArrays, debug_path: Iterable[str | int]
) -> None:
    debug_path_str = ",".join(map(str, debug_path))

    array1: np.ndarray | np.number
    array2: np.ndarray | np.number
    if debug_path_str != "":
        print(f"Tensor Collection mismatch occurred at {debug_path_str}")
        array1 = extract_nested_key(arrays1, debug_path)  # type: ignore
        array2 = extract_nested_key(arrays2, debug_path)  # type: ignore
    else:
        assert isinstance(arrays1, (np.ndarray, np.number))
        assert isinstance(arrays2, (np.ndarray, np.number))
        array1 = arrays1
        array2 = arrays2

    print(f"Mismatched tensors")
    print(array1)
    print(array2)

    shape_match = array1.shape == array2.shape
    dtype_match = array1.dtype == array2.dtype

    if not shape_match:
        print(f"Mismatch shapes")
        print(f"Shape 1: {array1.shape} Shape 2: {array2.shape}")

    if not dtype_match:
        print(f"Mismatch dtypes")
        print(f"Dtype 1: {array1.dtype} Dtype 2: {array2.dtype}")

    if shape_match and dtype_match:
        diff = np.absolute(array1 - array2) # type: ignore
        print(f"Maximum absolute difference: ", diff.max())
        print(f"Index Maximum Difference: ", np.unravel_index(diff.argmax(), diff.shape))


def check_container_tensors(tensors: ContainerTensors | ContainerArrays) -> TypeGuard[ContainerTensors]:
    if isinstance(tensors, (abc.Mapping, abc.Sequence)):
        if len(tensors) == 0:
            return True
        if isinstance(tensors, abc.Mapping):
            return check_container_tensors(next(iter(tensors.values())))
        else:
            return check_container_tensors(next(iter(tensors)))

    return isinstance(tensors, (tf.Tensor, tf.SparseTensor))


def tensor_collection_equality(
    tensors1: ContainerAnyTensors,
    tensors2: ContainerAnyTensors,
    atol: float = DEFAULT_ABSOLUTE_TOLERANCE,
    test_env: Optional[tf.test.TestCase] = None,
    log_debug_path: bool = True,
) -> bool:
    """
    Compares two collections of tensors. Tensors can either be numpy arrays or tensorflow tensors. The collection
    should all be of the same type and should not mix numpy/tensorflow. Mixing the two in one collection will error out.
    Both collections should also be consistent in either being numpy arrays or tensorflow tensors. An assertion error will
    happen if the two collections are inconsistent type wise. If the tensors are graph tensors then test_env is required.

    By default if this fails it will print out information about the mismatch. log_debug_path can be disabled
    if you not want any error messages and just need the return value.

    Dense and sparse tensors are both suported. Sparse tensors will all be converted to dense tensors. This does not
    check whether the two collections are consistently sparse or dense.
    """
    assert check_container_tensors(tensors1) == check_container_tensors(tensors2)

    if check_container_tensors(tensors1):
        arrays1 = evaluate_tensors(tensors1, test_env)
    else:
        arrays1 = cast(ContainerArrays, tensors1)

    if check_container_tensors(tensors2):
        arrays2 = evaluate_tensors(tensors2, test_env)
    else:
        arrays2 = cast(ContainerArrays, tensors2)

    result, debug_path = _tensor_collection_equality(arrays1, arrays2, atol=atol)

    if not result and log_debug_path:
        assert debug_path is not None
        _log_tensor_equality_mismatch(arrays1, arrays2, debug_path)

    return result
