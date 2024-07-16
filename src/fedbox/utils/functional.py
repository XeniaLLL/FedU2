from typing import Union, Callable, Sequence, Mapping, cast, overload

import torch
import torch.nn
from torch import Tensor
from torch.nn import Module


@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, tuple[Tensor, Tensor]]: ...
@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, tuple[Tensor, Tensor, Tensor]]: ...
@overload
def model_named_zip(
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    /,
    *models: Union[Module, Mapping[str, Tensor]],
) -> dict[str, tuple[Tensor, ...]]: ...


def model_named_zip(*models: Union[Module, Mapping[str, Tensor]]) -> dict[str, tuple[Tensor, ...]]:
    mappings: list[Mapping[str, Tensor]] = [m.state_dict() if isinstance(m, Module) else m for m in models]
    keys = mappings[0].keys()
    return {name: tuple(m[name] for m in mappings) for name in keys}  # TODO convert to tensor?


@overload
def model_aggregate(
    aggregator: Callable[[Sequence[Tensor]], Tensor],
    models: Sequence[Union[Module, Mapping[str, Tensor]]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    /
) -> dict[str, Tensor]: ...
@overload
def model_aggregate(
    aggregator: Callable[..., Tensor],
    model1: Union[Module, Mapping[str, Tensor]],
    model2: Union[Module, Mapping[str, Tensor]],
    model3: Union[Module, Mapping[str, Tensor]],
    model4: Union[Module, Mapping[str, Tensor]],
    model5: Union[Module, Mapping[str, Tensor]],
    /,
    *models: Module,
) -> dict[str, Tensor]: ...


@torch.no_grad()
def model_aggregate(aggregator: Callable[..., Tensor], *args) -> dict[str, Tensor]:
    """
    The sequence version: ::

        models: list[Module] = ...
        result: Module = ...
        assign[result] = model_aggregation(average, models)

    The unpacked version: ::

        ma: Module = ...
        mb: Module = ...
        result: Module = ...
        assign[result] = model_aggregation(lambda a, b: (a + b) / 2, ma, mb)
    """
    result: dict[str, Tensor] = {}
    if len(args) == 1:
        for name, params in model_named_zip(*args[0]).items():
            result[name] = aggregator(params)
    else:
        assert len(args) >= 2
        for name, params in model_named_zip(*args).items():
            result[name] = aggregator(*params)
    return result


def model_average(
    models: Sequence[Union[Module, Mapping[str, Tensor]]],
    weights: Sequence[float],
    normalize: bool = True,
) -> Mapping[str, Tensor]:
    return model_aggregate(
        lambda params: weighted_average(params, weights, normalize),
        models
    )


def weighted_average(
    tensors: Sequence[Tensor],
    weights: Sequence[float],
    normalize: bool = True
) -> Tensor:
    if normalize:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]
    result = sum(x * w for x, w in zip(tensors, weights))
    return cast(Tensor, result)


def model_assign(dest: Module, src: Union[Module, Mapping[str, Tensor]]):
    if isinstance(src, Module):
        src = src.state_dict()
    dest.load_state_dict(src, strict=False)


class Assign:
    def __setitem__(self, dest: Module, src: Union[Module, Mapping[str, Tensor]]):
        model_assign(dest, src)

    __call__ = __setitem__


assign = Assign()
"""
A helper object to assign model parameters. ::

    assign[dest] = src

is equivalent to ::

    model_assign(dest, src)
"""
