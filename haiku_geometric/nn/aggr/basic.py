import jax.numpy as jnp
from typing import Optional, Union
from jraph._src.utils import segment_sum, segment_mean, segment_max, segment_min_or_constant

from haiku_geometric.nn.aggr.base import Aggregation


class SumAggregation(Aggregation):

    def __call__(
            self,
            data: jnp.ndarray,
            segment_ids: jnp.ndarray,
            num_segments: Optional[int] = None,
            indices_are_sorted: bool = False,
            unique_indices: bool = False):
        return segment_sum(
            data,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices)


class MeanAggregation(Aggregation):

    def __call__(
            self,
            data: jnp.ndarray,
            segment_ids: jnp.ndarray,
            num_segments: Optional[int] = None,
            indices_are_sorted: bool = False,
            unique_indices: bool = False):
        return segment_mean(
            data,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices)


class MaxAggregation(Aggregation):

    def __call__(
            self,
            data: jnp.ndarray,
            segment_ids: jnp.ndarray,
            num_segments: Optional[int] = None,
            indices_are_sorted: bool = False,
            unique_indices: bool = False):
        return segment_max(
            data,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices)


class MinAggregation(Aggregation):

    def __call__(
            self,
            data: jnp.ndarray,
            segment_ids: jnp.ndarray,
            num_segments: Optional[int] = None,
            indices_are_sorted: bool = False,
            unique_indices: bool = False):
        return segment_min_or_constant(
            data,
            segment_ids,
            num_segments,
            indices_are_sorted,
            unique_indices)




