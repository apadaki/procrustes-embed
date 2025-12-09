from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class NuclearNormApproximation:
    """Configuration for the nuclear norm approximation used by competitors."""

    fn: str = "exact"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentParams:
    """
    Tunable parameters shared across competitors.

    Attributes:
        k: Number of pieces/segments the primary algorithm should produce.
        n_samples: Number of samples to generate for the dataset.
        order: Optional ordering for the samples. If omitted, the experiment
            runner fills it with the natural ordering.
        nuclear_norm: Parameters controlling the nuclear norm approximation.
        dp_samples: Optional down-sampling factor for the dynamic program.
    """

    k: int
    n_samples: int
    order: Sequence[int] | None = None
    nuclear_norm: NuclearNormApproximation = field(default_factory=NuclearNormApproximation)
    dp_samples: int | None = None

    def with_order(self, order: Sequence[int]) -> "ExperimentParams":
        """Return a copy with the provided order enforced."""
        return replace(self, order=list(order))


@dataclass
class ExperimentData:
    """Container for generated data and aux metadata."""

    X: np.ndarray
    Y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmOutcome:
    """Structured return for competitor algorithms."""

    cost: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMeasurement:
    """Metrics captured for each competitor run."""

    metrics: dict[str, float]
    details: dict[str, Any]


@dataclass
class ExperimentResult:
    """Full result of running one experiment."""

    params: ExperimentParams
    data: ExperimentData
    competitor_results: dict[str, ExperimentMeasurement]


@dataclass
class CompetitorAlgorithm:
    """Descriptor tying a friendly name to the evaluation callable."""

    name: str
    fn: Callable[[np.ndarray, np.ndarray, ExperimentParams], AlgorithmOutcome]


class ExperimentRunner:
    """Utility to orchestrate dataset generation and competitor execution."""

    _SUPPORTED_METRICS = {"runtime", "cost"}

    def __init__(self, data_generator: Callable[..., ExperimentData]):
        self._data_generator = data_generator

    def run(
        self,
        params: ExperimentParams,
        competitors: Iterable[CompetitorAlgorithm],
        *,
        metrics: Sequence[str] = ("runtime", "cost"),
        data_kwargs: Mapping[str, Any] | None = None,
    ) -> ExperimentResult:
        metric_list = list(metrics)
        invalid = [metric for metric in metric_list if metric not in self._SUPPORTED_METRICS]
        if invalid:
            raise ValueError(f"Unsupported metrics requested: {invalid}")

        data_kwargs = dict(data_kwargs or {})
        data_kwargs.setdefault("n_samples", params.n_samples)

        data = self._data_generator(**data_kwargs)
        if not isinstance(data, ExperimentData):
            raise TypeError("Data generator must return an ExperimentData instance.")

        resolved_params = params
        if params.order is None:
            resolved_params = params.with_order(range(data.X.shape[0]))

        competitor_results: dict[str, ExperimentMeasurement] = {}
        for competitor in competitors:
            start = time.perf_counter()
            outcome = competitor.fn(data.X, data.Y, resolved_params)
            runtime = time.perf_counter() - start

            metric_values: dict[str, float] = {}
            for metric in metric_list:
                if metric == "runtime":
                    metric_values["runtime"] = runtime
                elif metric == "cost":
                    metric_values["cost"] = outcome.cost

            competitor_results[competitor.name] = ExperimentMeasurement(
                metrics=metric_values,
                details=outcome.details,
            )

        return ExperimentResult(
            params=resolved_params,
            data=data,
            competitor_results=competitor_results,
        )
