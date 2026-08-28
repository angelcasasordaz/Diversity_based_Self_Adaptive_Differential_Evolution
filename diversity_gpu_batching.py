"""Batched numerical kernels for Diversity custom optimizers.

All estimator fitting and fitness evaluation remains on CPU. Only dense
population math crosses the optional CuPy boundary.
"""
from __future__ import annotations

import itertools
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import wait

import numpy as np

from numerical_backend import ComputeBackend, GPUBackendError


_REMOTE_GPU_CONNECTION = None
_REMOTE_GPU_TIMEOUT = 10.0
_REMOTE_GPU_DEBUG = False
_REMOTE_REQUEST_COUNTER = itertools.count()
_LOCAL_GPU_WORKER_BACKEND = None


def configure_local_gpu_worker(device_id=0, memory_fraction=0.85) -> None:
    """Initialize and retain one local CUDA backend for a persistent run worker."""
    global _LOCAL_GPU_WORKER_BACKEND
    _LOCAL_GPU_WORKER_BACKEND = ComputeBackend(
        "gpu", device_id=device_id, memory_fraction=memory_fraction,
    )
    if not _LOCAL_GPU_WORKER_BACKEND.uses_gpu:
        raise GPUBackendError(
            _LOCAL_GPU_WORKER_BACKEND.fallback_reason or "CUDA/runtime backend is unavailable"
        )


def configure_remote_gpu_client(connection_pool, timeout=10.0, debug=False) -> None:
    """Give each spawned run worker its own bounded GPU-service channel."""
    global _REMOTE_GPU_CONNECTION, _REMOTE_GPU_TIMEOUT, _REMOTE_GPU_DEBUG
    _REMOTE_GPU_CONNECTION = connection_pool.get(timeout=timeout)
    _REMOTE_GPU_TIMEOUT = float(timeout)
    _REMOTE_GPU_DEBUG = bool(debug)


class GPUServiceUnavailable(RuntimeError):
    """The required GPU service failed."""


class RemoteGPUClient:
    """Deadline-bound proxy preserving a run's numerical call order."""

    device = "gpu"
    uses_gpu = True

    def __init__(self):
        self.connection = _REMOTE_GPU_CONNECTION
        self.timeout = _REMOTE_GPU_TIMEOUT
        self.debug = _REMOTE_GPU_DEBUG
        self._send_queue = queue.Queue(maxsize=2)
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._sender.start()

    def _send_loop(self):
        while True:
            item = self._send_queue.get()
            if item is None:
                return
            try:
                self.connection.send(item)
            except (BrokenPipeError, EOFError, OSError):
                return

    def call(self, operation: str, *args):
        request_id = f"{os.getpid()}:{next(_REMOTE_REQUEST_COUNTER)}"
        submitted = time.monotonic()
        try:
            self._send_queue.put((request_id, operation, args, submitted), timeout=self.timeout)
        except queue.Full as exc:
            raise GPUServiceUnavailable("GPU request queue remained full") from exc
        if self.debug:
            print(f"GPU request submitted id={request_id} operation={operation} queue_depth={self._send_queue.qsize()}", flush=True)
        if not self.connection.poll(self.timeout):
            raise GPUServiceUnavailable(
                f"GPU request {request_id} exceeded {self.timeout:.1f}s deadline"
            )
        try:
            response_id, succeeded, payload = self.connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise GPUServiceUnavailable("GPU service connection closed") from exc
        if response_id != request_id:
            raise GPUServiceUnavailable(
                f"GPU response mismatch: expected {request_id}, received {response_id}"
            )
        latency = time.monotonic() - submitted
        if self.debug:
            print(f"GPU request completed id={request_id} operation={operation} latency={latency:.4f}s", flush=True)
        if not succeeded:
            raise GPUServiceUnavailable(payload)
        return payload


def serve_gpu_requests(backend: ComputeBackend, connections, stop_event, debug=False) -> None:
    """Concurrently dispatch independent requests through one CUDA owner/context."""
    if backend.uses_gpu:
        backend.xp.cuda.Device(backend.device_id).use()
    batcher = object.__new__(DiversityMathBatcher)
    batcher.backend = backend
    batcher.remote = None

    pending = 0
    pending_lock = threading.Lock()

    def execute(connection, request):
        nonlocal pending
        request_id, operation, args, submitted = request
        try:
            if backend.uses_gpu:
                backend.xp.cuda.Device(backend.device_id).use()
            result = getattr(batcher, operation)(*args)
        except Exception as exc:
            response = (request_id, False, f"{type(exc).__name__}: {exc}")
        else:
            response = (request_id, True, result)
        try:
            connection.send(response)
        except (BrokenPipeError, EOFError, OSError):
            pass
        finally:
            with pending_lock:
                pending -= 1
                depth = pending
            if debug:
                latency = time.monotonic() - submitted
                print(f"GPU request serviced id={request_id} queue_depth={depth} latency={latency:.4f}s", flush=True)

    # Several host dispatch threads may enqueue work/transfer data concurrently,
    # while all of them use the same CUDA context owned by this process.
    with ThreadPoolExecutor(max_workers=max(1, min(4, len(connections)))) as pool:
        active = list(connections)
        while active and not stop_event.is_set():
            for connection in wait(active, timeout=0.05):
                try:
                    request = connection.recv()
                except (EOFError, OSError):
                    active.remove(connection)
                    continue
                with pending_lock:
                    pending += 1
                    depth = pending
                if debug:
                    print(f"GPU request accepted id={request[0]} queue_depth={depth}", flush=True)
                pool.submit(execute, connection, request)


def start_gpu_request_service(device_id, memory_fraction, connections, stop_event, debug=False):
    """Spawn target: create CUDA only in the killable service process."""
    backend = ComputeBackend("gpu", device_id=device_id, memory_fraction=memory_fraction)
    serve_gpu_requests(backend, connections, stop_event, debug)


class DiversityMathBatcher:
    """Execute useful custom-optimizer math on NumPy or optional CuPy."""

    def __init__(
        self,
        compute_device: str = "cpu",
        gpu_device_id: int = 0,
        gpu_memory_fraction: float = 0.85,
    ):
        self.requested_device = str(compute_device).lower()
        self.remote = None
        if str(compute_device).lower() == "gpu" and _REMOTE_GPU_CONNECTION is not None:
            self.backend = None
            self.remote = RemoteGPUClient()
        elif str(compute_device).lower() == "gpu" and _LOCAL_GPU_WORKER_BACKEND is not None:
            self.backend = _LOCAL_GPU_WORKER_BACKEND
        else:
            try:
                self.backend = ComputeBackend(
                    compute_device,
                    device_id=gpu_device_id,
                    memory_fraction=gpu_memory_fraction,
                )
                if self.requested_device == "gpu" and not self.backend.uses_gpu:
                    raise GPUBackendError(self.backend.fallback_reason or "CUDA/runtime backend is unavailable")
            except GPUBackendError as exc:
                if self.requested_device == "gpu":
                    raise
                print(
                    f"GPU numerical backend unavailable; this run is using CPU fallback. Reason: {exc}",
                    flush=True,
                )
                self.backend = ComputeBackend("cpu")

    @property
    def uses_gpu(self) -> bool:
        return self.remote is not None or self.backend.uses_gpu

    @property
    def effective_device(self) -> str:
        return "gpu" if self.remote is not None else self.backend.device

    def awad(self, population, lb=None, ub=None) -> float:
        """Match the existing AWAD definition; bounds are intentionally unused."""
        if self.remote is not None:
            return self._remote_or_cpu("awad", population, lb, ub)
        _ = lb, ub
        if not self.uses_gpu:
            return self._awad_cpu(population)

        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        npop, n_dims = pop.shape
        median = xp.median(pop, axis=0)
        div = xp.sum(xp.mean(xp.abs(pop - median), axis=0)) / max(n_dims, 1)
        unique_count = xp.unique(pop, axis=0).shape[0]
        non_repeat_percent = unique_count * 100.0 / max(npop, 1)
        std = xp.std(pop, axis=0)
        std = xp.where(std == 0, 1.0e-5, std)
        if npop <= 1:
            min_distance = xp.asarray(0.0)
        else:
            scaled = (pop[:, None, :] - pop[None, :, :]) / std
            distances = xp.sqrt(xp.sum(scaled * scaled, axis=-1))
            diagonal = xp.eye(npop, dtype=bool)
            min_distance = xp.min(xp.where(diagonal, xp.inf, distances))
            min_distance = xp.where(xp.isfinite(min_distance), min_distance, 0.0)
        penalty = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return self.backend.scalar(div * 0.1 * non_repeat_percent * penalty)

    @staticmethod
    def _awad_cpu(population) -> float:
        pop = np.asarray(population, dtype=float)
        npop, n_dims = pop.shape
        med_dim = np.median(pop, axis=0)
        div_dim = np.mean(np.abs(pop - med_dim), axis=0)
        div = float(np.sum(div_dim) / max(n_dims, 1))
        unique_count = np.unique(pop, axis=0).shape[0]
        non_repeat_percent = unique_count * 100.0 / max(npop, 1)
        std_devs = np.std(pop, axis=0)
        std_devs[std_devs == 0] = 1.0e-5
        if npop <= 1:
            min_distance = 0.0
        else:
            min_distance = np.inf
            for idx in range(npop - 1):
                diff = (pop[idx + 1:] - pop[idx]) / std_devs
                distances = np.sqrt(np.sum(diff * diff, axis=1))
                if distances.size:
                    min_distance = min(min_distance, float(np.min(distances)))
            if not np.isfinite(min_distance):
                min_distance = 0.0
        penalty = ((min_distance + 0.1) ** 2) / (1.0 + min_distance**2)
        return float(div * 0.1 * non_repeat_percent * penalty)

    def covariance_inverse(self, population, n_dims: int) -> np.ndarray:
        if self.remote is not None:
            return self._remote_or_cpu("covariance_inverse", population, n_dims)
        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        sigma = xp.cov(pop, rowvar=False)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        if sigma.shape != (n_dims, n_dims):
            sigma = xp.eye(n_dims, dtype=xp.float64) * 1.0e-6
        sigma = (sigma + sigma.T) / 2.0 + 1.0e-6 * xp.eye(n_dims, dtype=xp.float64)
        try:
            chol = xp.linalg.cholesky(sigma)
            identity = xp.eye(n_dims, dtype=xp.float64)
            inverse = xp.linalg.solve(chol.T, xp.linalg.solve(chol, identity))
        except xp.linalg.LinAlgError:
            inverse = xp.linalg.pinv(sigma)
        return self.backend.to_cpu(inverse)

    def mahalanobis_distances(self, population, n_dims: int) -> np.ndarray:
        if self.remote is not None:
            return self._remote_or_cpu("mahalanobis_distances", population, n_dims)
        xp = self.backend.xp
        pop = self.backend.asarray(population, dtype=xp.float64)
        sigma = xp.cov(pop, rowvar=False)
        if sigma.ndim == 0:
            sigma = sigma.reshape(1, 1)
        if sigma.shape != (n_dims, n_dims):
            sigma = xp.eye(n_dims, dtype=xp.float64) * 1.0e-6
        sigma = (sigma + sigma.T) / 2.0 + 1.0e-6 * xp.eye(n_dims, dtype=xp.float64)
        try:
            chol = xp.linalg.cholesky(sigma)
            identity = xp.eye(n_dims, dtype=xp.float64)
            inverse = xp.linalg.solve(chol.T, xp.linalg.solve(chol, identity))
        except xp.linalg.LinAlgError:
            inverse = xp.linalg.pinv(sigma)
        centered = pop - xp.mean(pop, axis=0)
        distances = xp.sum((centered @ inverse) * centered, axis=1)
        return self.backend.to_cpu(distances)

    def mutate(self, x1, x2, x3, factor) -> np.ndarray:
        if self.remote is not None:
            return self._remote_or_cpu("mutate", x1, x2, x3, factor)
        if not self.uses_gpu:
            return np.asarray(x1) + np.asarray(factor) * (np.asarray(x2) - np.asarray(x3))
        xp = self.backend.xp
        result = (
            self.backend.asarray(x1, dtype=xp.float64)
            + self.backend.asarray(factor, dtype=xp.float64)
            * (
                self.backend.asarray(x2, dtype=xp.float64)
                - self.backend.asarray(x3, dtype=xp.float64)
            )
        )
        return self.backend.to_cpu(result)

    def crossover(self, parent, mutant, mask) -> np.ndarray:
        if self.remote is not None:
            return self._remote_or_cpu("crossover", parent, mutant, mask)
        if not self.uses_gpu:
            trial = np.asarray(parent).copy()
            trial[np.asarray(mask, dtype=bool)] = np.asarray(mutant)[np.asarray(mask, dtype=bool)]
            return trial
        xp = self.backend.xp
        trial = xp.where(
            self.backend.asarray(mask, dtype=bool),
            self.backend.asarray(mutant, dtype=xp.float64),
            self.backend.asarray(parent, dtype=xp.float64),
        )
        return self.backend.to_cpu(trial)

    def array_operation(self, kind, name, method, args, kwargs):
        """Execute one ordered NumPy-compatible array operation on the backend."""
        if self.remote is not None:
            return self._remote_or_cpu("array_operation", kind, name, method, args, kwargs)
        if not self.uses_gpu:
            target = getattr(np, name)
            return getattr(target, method)(*args, **kwargs) if kind == "ufunc" else target(*args, **kwargs)
        xp = self.backend.xp

        def device_value(value):
            if isinstance(value, np.ndarray):
                return self.backend.asarray(value)
            if isinstance(value, (list, tuple)):
                return type(value)(device_value(item) for item in value)
            return value

        device_args = tuple(device_value(value) for value in args)
        target = self._resolve_gpu_operation(name)
        result = getattr(target, method)(*device_args, **kwargs) if kind == "ufunc" else target(*device_args, **kwargs)

        def host_value(value):
            if isinstance(value, tuple):
                return tuple(host_value(item) for item in value)
            return self.backend.to_cpu(value) if isinstance(value, xp.ndarray) else value

        return host_value(result)

    def _resolve_gpu_operation(self, name):
        """Resolve NumPy operations first, then supported SciPy special functions."""
        xp = self.backend.xp
        target = getattr(xp, name, None)
        if target is not None:
            return target

        from cupyx.scipy import special

        target = getattr(special, name, None)
        if target is not None:
            return target
        raise GPUBackendError(
            f"GPU operation '{name}' is unsupported by CuPy and cupyx.scipy.special"
        )

    def _remote_or_cpu(self, operation, *args):
        """Fail the comparison if its shared GPU service becomes unavailable."""
        try:
            return self.remote.call(operation, *args)
        except GPUServiceUnavailable as exc:
            raise GPUBackendError(
                f"GPU execution failed for operation '{operation}': {exc}"
            ) from exc
