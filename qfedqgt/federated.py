"""Federated training loop, local updates, and aggregation rules.

One protocol serves every backend.  A client only sees measurement counts, and the
server only sees what the clients report.  All circuits of a training step are first
collected as :class:`Request` objects and then handed to an *executor*, so that an
emulator can run the circuits of all clients and all compared methods in a few jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from . import sim
from .model import VQC

__all__ = ["Request", "LocalBackend", "LocalExecutor", "SharedExecutor", "Client", "Method", "RunConfig", "METHODS",
           "run_federated", "random_effects_weights", "between_client_variance", "fisher_aggregate",
           "efficiency_ratio"]


# --------------------------------------------------------------------------- execution
@dataclass
class Request:
    """A batch of circuits sharing one template, to be run on one device.

    ``counts`` is filled by the executor with an array of shape (split, C, D): the
    outcome counts of ``split`` disjoint groups of ``shots / split`` shots.
    Circuits of requests with different ``group`` values are never packed into the same
    program, so that the programs of one federated run do not depend on the other runs
    that are executed with it.
    """
    device: object
    template: sim.Template
    A: np.ndarray
    shots: int
    split: int = 1
    tag: str = ""
    counts: np.ndarray | None = None
    group: int = 0

    @property
    def n_circuits(self) -> int:
        return self.A.shape[0]


class LocalBackend:
    """Exact simulation of the outcome distribution followed by multinomial sampling."""

    def __init__(self, noise: sim.NoiseModel | None = None, seed: int = 0):
        self.noise = noise or sim.NoiseModel()
        self.rng = np.random.default_rng(seed)

    def probs(self, template: sim.Template, A: np.ndarray) -> np.ndarray:
        if self.noise.is_noiseless:
            return np.abs(sim.statevector(template, A)) ** 2
        return sim.probabilities(sim.density(template, A, self.noise), self.noise, template.n)

    def counts(self, template: sim.Template, A: np.ndarray, shots: int, split: int = 1) -> np.ndarray:
        if shots % split:
            raise ValueError("shots must be divisible by split")
        p = self.probs(template, np.atleast_2d(A))
        return np.stack([sim.sample_counts(p, shots // split, self.rng) for _ in range(split)])


class LocalExecutor:
    """Runs every request on the :class:`LocalBackend` stored in ``request.device``."""

    def __init__(self):
        self.n_circuits = 0
        self.n_shots = 0

    def describe(self) -> dict:
        return dict(kind="local simulation (exact outcome distributions of the gate-level noise model, "
                         "multinomial sampling of the shots)")

    def run(self, requests: list) -> None:
        for req in requests:
            req.counts = req.device.counts(req.template, req.A, req.shots, req.split)
            self.n_circuits += req.n_circuits
            self.n_shots += req.n_circuits * req.shots


class SharedExecutor:
    """Lets several federated runs advance in lockstep and executes their circuits together.

    Every run works in its own thread and owns a view (``view(i)``) of this object.  When all
    runs have handed in the requests of their current step, the requests are executed in one
    call of the underlying executor, in the order of the run indices.  The order is fixed so
    that a restarted campaign builds the same programs and finds its results in the cache.
    On an emulator, the circuits of all runs then wait in the queue once per step.
    """

    def __init__(self, executor, parties: int):
        import threading

        self.executor = executor
        self.parties = parties
        self._pending = [None] * parties
        self._barrier = threading.Barrier(parties, action=self._flush)
        self.error = None
        self.steps = 0

    def _flush(self) -> None:
        requests = [req for part in self._pending for req in part]
        self._pending = [None] * self.parties
        try:
            self.executor.run(requests)
        except BaseException as exc:      # the barrier breaks and every run stops
            self.error = exc
            raise
        self.steps += 1

    def view(self, index: int) -> "_SharedView":
        return _SharedView(self, index)


class _SharedView:
    def __init__(self, shared: SharedExecutor, index: int):
        self.shared, self.index = shared, index

    def run(self, requests: list) -> None:
        import threading

        for req in requests:
            req.group = self.index
        self.shared._pending[self.index] = list(requests)
        try:
            self.shared._barrier.wait()
        except threading.BrokenBarrierError:
            raise RuntimeError(f"shared execution stopped: {self.shared.error!r}") from self.shared.error

    def __getattr__(self, name):          # usage counters of the underlying executor
        return getattr(self.shared.executor, name)


# --------------------------------------------------------------------------- configuration
@dataclass
class Client:
    X: np.ndarray
    y: np.ndarray
    device: object            # LocalBackend, or a device profile understood by the executor
    shots: int                # shots per circuit (even)
    name: str = ""


@dataclass(frozen=True)
class Method:
    """A federated scheme = local preconditioner + aggregation rule.

    ``precond``: ``"euclid"`` (plain gradient), ``"pure"`` (pure-state quantum natural
    gradient, device independent) or ``"bures"`` (mixed-state natural gradient with the
    client's own Bures metric).
    ``weights``: ``"data"`` (data shares), ``"shots"`` (data share times shots),
    ``"fisher"`` (parameter-wise empirical Fisher average), ``"invvar"`` (inverse
    variance) or ``"re"`` (bias-variance optimal weights with the estimated
    between-client variance).
    """
    name: str
    precond: str
    weights: str
    lr: float
    eps: float = 0.05


METHODS = {
    "QFedAvg": dict(precond="euclid", weights="data"),
    "QFedFisher": dict(precond="euclid", weights="fisher"),
    "QFedQGT": dict(precond="bures", weights="re"),
    # further baselines and ablations
    "QFedAvg-shots": dict(precond="euclid", weights="shots"),
    "QNG+data": dict(precond="pure", weights="data"),
    "Bures+data": dict(precond="bures", weights="data"),
    "Euclid+re": dict(precond="euclid", weights="re"),
    "QFedQGT-invvar": dict(precond="bures", weights="invvar"),
}


@dataclass
class RunConfig:
    rounds: int = 20
    batch: int = 8
    local_steps: int = 1
    echo_shots: int = 20              # shots of each purity (echo) circuit
    server_lr: float = 1.0
    fisher_delta: float = 0.01        # threshold of the QFedFisher mask
    p_max: float = 0.9                # cap of the estimated depolarizing strength (bounds the preconditioner)
    eval_shots: int = 100
    eval_every: int = 1
    seed: int = 0
    metric_inputs: int = 8            # number of minibatch inputs over which the metric is averaged
    lagged: bool = True               # weights of round t use the statistics reported up to round t - 1
    smoothing: float = 0.5            # weight of the newest report in the running (log-domain) variances


# --------------------------------------------------------------------------- aggregation rules
def between_client_variance(updates: np.ndarray, sigma2: np.ndarray, raw: bool = False) -> float:
    """Moment estimate of the between-client variance tau^2 of the client updates.

    With precisions ``a_k = 1 / sigma_k^2`` and ``Q = sum_k a_k |Delta_k - mean_a|^2``,
    ``E[Q] = (K - 1) + tau^2 (A - sum_k a_k^2 / A)`` with ``A = sum_k a_k``; ``sigma_k^2``
    and ``tau^2`` are total (trace) variances of the update vectors.  The estimate is
    truncated at zero unless ``raw`` is set.
    """
    a = 1.0 / np.maximum(sigma2, 1e-300)
    A = a.sum()
    mean = (a[:, None] * updates).sum(axis=0) / A
    Q = float((a * ((updates - mean) ** 2).sum(axis=1)).sum())
    denom = A - (a ** 2).sum() / A
    if denom <= 0:
        return 0.0
    est = (Q - (len(a) - 1)) / denom
    return est if raw else max(0.0, est)


def random_effects_weights(share: np.ndarray, sigma2: np.ndarray, zeta2: float) -> np.ndarray:
    """Minimizer of ``zeta^2 chi^2(w || p) + sum_k w_k^2 sigma_k^2`` on the simplex."""
    w = share / (zeta2 + share * sigma2)
    return w / w.sum()


def efficiency_ratio(share: np.ndarray, sigma2: np.ndarray) -> float:
    """Variance of data-share averaging divided by the minimal variance of a weighted average."""
    return float((share ** 2 * sigma2).sum() * (1.0 / sigma2).sum())


def fisher_aggregate(thetas: np.ndarray, grads: np.ndarray, share: np.ndarray, blocks, delta: float):
    """Parameter-wise empirical-Fisher average of Bhatia and Kais (QFedFisher).

    ``F_kj = g_kj^2`` is min-max normalized inside every layer of every client; parameters
    whose summed Fisher value is below ``delta`` fall back to the data-share average.
    """
    F = grads ** 2
    Fn = np.zeros_like(F)
    for b in blocks:
        b = list(b)
        lo = F[:, b].min(axis=1, keepdims=True)
        hi = F[:, b].max(axis=1, keepdims=True)
        Fn[:, b] = (F[:, b] - lo) / np.maximum(hi - lo, 1e-300)
    Fs = Fn.sum(axis=0)
    avg = (share[:, None] * thetas).sum(axis=0)
    out = np.where(Fs < delta, avg, (Fn * thetas).sum(axis=0) / np.maximum(Fs, 1e-300))
    return out, Fn


# --------------------------------------------------------------------------- client side
def _mean_metric(model: VQC, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Block-diagonal pure-state metric averaged over the inputs."""
    t = model.template
    Q = sim.pure_qgt(t, t.angles(X, theta))
    return geo.block_diagonal(np.real(Q).mean(axis=0), t.blocks)


@dataclass
class _ClientState:
    """Running state of one client for one method."""
    theta: np.ndarray
    sigma2: float = 0.0
    loss: float = 0.0
    grad: np.ndarray | None = None
    echo_hits: float = 0.0        # cumulative echo survival counts (device characterization)
    echo_total: float = 0.0
    p_hat: float = np.nan
    kappa_hat: float = 1.0
    pending: dict = field(default_factory=dict)


def _needs_purity(method: Method) -> bool:
    return method.precond == "bures"


def _prepare_step(model: VQC, client: Client, st: _ClientState, method: Method, cfg: RunConfig,
                  rng: np.random.Generator) -> list:
    """Requests of one local step: purity (echo) circuits and parameter-shift circuits."""
    t = model.template
    idx = rng.choice(len(client.y), size=min(cfg.batch, len(client.y)), replace=False)
    Xb, yb = client.X[idx], client.y[idx]
    reqs = [Request(client.device, t, model.gradient_angles(Xb, st.theta), client.shots, tag="grad")]
    if _needs_purity(method):
        e = t.echo()
        reqs.append(Request(client.device, e, e.angles(Xb, st.theta), cfg.echo_shots, tag="echo"))
    st.pending = dict(Xb=Xb, yb=yb, reqs=reqs)
    return reqs


def _finish_step(model: VQC, client: Client, st: _ClientState, method: Method, cfg: RunConfig) -> None:
    """Turn the measured counts into a local update and into the reported statistics."""
    t = model.template
    d, N = t.n_params, t.dim
    Xb, yb, reqs = st.pending["Xb"], st.pending["yb"], st.pending["reqs"]
    B = len(yb)
    P = np.eye(d)
    if method.precond in ("pure", "bures"):
        G = _mean_metric(model, Xb[: cfg.metric_inputs], st.theta)
        P = geo.regularized_inverse(G, method.eps)
    if _needs_purity(method):
        c = reqs[1].counts[0]
        st.echo_hits += float(c[:, 0].sum())
        st.echo_total += float(c.sum())
        gamma = st.echo_hits / st.echo_total
        st.p_hat = float(min(geo.depolarizing_from_purity(gamma, N), cfg.p_max))
        st.kappa_hat = float(geo.kappa(st.p_hat, N))
        P = P / st.kappa_hat
    gb, loss = model.per_sample_gradients(reqs[0].counts[0], yb, client.shots)   # (B, d), (B,)
    g = gb.mean(axis=0)
    step = -method.lr * P @ g
    # within-client variance of the step (shots and minibatch): squared standard error of the mean
    dev = (gb - g) @ P.T
    st.sigma2 += method.lr ** 2 * float(np.sum(dev ** 2)) / (B * max(B - 1, 1))
    losses = [loss.mean()]
    st.theta = st.theta + step
    st.loss += float(np.mean(losses)) / cfg.local_steps
    st.grad = g
    st.pending = {}


# --------------------------------------------------------------------------- server side
def _aggregate(method: Method, theta: np.ndarray, states: list, share: np.ndarray, shots: np.ndarray,
               blocks, cfg: RunConfig, memory: dict):
    """Server update of one method.

    For the precision-weighted rules the weights are *predictable*: with ``cfg.lagged`` the
    weights applied in a round are computed from the statistics that the clients reported in
    earlier rounds (running averages kept in ``memory``).  The first round then only collects
    statistics and leaves the parameters unchanged (calibration round).
    """
    deltas = np.array([st.theta - theta for st in states])
    sigma2 = np.array([st.sigma2 for st in states])
    zeta2 = np.nan
    if method.weights == "fisher":
        thetas = np.array([st.theta for st in states])
        grads = np.array([st.grad for st in states])
        agg, Fn = fisher_aggregate(thetas, grads, share, blocks, cfg.fisher_delta)
        w = Fn.sum(axis=1) / max(Fn.sum(), 1e-300)        # effective client weights (for the record)
        return theta + cfg.server_lr * (agg - theta), w, sigma2, zeta2
    if method.weights == "data":
        w = share.copy()
        return theta + cfg.server_lr * (w[:, None] * deltas).sum(axis=0), w, sigma2, zeta2
    if method.weights == "shots":
        w = share * shots
        w = w / w.sum()
        return theta + cfg.server_lr * (w[:, None] * deltas).sum(axis=0), w, sigma2, zeta2
    if method.weights not in ("invvar", "re"):
        raise ValueError(method.weights)
    s2_now = np.maximum(sigma2, 1e-300)
    z2_raw = 0.0
    if method.weights == "re":
        z2_raw = between_client_variance(deltas, s2_now, raw=True) * (1.0 - float((share ** 2).sum()))
    if cfg.lagged:
        w = memory.get("w")                     # determined before this round
        zeta2 = memory.get("zeta2", np.nan)
        # running statistics -> weights of the next round.  The variances are averaged in the
        # logarithmic domain and the between-client variance by the median of the untruncated
        # estimates, which keeps single outliers from changing the weights.
        rho = cfg.smoothing
        log_now = np.log(s2_now)
        memory["log_s2"] = log_now if "log_s2" not in memory else (1 - rho) * memory["log_s2"] + rho * log_now
        memory.setdefault("z2_raw", []).append(z2_raw)
        memory["zeta2"] = max(0.0, float(np.median(memory["z2_raw"])))
        memory["w"] = random_effects_weights(share, np.exp(memory["log_s2"]), memory["zeta2"])
        if w is None:                           # calibration round: statistics only
            return theta.copy(), share.copy(), sigma2, np.nan
    else:
        zeta2 = max(0.0, z2_raw)
        w = random_effects_weights(share, s2_now, zeta2)
    return theta + cfg.server_lr * (w[:, None] * deltas).sum(axis=0), w, sigma2, zeta2


def run_federated(model: VQC, clients: list, methods: list, cfg: RunConfig, theta0: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, executor=None, eval_device=None,
                  verbose: bool = False, on_round=None) -> dict:
    """Run all ``methods`` in lockstep for ``cfg.rounds`` rounds; returns ``{name: history}``.

    Every method starts from ``theta0`` and sees the same minibatches.  The circuits of
    all methods and clients of a step are executed together by ``executor``.
    """
    executor = executor or LocalExecutor()
    K = len(clients)
    sizes = np.array([len(c.y) for c in clients], float)
    share = sizes / sizes.sum()
    shots = np.array([c.shots for c in clients], float)
    t = model.template
    theta = {m.name: theta0.copy() for m in methods}
    carry = {m.name: [(0.0, 0.0)] * K for m in methods}       # cumulative echo statistics per client
    memory = {m.name: {} for m in methods}                    # running statistics of the server
    keys = ("round", "acc_exact", "loss_exact", "acc_shots", "weights", "sigma2", "zeta2", "p_hat",
            "kappa_hat", "train_loss", "theta")
    hist = {m.name: {k: [] for k in keys} for m in methods}
    for m in methods:
        hist[m.name]["theta"].append(theta0.copy())

    def evaluate(r):
        reqs = {}
        if eval_device is not None:
            for m in methods:
                reqs[m.name] = Request(eval_device, t, t.angles(X_test, theta[m.name]), cfg.eval_shots, tag="eval")
            executor.run(list(reqs.values()))
        for m in methods:
            h = hist[m.name]
            h["round"].append(r)
            h["acc_exact"].append(model.accuracy_exact(X_test, y_test, theta[m.name]))
            h["loss_exact"].append(model.loss_exact(X_test, y_test, theta[m.name]))
            if eval_device is not None:
                phi = reqs[m.name].counts[0] @ model.obs / cfg.eval_shots
                h["acc_shots"].append(float((model.predict(phi) == y_test).mean()))
            else:
                h["acc_shots"].append(np.nan)

    evaluate(0)
    for r in range(1, cfg.rounds + 1):
        states = {m.name: [_ClientState(theta[m.name].copy(), echo_hits=carry[m.name][k][0],
                                        echo_total=carry[m.name][k][1]) for k in range(K)] for m in methods}
        for step in range(cfg.local_steps):
            reqs = []
            for m in methods:
                for k, client in enumerate(clients):
                    rng = np.random.default_rng([cfg.seed, r, k, step])   # same minibatches for every method
                    reqs += _prepare_step(model, client, states[m.name][k], m, cfg, rng)
            executor.run(reqs)
            for m in methods:
                for k, client in enumerate(clients):
                    _finish_step(model, client, states[m.name][k], m, cfg)
        for m in methods:
            sts = states[m.name]
            new_theta, w, sigma2, zeta2 = _aggregate(m, theta[m.name], sts, share, shots, t.blocks, cfg, memory[m.name])
            theta[m.name] = new_theta
            carry[m.name] = [(st.echo_hits, st.echo_total) for st in sts]
            h = hist[m.name]
            h["weights"].append(w)
            h["sigma2"].append(sigma2)
            h["zeta2"].append(zeta2)
            h["p_hat"].append([st.p_hat for st in sts])
            h["kappa_hat"].append([st.kappa_hat for st in sts])
            h["train_loss"].append(float(np.dot(share, [st.loss for st in sts])))
            h["theta"].append(new_theta.copy())
        if r % cfg.eval_every == 0 or r == cfg.rounds:
            evaluate(r)
            if verbose:
                for m in methods:
                    h = hist[m.name]
                    print(f"[{m.name}] round {r:3d}  acc {h['acc_exact'][-1]:.3f}  loss {h['loss_exact'][-1]:.3f}"
                          f"  w {np.round(h['weights'][-1], 3)}", flush=True)
        if on_round is not None:
            on_round(r, theta, hist)
    out = {}
    for m in methods:
        out[m.name] = {k: np.array(v) for k, v in hist[m.name].items()}
        out[m.name]["method"] = m.name
        out[m.name]["lr"] = m.lr
    out["_usage"] = dict(n_circuits=getattr(executor, "n_circuits", None), n_shots=getattr(executor, "n_shots", None))
    return out
