"""Warm-up PINN: recover known physics from synthetic data.

WHY THIS FILE EXISTS
--------------------
Before touching real battery data, generate capacity curves *from the SEI growth
law itself* with a known rate constant, then check that the PINN recovers that
constant. If the model cannot recover parameters from data it generated, the bug
is in the code — not in the battery, not in the data cleaning, not in the
hyperparameters. This collapses the search space for every bug that follows.

It is also the smallest complete example of the pattern used everywhere later:
a network of time, a data loss, and a physics residual built with autograd.

    python src/warmup_pinn.py

THE PHYSICS
-----------
Diffusion-limited SEI growth, solved in u = delta^2 to remove the 1/delta
singularity (decisions.md D-009):

    du/dt = 2 * k_ref * exp( -(Ea/R) * (1/T - 1/T_ref) )

Arrhenius is written relative to a reference temperature so that the exponent is
O(1) and `k_ref` is the growth rate *at* T_ref. Written the textbook way as
exp(-Ea/RT), the factor is ~1e-9 at room temperature and `k_ref` would have to
be ~1e7 to compensate — the optimiser cannot work across that many decades.
This reparametrisation is not cosmetic; it is the difference between converging
and not.

Capacity follows the film, with the initial condition baked in so SOH(0) = 1
exactly rather than approximately:

    delta(t) = sqrt(u(t))
    Q(t)     = Q0 - c_Li * (delta(t) - delta0)

A NOTE ON IDENTIFIABILITY (this matters later)
----------------------------------------------
There is an approximate scaling degeneracy in this model: rescale u -> s*u and
c_Li -> c_Li/sqrt(s) and the long-time capacity curve is nearly unchanged, since
only the product c_Li*sqrt(2*k_ref) sets the slope against sqrt(t). Gauge-fixing
delta0 = 1 plus the exact initial condition breaks it in principle.

Whether it is broken *in practice* turns out to depend entirely on lambda_phys,
and not on adding more observables:

    lambda_phys = 1     k_ref spans 7x across initialisations   (physics ignored)
    lambda_phys = 1e2   k_ref recovered to within 3%, any init  (balanced)
    lambda_phys = 1e4   k_ref spans 3x again, RMSE 8x worse     (data ignored)

So the degeneracy is soft, and a correctly weighted physics residual is enough
to resolve it. This was measured, not assumed -- see decisions.md D-012.

The useful diagnostic that falls out: agreement between independently
initialised runs is a *test* for whether the physics is actually constraining
anything. Two seeds landing on different parameters with equally good fits means
the physics term is decorative.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

R_GAS = 8.314462618  # J / (mol K)


@dataclass
class Config:
    """Single source of truth for the run. Dumped alongside every result so a
    figure can always be traced back to the settings that produced it."""

    # --- synthetic ground truth (what the PINN must recover) ---
    q0_ah: float = 2.0           # rated capacity, Ah
    k_ref_true: float = 0.005    # SEI rate at T_ref, in u-units per hour
    c_li_true: float = 0.1675    # Ah of lithium lost per unit film thickness
    delta0: float = 1.0          # gauge-fixed initial film thickness (see above)
    t_max_h: float = 2000.0      # length of the synthetic experiment, hours
    n_obs: int = 120             # measured cycles
    noise_ah: float = 0.008      # sensor noise std on capacity, Ah

    # --- fixed physical constants ---
    ea_j_per_mol: float = 50_000.0   # HELD FIXED, not fitted — decisions.md D-006
    t_ref_k: float = 297.15          # 24 C reference
    t_cell_k: float = 297.15         # isothermal in this warm-up

    # --- training ---
    n_collocation: int = 256
    hidden: int = 32
    depth: int = 3               # >= 3 hidden layers is a brief requirement
    lr: float = 5e-3
    epochs: int = 4000
    # Selected empirically, not guessed -- see decisions.md D-012. At 1.0 the
    # data loss outvotes the physics and the fitted parameters depend on their
    # initialisation; at 1e4 the reverse. 100 is the basin where independent
    # initialisations agree AND held-out RMSE is minimised.
    lambda_phys: float = 100.0
    seed: int = 0


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a result. Without this, 'the model got
    worse' is indistinguishable from 'the model got a different seed'."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def arrhenius(cfg: Config, t_kelvin: torch.Tensor | float) -> torch.Tensor | float:
    """Temperature factor, normalised to 1.0 at T_ref."""
    if isinstance(t_kelvin, torch.Tensor):
        return torch.exp(-(cfg.ea_j_per_mol / R_GAS) * (1.0 / t_kelvin - 1.0 / cfg.t_ref_k))
    return math.exp(-(cfg.ea_j_per_mol / R_GAS) * (1.0 / t_kelvin - 1.0 / cfg.t_ref_k))


def make_synthetic(cfg: Config):
    """Generate capacity observations from the exact analytic solution.

    Because du/dt is constant at fixed temperature, u(t) = u0 + 2*k_ref*A*t and
    the whole curve is closed-form. No ODE solver, so the ground truth carries
    no integration error to confuse the comparison.
    """
    a = arrhenius(cfg, cfg.t_cell_k)
    u0 = cfg.delta0 ** 2

    t = np.linspace(0.0, cfg.t_max_h, cfg.n_obs)
    u = u0 + 2.0 * cfg.k_ref_true * a * t
    delta = np.sqrt(u)
    q_clean = cfg.q0_ah - cfg.c_li_true * (delta - cfg.delta0)

    rng = np.random.default_rng(cfg.seed)
    q_noisy = q_clean + rng.normal(0.0, cfg.noise_ah, size=t.shape)

    return t, q_clean, q_noisy


class SEIPinn(nn.Module):
    """Predicts the film state u(t); capacity is derived from it, not predicted.

    This ordering is deliberate. If the network predicted capacity directly and
    the physics were merely a penalty, nothing would stop it from emitting a
    capacity curve with no consistent film behind it. Making u the only free
    field means every capacity value the model can express corresponds to some
    real film trajectory.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = [nn.Linear(1, cfg.hidden), nn.Tanh()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), nn.Tanh()]
        layers += [nn.Linear(cfg.hidden, 1)]
        self.net = nn.Sequential(*layers)

        # Physical parameters are stored as logs so they cannot go negative,
        # whatever the optimiser does. A negative rate constant or a negative
        # lithium coefficient is not something we want to merely penalise --
        # exp() makes it unrepresentable.
        self.log_k_ref = nn.Parameter(torch.tensor(math.log(1e-3)))
        self.log_c_li = nn.Parameter(torch.tensor(math.log(0.1)))

    @property
    def k_ref(self) -> torch.Tensor:
        return torch.exp(self.log_k_ref)

    @property
    def c_li(self) -> torch.Tensor:
        return torch.exp(self.log_c_li)

    def u(self, tau: torch.Tensor) -> torch.Tensor:
        """Film state u = delta^2 at normalised time tau in [0, 1].

        Time is normalised because a tanh network fed raw hours (O(1e3))
        saturates on the first layer and stops learning.

        softplus keeps u > u0: the film can never be thinner than it started.
        This is a hard constraint on *positivity*. It is NOT yet monotonicity --
        u may still dip between two interior times, and watching that happen is
        part of the point of the later ablations.

        The leading `tau` factor is what makes the initial condition exact.
        Without it, softplus(.) > 0 everywhere including tau = 0, so u(0) > u0
        and SOH(0) came out at 0.996 instead of 1. Multiplying by tau forces the
        whole correction to vanish at t = 0 regardless of what the network says.
        This is the standard hard-IC ansatz: the boundary condition is satisfied
        by the functional form, so the optimiser never gets a vote on it.
        """
        u0 = self.cfg.delta0 ** 2
        return u0 + tau * nn.functional.softplus(self.net(tau))

    def capacity(self, tau: torch.Tensor) -> torch.Tensor:
        """Q(t) = Q0 - c_Li * (delta(t) - delta0), so Q(0) = Q0 by construction."""
        delta = torch.sqrt(self.u(tau))
        return self.cfg.q0_ah - self.c_li * (delta - self.cfg.delta0)


def physics_residual(model: SEIPinn, tau: torch.Tensor) -> torch.Tensor:
    """du/dt - 2*k_ref*A(T), evaluated at collocation points by autodiff.

    This is the PINN core. du/dtau comes from torch.autograd.grad -- an exact
    derivative of the network, not a finite difference -- and is converted back
    to physical time by dividing by t_max.

    Keeping the residual in physical units (per hour) rather than per-tau
    matters: it puts the residual at the same order of magnitude as the capacity
    error, so lambda_phys = 1 actually balances the two terms. Left in tau-units
    the residual would be ~1e3x larger and would swamp the data loss regardless
    of the weight.
    """
    cfg = model.cfg
    tau = tau.requires_grad_(True)
    u = model.u(tau)

    du_dtau = torch.autograd.grad(
        u, tau,
        grad_outputs=torch.ones_like(u),
        create_graph=True,   # needed: this derivative is itself part of the loss
    )[0]
    du_dt = du_dtau / cfg.t_max_h

    target = 2.0 * model.k_ref * arrhenius(cfg, cfg.t_cell_k)
    return du_dt - target


def train(cfg: Config, verbose: bool = True):
    set_seed(cfg.seed)
    t_np, q_clean, q_noisy = make_synthetic(cfg)

    tau_obs = torch.tensor(t_np / cfg.t_max_h, dtype=torch.float32).reshape(-1, 1)
    q_obs = torch.tensor(q_noisy, dtype=torch.float32).reshape(-1, 1)
    tau_col = torch.linspace(0, 1, cfg.n_collocation).reshape(-1, 1)

    model = SEIPinn(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        loss_data = torch.mean((model.capacity(tau_obs) - q_obs) ** 2)
        loss_phys = torch.mean(physics_residual(model, tau_col.clone()) ** 2)
        loss = loss_data + cfg.lambda_phys * loss_phys
        loss.backward()
        opt.step()

        if epoch % 500 == 0 or epoch == cfg.epochs - 1:
            history.append((epoch, loss_data.item(), loss_phys.item()))
            if verbose:
                print(
                    f"  epoch {epoch:5d}  data={loss_data.item():.3e}  "
                    f"phys={loss_phys.item():.3e}  "
                    f"k_ref={model.k_ref.item():.5f}  c_Li={model.c_li.item():.5f}"
                )

    return model, (t_np, q_clean, q_noisy), history


def check_physics(model: SEIPinn, cfg: Config) -> list[str]:
    """Physical invariants. These are assertions about reality, not about code,
    and every one of them has a corresponding way for the model to be wrong."""
    failures = []
    with torch.no_grad():
        tau = torch.linspace(0, 1, 500).reshape(-1, 1)
        q = model.capacity(tau).squeeze().numpy()
        u = model.u(tau).squeeze().numpy()
        delta = np.sqrt(u)

    if not np.isfinite(q).all():
        failures.append("capacity contains NaN or inf")
    if abs(q[0] - cfg.q0_ah) > 1e-4:
        failures.append(f"SOH(0) != 1: Q(0) = {q[0]:.6f}, expected {cfg.q0_ah}")
    if (delta <= 0).any():
        failures.append("film thickness went non-positive")
    if (q < 0).any():
        failures.append("capacity went negative")

    # Not asserted, only counted: the soft-constraint model is *allowed* to
    # produce a thinning film. Measuring how often it does is the baseline that
    # the hard-constrained versions are later compared against.
    n_viol = int((np.diff(delta) < -1e-9).sum())
    print(f"  film-thinning steps (soft model, expected > 0): {n_viol} / {len(delta) - 1}")

    return failures


def identifiable_combination(k_ref: float, c_li: float) -> float:
    """The quantity capacity data can actually constrain.

    At long times delta -> sqrt(2*k_ref*A*t), so

        Q(t) ~ Q0 - c_Li * sqrt(2*k_ref*A) * sqrt(t)

    and only the *product* c_Li*sqrt(2*k_ref) shapes the curve. Any (k_ref,
    c_Li) pair holding that product constant fits the data equally well. Fixing
    delta0 = 1 nominally breaks the tie through the early-time curvature, but
    that is a weak lever and sensor noise largely erases it.

    Tracking this separately from the individual parameters is what turns
    "the fit is bad" into "the fit is fine, the parameters are unidentifiable" --
    two very different problems with two very different fixes.
    """
    return c_li * math.sqrt(2.0 * k_ref)


def main() -> int:
    cfg = Config()
    seeds = [0, 1, 2, 3, 4]

    print("Warm-up PINN — recovering known physics from synthetic data")
    print(f"  config: {asdict(cfg)}\n")

    true_comb = identifiable_combination(cfg.k_ref_true, cfg.c_li_true)
    rows, failures = [], []

    for seed in seeds:
        cfg_s = Config(**{**asdict(cfg), "seed": seed})
        print(f"--- seed {seed} ---")
        model, (t_np, q_clean, _), _ = train(cfg_s, verbose=False)

        with torch.no_grad():
            tau = torch.tensor(t_np / cfg_s.t_max_h, dtype=torch.float32).reshape(-1, 1)
            q_pred = model.capacity(tau).squeeze().numpy()
        rmse = float(np.sqrt(np.mean((q_pred - q_clean) ** 2)))

        k_rec, c_rec = model.k_ref.item(), model.c_li.item()
        rows.append((seed, k_rec, c_rec, identifiable_combination(k_rec, c_rec), rmse))
        failures += [f"seed {seed}: {f}" for f in check_physics(model, cfg_s)]

    print("\n" + "=" * 74)
    print("PARAMETER RECOVERY ACROSS SEEDS")
    print("=" * 74)
    print(f"{'seed':>5} {'k_ref':>10} {'c_Li':>10} {'c_Li*sqrt(2k)':>15} {'RMSE (Ah)':>11}")
    for seed, k, c, comb, rmse in rows:
        print(f"{seed:>5} {k:>10.5f} {c:>10.5f} {comb:>15.5f} {rmse:>11.5f}")

    ks = np.array([r[1] for r in rows])
    cs = np.array([r[2] for r in rows])
    combs = np.array([r[3] for r in rows])
    print(f"{'true':>5} {cfg.k_ref_true:>10.5f} {cfg.c_li_true:>10.5f} "
          f"{true_comb:>15.5f} {'--':>11}")
    print("-" * 74)

    def spread(name: str, arr: np.ndarray, truth: float) -> None:
        cv = 100 * arr.std() / arr.mean()
        bias = 100 * abs(arr.mean() - truth) / truth
        print(f"  {name:<16} mean {arr.mean():>9.5f}   "
              f"seed spread {cv:>5.1f}%   bias vs truth {bias:>5.1f}%")

    spread("k_ref", ks, cfg.k_ref_true)
    spread("c_Li", cs, cfg.c_li_true)
    spread("c_Li*sqrt(2k)", combs, true_comb)

    print("\nPhysical checks")
    for f in failures:
        print(f"  FAIL: {f}")
    if not failures:
        print("  all invariants hold, all seeds")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
