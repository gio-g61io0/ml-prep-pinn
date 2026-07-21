"""Verification for the amortized Makilala EIL PINN architecture + residuals.

Run:  cd mlprep && venv/bin/python -m pinn_eil.check_residuals

Checks:
  1. Smoke: build the real network, sample REAL Makilala units, run all four residuals
     -> finite, shape (N, 1).
  2. Manufactured solution: swap in a known analytic u(z,t), Phi(z,t) (constant coeffs);
     verify the tape-computed derivatives AND assembled r_pde / r_bw / BC / IC match the
     hand-derived closed forms (relative L2 < 2e-4).
  3. Linear-elastic reduction (alpha=1): r_pde -> rho*u_tt - k*u_zz - rho*g; report c=sqrt(k/rho).
  4. Training smoke: a few epochs on real units -> loss finite and decreasing.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import tensorflow as tf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pinn_eil as P
    from pinn_eil import physics, train
else:
    from . import __init__ as _  # noqa
    import pinn_eil as P  # type: ignore
    from pinn_eil import physics, train  # type: ignore

RTOL = 2e-4


def _rel_l2(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / denom) if denom > 1e-30 else float(np.linalg.norm(a - b))


def _const_batch(keys, n, coeffs, z=None, t=None):
    """Build a batch dict with constant per-point coeffs plus optional z / t arrays."""
    b = {c: np.full((n, 1), coeffs[c], dtype=np.float32) for c in ("rho", "k", "eps_y", "H", "pga")}
    if z is not None:
        b["z"] = z.astype(np.float32)
    if t is not None:
        b["t"] = t.astype(np.float32)
    return b


# --- Manufactured analytic solution ----------------------------------------
class FakeAnalyticModel:
    """u = a sin(pi z/H) cos(w t);  Phi = b sin(pi z/H) sin(w t). Reads cols 0,1 of input."""

    def __init__(self, a, b, w, H):
        self.a, self.b, self.w, self.H = a, b, w, H

    def __call__(self, x, training=False):
        z, t = x[:, 0:1], x[:, 1:2]
        kz = math.pi / self.H
        u = self.a * tf.sin(kz * z) * tf.cos(self.w * t)
        phi = self.b * tf.sin(kz * z) * tf.sin(self.w * t)
        return {"u": u, "phi": phi}


def _analytic_fields(z, t, a, b, w, H):
    kz = math.pi / H
    sinz, cosz = np.sin(kz * z), np.cos(kz * z)
    cost, sint = np.cos(w * t), np.sin(w * t)
    return {
        "u": a * sinz * cost, "phi": b * sinz * sint,
        "u_z": a * kz * cosz * cost, "u_zz": -a * kz**2 * sinz * cost,
        "u_t": -a * w * sinz * sint, "u_tt": -a * w**2 * sinz * cost,
        "phi_z": b * kz * cosz * sint, "phi_t": b * w * sinz * cost,
        "eps_dot": -a * kz * w * cosz * sint,
    }


def _tape_derivatives(model, z_np, t_np, coeffs):
    z = tf.constant(z_np, tf.float32)
    t = tf.constant(t_np, tf.float32)
    c = {k: tf.constant(np.full_like(z_np, coeffs[k]), tf.float32)
         for k in ("rho", "k", "eps_y", "H", "pga")}
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(z)
        tape2.watch(t)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(z)
            tape1.watch(t)
            out = model(physics._model_input(z, t, c), training=False)
            u, phi = out["u"], out["phi"]
        u_z, u_t = tape1.gradient(u, z), tape1.gradient(u, t)
        phi_z, phi_t = tape1.gradient(phi, z), tape1.gradient(phi, t)
        del tape1
    u_zz, u_tt = tape2.gradient(u_z, z), tape2.gradient(u_t, t)
    eps_dot = tape2.gradient(u_z, t)
    del tape2
    return {"u": u, "phi": phi, "u_z": u_z, "u_t": u_t, "u_zz": u_zz,
            "u_tt": u_tt, "phi_z": phi_z, "phi_t": phi_t, "eps_dot": eps_dot}


def _an_r_pde(f, alpha, k, eps_y, rho, g):
    sigma_z = alpha * k * f["u_zz"] + (1 - alpha) * k * eps_y * f["phi_z"]
    return rho * f["u_tt"] - sigma_z - rho * g


def _an_r_bw(f, A, beta, gamma, n, eps_y):
    phi_abs = np.abs(f["phi"]) + physics._EPS
    ed = f["eps_dot"]
    rhs = (1.0 / eps_y) * (A * ed - beta * np.abs(ed) * phi_abs**(n - 1) * f["phi"]
                           - gamma * ed * phi_abs**n)
    return f["phi_t"] - rhs


def main() -> None:
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    ranges = P.load_norm_ranges()
    consts = P.load_global_consts()               # synthetic u_scale
    scales = P.load_scales(consts)
    from pinn_eil import config as CFG
    means = CFG.dataset_means()
    print(f"[consts] alpha={consts.alpha} g={consts.g} T={consts.T} u_scale={consts.u_scale:.4e}")
    print(f"[scales] U={scales.U:.3e} S_pde={scales.S_pde:.3e} S_bw={scales.S_bw:.3e} "
          f"S_surf={scales.S_surf:.3e}")

    failures: list[str] = []

    # --- 1. Smoke on the real network + real units ------------------------- #
    print("\n[1] Smoke test (real network, real Makilala units)")
    model = P.build_pinn(ranges, consts.T)
    sampler = P.UnitSampler(consts.T, seed=0)
    col = sampler.collocation(256)
    bc = sampler.boundary(128)
    ic = sampler.initial(128)
    checks = {
        "r_pde": physics.pde_residual(model, consts, col),
        "r_bw": physics.boucwen_residual(model, consts, col),
    }
    b = physics.bc_residual(model, consts, bc)
    i = physics.ic_residual(model, consts, ic)
    checks.update({"bc_base": b["base"], "bc_surface": b["surface"],
                   "ic_u": i["u"], "ic_u_t": i["u_t"], "ic_phi": i["phi"]})
    for name, r in checks.items():
        arr = r.numpy()
        ok = np.all(np.isfinite(arr)) and arr.shape[1] == 1
        print(f"    {name:12s} shape={arr.shape} finite={np.all(np.isfinite(arr))} "
              f"mean|.|={np.mean(np.abs(arr)):.3e}")
        if not ok:
            failures.append(f"smoke:{name}")

    # --- 2. Manufactured-solution correctness ------------------------------ #
    print("\n[2] Manufactured solution")
    coeffs = {"rho": means["rho"], "k": means["k"], "eps_y": means["eps_y"],
              "H": means["H"], "pga": 0.3}
    ct = replace(consts, u_scale=1.0)  # so u_phys == analytic u
    a, bb, w = 1e-3, 0.5, 2 * math.pi / 5.0
    fake = FakeAnalyticModel(a, bb, w, coeffs["H"])

    N = 400
    z_np = rng.uniform(0.0, coeffs["H"], (N, 1))
    t_np = rng.uniform(0.0, consts.T, (N, 1))
    f = _analytic_fields(z_np, t_np, a, bb, w, coeffs["H"])

    d = _tape_derivatives(fake, z_np, t_np, coeffs)
    for key in ["u", "phi", "u_z", "u_t", "u_zz", "u_tt", "phi_z", "phi_t", "eps_dot"]:
        err = _rel_l2(d[key].numpy(), f[key])
        print(f"    d/{key:8s} relL2={err:.2e} {'OK' if err < RTOL else 'FAIL'}")
        if err >= RTOL:
            failures.append(f"deriv:{key}={err:.1e}")

    col_b = _const_batch(None, N, coeffs, z=z_np, t=t_np)
    r_pde = physics.pde_residual(fake, ct, col_b).numpy()
    r_bw = physics.boucwen_residual(fake, ct, col_b).numpy()
    an_pde = _an_r_pde(f, ct.alpha, coeffs["k"], coeffs["eps_y"], coeffs["rho"], ct.g)
    an_bw = _an_r_bw(f, ct.A, ct.beta, ct.gamma, ct.n, coeffs["eps_y"])
    for name, tfv, anv in [("r_pde", r_pde, an_pde), ("r_bw", r_bw, an_bw)]:
        err = _rel_l2(tfv, anv)
        print(f"    {name:8s} relL2={err:.2e} {'OK' if err < RTOL else 'FAIL'}")
        if err >= RTOL:
            failures.append(f"residual:{name}={err:.1e}")

    # BC (base uses zero base motion so residual == analytic u(0,t)); surface stress.
    Nb = 200
    t_bc = rng.uniform(0.0, consts.T, (Nb, 1))
    bc_b = _const_batch(None, Nb, coeffs, t=t_bc)
    r_bc = physics.bc_residual(fake, ct, bc_b, u_input=P.zero_base_motion)
    fb = _analytic_fields(np.full_like(t_bc, coeffs["H"]), t_bc, a, bb, w, coeffs["H"])
    surf_an = ct.alpha * coeffs["k"] * fb["u_z"] + (1 - ct.alpha) * coeffs["k"] * coeffs["eps_y"] * fb["phi"]
    u0_an = _analytic_fields(np.zeros_like(t_bc), t_bc, a, bb, w, coeffs["H"])["u"]
    for name, err in [("bc_surface", _rel_l2(r_bc["surface"].numpy(), surf_an)),
                      ("bc_base", _rel_l2(r_bc["base"].numpy(), u0_an))]:
        print(f"    {name:12s} relL2={err:.2e} {'OK' if err < RTOL else 'FAIL'}")
        if err >= RTOL:
            failures.append(f"{name}={err:.1e}")

    # IC
    Ni = 200
    z_ic = rng.uniform(0.0, coeffs["H"], (Ni, 1))
    ic_b = _const_batch(None, Ni, coeffs, z=z_ic)
    r_ic = physics.ic_residual(fake, ct, ic_b)
    fic = _analytic_fields(z_ic, np.zeros_like(z_ic), a, bb, w, coeffs["H"])
    for key in ["u", "u_t", "phi"]:
        err = _rel_l2(r_ic[key].numpy(), fic[key])
        print(f"    ic_{key:4s} relL2={err:.2e} {'OK' if err < RTOL else 'FAIL'}")
        if err >= RTOL:
            failures.append(f"ic:{key}={err:.1e}")

    # --- 3. Linear-elastic reduction --------------------------------------- #
    print("\n[3] Linear-elastic reduction (alpha=1)")
    ct_lin = replace(ct, alpha=1.0)
    r_lin = physics.pde_residual(fake, ct_lin, col_b).numpy()
    an_lin = coeffs["rho"] * f["u_tt"] - coeffs["k"] * f["u_zz"] - coeffs["rho"] * ct_lin.g
    err_lin = _rel_l2(r_lin, an_lin)
    c = math.sqrt(coeffs["k"] / coeffs["rho"])
    print(f"    r_pde(alpha=1) relL2={err_lin:.2e} {'OK' if err_lin < RTOL else 'FAIL'}")
    print(f"    wave speed c=sqrt(k/rho)={c:.1f} m/s (FKMODEL soil Vs~167 -> plausible)")
    if err_lin >= RTOL:
        failures.append(f"linear={err_lin:.1e}")

    # --- 4. Training smoke ------------------------------------------------- #
    print("\n[4] Training smoke (real units, 60 epochs)")
    hist, _ = train.train(model, consts, scales, sampler, epochs=60, n_col=1024,
                          n_bc=256, n_ic=256, resample_every=30, log_every=20)
    l0, l1 = hist[0]["loss"], hist[-1]["loss"]
    ok = np.isfinite(l1) and l1 <= l0
    print(f"    loss {l0:.4e} -> {l1:.4e}  {'OK (finite, non-increasing)' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"train:{l0:.2e}->{l1:.2e}")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
