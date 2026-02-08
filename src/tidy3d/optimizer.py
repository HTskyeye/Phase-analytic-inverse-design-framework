import numpy as np
import autograd.numpy as anp
import tidy3d.web as web
from tidy3d.plugins.autograd.differential_operators import value_and_grad
import optax


def _get_monitor_freqs_from_amps(mnt_data):
    """
    For ModeMonitor data, frequency axis is usually mnt_data.amps.f
    (match your notebook habit).
    """
    if hasattr(mnt_data, "freqs"):
        return np.asarray(mnt_data.freqs)
    # common case: xarray-like
    return np.asarray(mnt_data.amps.f)


def measure_power(sim_data, *, terminal_names=("terminal_1", "terminal_2")) -> float:
    """
    Your notebook-style power measure:
      power = sum_k sum_f |amp(direction='+', mode_index=0, f)|^2  / 42

    Notes:
    - In your current pipeline, both terminals are on the "right" side,
      so we consistently take direction="+" (same as your else-branch).
    - Keep '/42' EXACTLY as your notebook (do not change normalization).
    """
    power = 0.0
    for name in terminal_names:
        mnt_data = sim_data[name]
        freqs = _get_monitor_freqs_from_amps(mnt_data)

        # emulate your original index-loop behavior (but simpler & equivalent)
        for f in freqs:
            amp = mnt_data.amps.sel(direction="+", mode_index=0, f=f)
            power = power + anp.abs(amp) ** 2

    return power / 42.0


def make_objective(*, make_sim, task_name: str, folder_name: str):
    """
    Create objective closure exactly like notebook style:
      objective(params, beta) -> (J, sim_data)
    Only required change: use web.run (NOT td.web.run).
    """

    def objective(params, beta, verbose=False):
        sim = make_sim(params, beta=beta)
        sim_data = web.run(sim, task_name=task_name, folder_name=folder_name, verbose=verbose)
        J = measure_power(sim_data)
        return J, sim_data  # has_aux=True will keep sim_data as aux

    return objective


def optimize_notebook_style(
    *,
    params_init,
    make_sim,
    task_name: str,
    folder_name: str,
    num_steps: int,
    ckpt_every: int,
    lr_init: float,
    decay_steps: int,
    decay_rate: float,
    beta_schedule=None,
    run_dir=None,
):
    """
    Notebook-like optimizer:
      - maximize J via applying updates with -gradient
      - clamp params into [0, 1]
      - optional checkpoint save
    """
    params = anp.array(params_init)

    if beta_schedule is None:
        # keep your current notebook behavior: beta fixed = 1
        def beta_schedule(step):
            return 1.0

    lr = optax.exponential_decay(
        init_value=lr_init,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        transition_begin=0,
    )
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)

    objective = make_objective(make_sim=make_sim, task_name=task_name, folder_name=folder_name)
    grad_fn = value_and_grad(objective, has_aux=True)

    history = {"J": [], "beta": [], "grad_norm": []}

    for step in range(num_steps):
        beta = float(beta_schedule(step))

        out = grad_fn(params, beta)

        # Most common: ((J, aux), grads)
        # Some versions: ((J, grads), aux)
        # We unpack first, then fix by type/shape.
        if isinstance(out[0], tuple):
            (J, aux), grads = out
        else:
            grads, (J, aux) = out

        # ---- FIX: if grads is actually SimulationData, swap with aux ----
        # (SimulationData has attribute 'simulation'; grads should be array-like of same shape as params)
        if hasattr(grads, "simulation"):
            grads, aux = aux, grads

        # ensure grads is array-like
        grads = anp.array(grads)

        # maximize J => descent on (-J) => apply updates with -grads (same as your notebook)
        updates, opt_state = opt.update(-grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # clamp [0, 1]
        params = anp.clip(params, 0.0, 1.0)

        gnorm = float(np.linalg.norm(np.asarray(grads)))

        history["J"].append(float(J))
        history["beta"].append(beta)
        history["grad_norm"].append(gnorm)

        print(f"[{step+1:04d}/{num_steps}] J={float(J):.6e} beta={beta:.2f} grad_norm={gnorm:.3e}")

        # optional checkpoint (only if run_dir is provided)
        if (run_dir is not None) and ((step + 1) % ckpt_every == 0 or (step + 1) == num_steps):
            from pathlib import Path
            import pickle
            run_dir = Path(run_dir)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            with open(run_dir / "checkpoints" / f"params_step{step+1:04d}.pkl", "wb") as f:
                pickle.dump(np.asarray(params), f)

    return np.asarray(params), history
