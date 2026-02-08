import argparse
from pathlib import Path
import datetime
import yaml
import numpy as np
import tidy3d as td

from src.tidy3d.simulation import (
    build_part1_context,
    make_sim_base,
    build_mode_source,
    make_terminal_monitor,
    build_flux_monitors,
    make_sim_full,
)

from src.tidy3d.optimizer import optimize_notebook_style


def infer_regime(sf: float) -> str:
    if sf < 0.3:
        raise ValueError("scale_factor must be >= 0.3")
    if sf < 0.6:
        return "insufficient"
    if sf <= 0.8:
        return "stable"
    return "redundant"


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config is empty: {path}")
    return cfg


def init_params(nx: int, ny: int, mode: str, seed: int = 0):
    n = nx * ny
    if mode == "all05":
        return np.full((n,), 0.5, dtype=float)
    if mode == "all1":
        return np.ones((n,), dtype=float)
    if mode == "all0":
        return np.zeros((n,), dtype=float)
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.random((n,), dtype=float)
    raise ValueError(f"Unknown init mode: {mode}")


def compute_layout(cfg, sf: float):
    pix = cfg["design_region"]["pixel_size"]
    nx0 = cfg["design_region"]["nx_orig"]
    ny0 = cfg["design_region"]["ny_orig"]

    nx = int(nx0 * sf)
    ny = int(ny0 * sf)
    design_x = nx * pix
    design_y = ny * pix

    port_offset_x = cfg["ports"].get("output_plane_x_offset", 0.8)
    x_margin = cfg["sim_size"].get("x_margin", 1.0)
    y_margin = cfg["sim_size"].get("y_margin", 2.0)

    Lx = design_x + 2.0 * port_offset_x + x_margin
    Ly = design_y + y_margin
    return nx, ny, design_x, design_y, Lx, Ly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scale_factor", type=float, required=True)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--init", type=str, default="all05", choices=["all05", "all1", "all0", "random"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sf = args.scale_factor
    regime = infer_regime(sf)

    # ---- run dir ----
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{ts}_sf{sf:.2f}_{args.tag}_{args.init}"
    run_dir = Path(cfg["output"]["root"]) / regime / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- layout info (for record only) ----
    nx, ny, design_x, design_y, Lx, Ly = compute_layout(cfg, sf)
    params0 = init_params(nx, ny, args.init, seed=args.seed)

    cfg_out = dict(cfg)
    cfg_out["regime"] = {"name": regime, "scale_factor": sf, "run_id": run_id}
    cfg_out["layout"] = {"nx": nx, "ny": ny, "design_x": float(design_x), "design_y": float(design_y), "Lx": float(Lx), "Ly": float(Ly)}

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)

    print(f"[OK] regime={regime} sf={sf} init={args.init}")
    print(f"[OK] nx={nx} ny={ny} design_x={design_x:.3f} design_y={design_y:.3f} Lx={Lx:.3f} Ly={Ly:.3f}")
    print(f"[OK] run_dir={run_dir}")
    print(f"[OK] params0 shape={params0.shape} min={params0.min():.3f} max={params0.max():.3f}")

    # ============================================================
    # Part1: build ctx (use cfg as source of truth)
    # ============================================================
    waveguides = cfg.get("waveguides", {})
    mesh_cfg = cfg.get("mesh", {})
    flux_cfg = cfg.get("flux", {})

    ctx = build_part1_context(
        n_si=cfg["materials"]["n_si"],
        eps_min=cfg["materials"]["eps_min"],
        scale_factor=sf,
        wavelength_max=cfg["spectral"]["wavelength_max"],
        wavelength_min=cfg["spectral"]["wavelength_min"],
        delta_wl=cfg["spectral"]["delta_wl"],
        pixel_size=cfg["design_region"]["pixel_size"],
        nx_orig=cfg["design_region"]["nx_orig"],
        ny_orig=cfg["design_region"]["ny_orig"],

        # ---- defaults match your Part1/2 ----
        wg_width=float(waveguides.get("wg_width", 0.15)),
        wg_length_2=float(waveguides.get("wg_length_2", 6.5)),
        y_pos_out_orig=float(waveguides.get("y_pos_out_orig", 1.5)),

        port_offset_x=cfg["ports"].get("output_plane_x_offset", 0.8),
        x_margin=cfg["sim_size"].get("x_margin", 1.0),
        y_margin=cfg["sim_size"].get("y_margin", 2.0),

        min_steps_per_wvl=int(mesh_cfg.get("min_steps_per_wvl", 16)),
    )

    # sanity: ctx nx/ny should match compute_layout (same formula)
    if ctx["nx"] != nx or ctx["ny"] != ny:
        print(f"[WARN] ctx nx/ny ({ctx['nx']},{ctx['ny']}) != layout nx/ny ({nx},{ny})")

    # ============================================================
    # Part2: build source + monitors
    # ============================================================
    sim_ref = make_sim_base(np.random.random((ctx["nx"] * ctx["ny"],)), ctx=ctx, beta=1)

    mode_src, freq_mid, fwidth = build_mode_source(ctx=ctx, sim_ref=sim_ref, num_modes=1, mode_index=0)

    terminal_1 = make_terminal_monitor(
        ctx=ctx, sim_ref=sim_ref, direction="right",
        freqs=ctx["freqs_object"]["terminal_1"], name="terminal_1",
        num_modes=1, freq_mid=freq_mid
    )
    terminal_2 = make_terminal_monitor(
        ctx=ctx, sim_ref=sim_ref, direction="right",
        freqs=ctx["freqs_object"]["terminal_2"], name="terminal_2",
        num_modes=1, freq_mid=freq_mid
    )

    freq_max = td.C_0 / ctx["wavelength_min"]
    freq_min = td.C_0 / ctx["wavelength_max"]

    Nf = int(flux_cfg.get("Nf", 501))
    flux_mnts, _ = build_flux_monitors(
        terminal_1=terminal_1, terminal_2=terminal_2,
        freq_min=freq_min, freq_max=freq_max, fwidth=fwidth,
        Nf=Nf,
    )

    def make_sim(p, beta):
        return make_sim_full(
            params=p, beta=beta, ctx=ctx,
            mode_src=mode_src,
            terminal_1=terminal_1, terminal_2=terminal_2,
            flux_mnts=flux_mnts
        )

    # ============================================================
    # optimizer: optimize (A-spec directory rule)
    # ============================================================
    final_sim_dir = run_dir / "final" / "sim_data"
    final_sim_dir.mkdir(parents=True, exist_ok=True)
    task_name = f"{regime}_{run_id}_final"
    folder_name = str(final_sim_dir)

    params_final, history = optimize_notebook_style(
        params_init=params0,
        make_sim=make_sim,
        task_name=task_name,
        folder_name=folder_name,
        num_steps=int(cfg["optim"]["steps"]),
        ckpt_every=int(cfg["optim"]["save_every"]),
        lr_init=float(cfg["optim"]["adam"]["lr_init"]),
        decay_steps=int(cfg["optim"]["adam"]["decay_steps"]),
        decay_rate=float(cfg["optim"]["adam"]["decay_rate"]),
        beta_schedule=None,     # keep notebook behavior: beta=1
        run_dir=run_dir,        # will save checkpoints if enabled in optimizer.py
    )
    
    import pickle

    with open(run_dir / "final_params.pkl", "wb") as f:
        pickle.dump(np.asarray(params_final), f)

    with open(run_dir / "history.pkl", "wb") as f:
        pickle.dump(history, f)

    print("[OK] saved", run_dir / "final_params.pkl")
    print("[OK] saved", run_dir / "history.pkl")

    print("[DONE] Optimization finished.")
    print("[DONE] final params range:", float(params_final.min()), float(params_final.max()))
    print("[DONE] run_dir =", run_dir)


if __name__ == "__main__":
    main()
