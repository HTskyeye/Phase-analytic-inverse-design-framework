import argparse
from pathlib import Path
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

from src.tidy3d.postprocess import postprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--config", default=None, help="optional: path to base.yaml; if not set, use run_dir/config.yaml")
    ap.add_argument("--beta_final", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    # load cfg: prefer run_dir/config.yaml (recorded at run time)
    cfg_path = Path(args.config) if args.config else (run_dir / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sf = float(cfg["regime"]["scale_factor"])
    regime = str(cfg["regime"]["name"])
    run_id = str(cfg["regime"]["run_id"])

    # build ctx with defaults (match your run.py behavior)
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
        wg_width=float(waveguides.get("wg_width", 0.15)),
        wg_length_2=float(waveguides.get("wg_length_2", 6.5)),
        y_pos_out_orig=float(waveguides.get("y_pos_out_orig", 1.5)),
        port_offset_x=cfg.get("ports", {}).get("output_plane_x_offset", 0.8),
        x_margin=cfg.get("sim_size", {}).get("x_margin", 1.0),
        y_margin=cfg.get("sim_size", {}).get("y_margin", 2.0),
        min_steps_per_wvl=int(mesh_cfg.get("min_steps_per_wvl", 16)),
    )

    # rebuild Part2 pieces (same as run.py)
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

    task_name_final = f"{regime}_{run_id}_final"
    postprocess(
        run_dir=run_dir,
        ctx=ctx,
        make_sim=make_sim,
        task_name_final=task_name_final,
        beta_final=args.beta_final,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
