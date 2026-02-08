import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import tidy3d as td
import tidy3d.web as web


def _flux_to_db(flux: np.ndarray) -> np.ndarray:
    # user confirmed flux won't be 0/negative
    return 10.0 * np.log10(flux)


def _interp_flux(flux_data, freqs_target: np.ndarray) -> np.ndarray:
    """
    Keep your notebook convention:
      freqs: flux_data.flux.f
      flux : abs(flux_data.flux.values)
    """
    freqs_m = np.asarray(flux_data.flux.f)
    flux_m = np.asarray(np.abs(flux_data.flux.values))
    return np.interp(freqs_target, freqs_m, flux_m)


def _save_density_png(params_1d, nx: int, ny: int, out_png: Path):
    arr = np.asarray(params_1d).reshape(nx, ny)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(arr.T, origin="lower", aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _export_flux_csv(flux_data, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    freqs = np.asarray(flux_data.flux.f)
    flux = np.asarray(np.abs(flux_data.flux.values))
    wl_nm = (td.C_0 / freqs) * 1e9
    flux_db = _flux_to_db(flux)

    df = pd.DataFrame(
        {"wavelength_nm": wl_nm, "flux": flux, "flux_db": flux_db}
    )
    df.to_csv(out_csv, index=False)


def _compute_il_xt_db(sim_data, freqs_object):
    f1 = np.asarray(freqs_object["terminal_1"])
    f2 = np.asarray(freqs_object["terminal_2"])

    flux1 = sim_data["flux_1"]
    flux2 = sim_data["flux_2"]

    # terminal_1 band: desired flux_1, xt flux_2
    d1 = _interp_flux(flux1, f1)
    x1 = _interp_flux(flux2, f1)
    il1 = -float(np.mean(_flux_to_db(d1)))
    xt1 = float(np.mean(_flux_to_db(x1)))

    # terminal_2 band: desired flux_2, xt flux_1
    d2 = _interp_flux(flux2, f2)
    x2 = _interp_flux(flux1, f2)
    il2 = -float(np.mean(_flux_to_db(d2)))
    xt2 = float(np.mean(_flux_to_db(x2)))

    return {
        "il1_db": il1,
        "il2_db": il2,
        "xt1_db": xt1,
        "xt2_db": xt2,
        "il_avg_db": 0.5 * (il1 + il2),
        "xt_avg_db": 0.5 * (xt1 + xt2),
        "n1": int(len(f1)),
        "n2": int(len(f2)),
    }


def _plot_transmission_db(sim_data, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    f = np.asarray(sim_data["flux_1"].flux.f)
    wl = (td.C_0 / f) * 1e9

    y1 = _flux_to_db(np.asarray(np.abs(sim_data["flux_1"].flux.values)))
    y2 = _flux_to_db(np.asarray(np.abs(sim_data["flux_2"].flux.values)))

    plt.figure()
    plt.plot(wl, y1, label="flux_1 (dB)")
    plt.plot(wl, y2, label="flux_2 (dB)")
    plt.gca().invert_xaxis()  # optional: wavelength decreasing to the right like many papers
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def postprocess(
    *,
    run_dir,
    ctx,
    make_sim,                 # callable: sim = make_sim(params, beta)
    task_name_final: str,
    beta_final: float = 1.0,
    verbose: bool = False,
):
    """
    Part4 postprocess (A-spec):
      - final/density.png
      - final/eps.png
      - final/flux_terminal_1.csv, final/flux_terminal_2.csv
      - final/transmission_db.png
      - final/metrics.csv, final/metrics.txt

    Notes:
      - Uses web.run (NOT td.web.run).
      - Output folder: run_dir/final/*
      - Final sim cache folder: run_dir/final/sim_data
    """
    run_dir = Path(run_dir)
    final_dir = run_dir / "final"
    final_sim_dir = final_dir / "sim_data"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_sim_dir.mkdir(parents=True, exist_ok=True)

    # load final params
    params_path = run_dir / "final_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"final_params.pkl not found: {params_path}")
    with open(params_path, "rb") as f:
        params_final = pickle.load(f)
    params_final = np.asarray(params_final).reshape(-1)

    # density
    _save_density_png(params_final, ctx["nx"], ctx["ny"], final_dir / "density.png")

    # run final sim
    sim_final = make_sim(params_final, beta_final)
    sim_data = web.run(
        sim_final,
        task_name=task_name_final,
        folder_name=str(final_sim_dir),
        verbose=verbose,
    )

    # eps figure
    ax = sim_data.simulation.plot_eps(z=0, monitor_alpha=0.0)
    eps_png = final_dir / "eps.png"
    ax.figure.savefig(eps_png, dpi=200, bbox_inches="tight")
    plt.close(ax.figure)

    # export flux csv
    _export_flux_csv(sim_data["flux_1"], final_dir / "flux_terminal_1.csv")
    _export_flux_csv(sim_data["flux_2"], final_dir / "flux_terminal_2.csv")

    # transmission plot
    _plot_transmission_db(sim_data, final_dir / "transmission_db.png")

    # metrics (IL/XT)
    metrics = _compute_il_xt_db(sim_data, ctx["freqs_object"])

    df = pd.DataFrame([metrics])
    df.to_csv(final_dir / "metrics.csv", index=False)

    with open(final_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("[OK] postprocess done:", final_dir)
    print("[METRIC] IL avg (dB):", metrics["il_avg_db"])
    print("[METRIC] XT avg (dB):", metrics["xt_avg_db"])

    return metrics
