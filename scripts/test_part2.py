import numpy as np
import matplotlib.pylab as plt
import tidy3d as td

from src.tidy3d.simulation import (
    build_part1_context,
    make_sim_base,
    build_mode_source,
    make_terminal_monitor,
    build_flux_monitors,
    make_sim_full,
)

def main():
    ctx = build_part1_context(scale_factor=0.70)

    # reference params: random (只用于生成 sim_ref 给 mode solver；不影响结构自适应逻辑)
    params_ref = np.random.random((ctx["nx"] * ctx["ny"],))
    sim_ref = make_sim_base(params_ref, ctx=ctx, beta=1)

    mode_src, freq_mid, fwidth = build_mode_source(ctx=ctx, sim_ref=sim_ref, num_modes=1, mode_index=0)

    # terminals
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

    # flux monitors
    freq_max = td.C_0 / ctx["wavelength_min"]
    freq_min = td.C_0 / ctx["wavelength_max"]
    flux_mnts, freqs_flux = build_flux_monitors(
        terminal_1=terminal_1, terminal_2=terminal_2,
        freq_min=freq_min, freq_max=freq_max, fwidth=fwidth, Nf=501
    )

    # full sim with random params
    params = np.random.random((ctx["nx"] * ctx["ny"],))
    sim = make_sim_full(
        params=params, beta=1, ctx=ctx,
        mode_src=mode_src,
        terminal_1=terminal_1, terminal_2=terminal_2,
        flux_mnts=flux_mnts
    )

    ax = sim.plot_eps(z=0.1)
    plt.show()

if __name__ == "__main__":
    main()
