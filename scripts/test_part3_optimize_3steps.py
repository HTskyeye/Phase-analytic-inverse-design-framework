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


def main():
    sf = 0.70
    ctx = build_part1_context(scale_factor=sf)

    # init params: all 0.5
    params0 = np.full((ctx["nx"] * ctx["ny"],), 0.5)

    # Part2 build
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
    flux_mnts, _ = build_flux_monitors(
        terminal_1=terminal_1, terminal_2=terminal_2,
        freq_min=freq_min, freq_max=freq_max, fwidth=fwidth, Nf=101
    )

    def make_sim(p, beta):
        return make_sim_full(
            params=p, beta=beta, ctx=ctx,
            mode_src=mode_src,
            terminal_1=terminal_1, terminal_2=terminal_2,
            flux_mnts=flux_mnts
        )

    params_final, history = optimize_notebook_style(
        params_init=params0,
        make_sim=make_sim,
        task_name="optimizer_opt_test",
        folder_name="tmp_optimizer_opt",
        num_steps=3,          # <<< only 3 steps sanity check
        ckpt_every=1,
        lr_init=5e-2,
        decay_steps=5,
        decay_rate=0.8,
        run_dir=None,         # no checkpoint files for sanity test
    )

    print("DONE. J history:", history["J"])
    print("params range:", float(params_final.min()), float(params_final.max()))


if __name__ == "__main__":
    main()
