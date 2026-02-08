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

from src.tidy3d.optimizer import make_objective


def main():
    sf = 0.70
    ctx = build_part1_context(scale_factor=sf)

    # init params
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

    objective = make_objective(make_sim=make_sim, task_name="optimizer_obj_test", folder_name="tmp_optimizer")

    J, sim_data = objective(params0, beta=1, verbose=False)
    print("monitor names:", list(sim_data.monitor_data.keys()))


if __name__ == "__main__":
    main()
