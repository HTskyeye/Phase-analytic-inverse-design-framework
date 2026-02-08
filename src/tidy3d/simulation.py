# ============================================================
# Part 1 — Build base simulation (geometry + materials + grid)
# ============================================================

import numpy as np
import autograd.numpy as anp
import tidy3d as td


def build_part1_context(
    *,
    # Material
    n_si: float = 3.1,
    eps_min: float = 4.6,
    # Regime control
    scale_factor: float = 0.70,
    # Spectral
    wavelength_max: float = 1.470,
    wavelength_min: float = 1.240,
    delta_wl: float = 0.005,
    # Design region base
    pixel_size: float = 0.08,
    nx_orig: int = 45,
    ny_orig: int = 60,
    # Waveguides
    wg_width: float = 0.15,
    wg_length_2: float = 6.5,
    y_pos_out_orig: float = 1.5,
    # Sim size rule
    port_offset_x: float = 0.8,
    x_margin: float = 1.0,
    y_margin: float = 2.0,
    # Mesh
    min_steps_per_wvl: int = 16,
):
    """
    Build the Part1 context (all derived constants + geometries).
    This function is deterministic given inputs.
    """
    eps_max = n_si**2

    # Frequency grid
    node_num = int(round(abs(wavelength_min - wavelength_max) / delta_wl)) + 1
    Wls = np.linspace(wavelength_min, wavelength_max, node_num)
    Freqs = td.C_0 / Wls

    freq_max = td.C_0 / wavelength_min
    freq_min = td.C_0 / wavelength_max
    fwidth = abs(freq_max - freq_min)
    run_time = 220 / fwidth

    # Channel definitions (kept exactly)
    freqs_object = dict(terminal_1=Freqs[0:21], terminal_2=Freqs[26:47])
    freqs_observe = dict(ck_1=Freqs[10], ck_2=Freqs[36])

    # Design region size (scaled)
    nx = int(nx_orig * scale_factor)
    ny = int(ny_orig * scale_factor)
    design_x = nx * pixel_size
    design_y = ny * pixel_size

    # Design region geometry
    design_region_geo = td.Box(
        size=(design_x, design_y, td.inf),
        center=(design_x / 2, 0, 0),
    )

    # Mesh refinement region
    refine_box = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(design_x / 2, 0, 0),
            size=(design_x + 0.4, design_y + 0.4, td.inf),
        ),
        dl=[0.01, 0.01, None],
    )

    # Waveguide geometry (auto-follow design_x)
    y_pos_out = y_pos_out_orig * scale_factor

    x_in = -wg_length_2 / 2
    box_wg_in = td.Box(center=(x_in, 0, 0), size=(wg_length_2, wg_width, td.inf))

    x_out = design_x + wg_length_2 / 2
    box_wg_1 = td.Box(center=(x_out,  y_pos_out, 0), size=(wg_length_2, wg_width, td.inf))
    box_wg_2 = td.Box(center=(x_out, -y_pos_out, 0), size=(wg_length_2, wg_width, td.inf))

    # Background structure (permittivity=10), kept
    wg_union_geo = box_wg_in + box_wg_1 + box_wg_2
    sector_structure = td.Structure(geometry=wg_union_geo, medium=td.Medium(permittivity=10))
    Structures = [sector_structure]

    # Simulation domain size (strictly follow your rule)
    Lx = design_x + 2 * port_offset_x + x_margin
    Ly = design_y + y_margin
    Lz = 0.0

    # Field monitor (kept)
    fld_mnt = td.FieldMonitor(
        center=(0, 0, 0),
        size=(td.inf, td.inf, 0),
        freqs=(freqs_observe["ck_1"], freqs_observe["ck_2"]),
        name="field",
    )

    return dict(
        # raw
        n_si=n_si,
        eps_min=eps_min,
        eps_max=eps_max,
        scale_factor=scale_factor,
        wavelength_max=wavelength_max,
        wavelength_min=wavelength_min,
        delta_wl=delta_wl,
        pixel_size=pixel_size,
        nx_orig=nx_orig,
        ny_orig=ny_orig,
        wg_width=wg_width,
        wg_length_2=wg_length_2,
        y_pos_out_orig=y_pos_out_orig,
        port_offset_x=port_offset_x,
        x_margin=x_margin,
        y_margin=y_margin,
        min_steps_per_wvl=min_steps_per_wvl,
        # derived
        nx=nx,
        ny=ny,
        design_x=design_x,
        design_y=design_y,
        run_time=run_time,
        freqs_object=freqs_object,
        freqs_observe=freqs_observe,
        # geometries
        design_region_geo=design_region_geo,
        refine_box=refine_box,
        Structures=Structures,
        fld_mnt=fld_mnt,
        # sim size
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )


def make_eps(params, *, eps_min: float, eps_max: float, beta=1):
    """
    Map params in [0,1] to permittivity in [eps_min, eps_max].
    beta kept for API compatibility (used later).
    """
    return eps_min + (eps_max - eps_min) * params


def make_custom_medium_structure(params, *, ctx, beta=1):
    """
    Build a td.Structure using CustomMedium + ScalarFieldDataArray.
    This matches your notebook implementation.
    """
    nx, ny = ctx["nx"], ctx["ny"]
    design_x, design_y = ctx["design_x"], ctx["design_y"]
    pixel_size = ctx["pixel_size"]
    eps_max = ctx["eps_max"]

    eps = make_eps(params, eps_min=ctx["eps_min"], eps_max=eps_max, beta=beta).reshape((nx, ny, 1))
    eps = anp.where(eps < 1, 1, eps)
    eps = anp.where(eps > eps_max, eps_max, eps)

    # pixel center coords (auto-update)
    center_x = design_x / 2
    span_x = (nx - 1) * pixel_size
    xs_start = center_x - span_x / 2
    xs_end = center_x + span_x / 2

    center_y = 0.0
    span_y = (ny - 1) * pixel_size
    ys_start = center_y - span_y / 2
    ys_end = center_y + span_y / 2

    xs = list(anp.linspace(xs_start, xs_end, nx))
    ys = list(anp.linspace(ys_start, ys_end, ny))
    zs = [0]

    coords = dict(x=xs, y=ys, z=zs)
    eps_arr = td.ScalarFieldDataArray(data=eps, coords=coords)
    medium = td.CustomMedium(permittivity=eps_arr)

    return td.Structure(geometry=ctx["design_region_geo"], medium=medium)


def make_sim_base(params, *, ctx, beta=1):
    """
    Base simulation: geometry + materials + grid + field monitor.
    No sources / no terminal mode solver / no flux monitors here.
    """
    design_struct = make_custom_medium_structure(params, ctx=ctx, beta=beta)
    all_structures = ctx["Structures"] + [design_struct]

    sim = td.Simulation(
        size=(ctx["Lx"], ctx["Ly"], ctx["Lz"]),
        center=(ctx["design_x"] / 2, 0, 0),
        grid_spec=td.GridSpec.auto(
            wavelength=(ctx["wavelength_max"] + ctx["wavelength_min"]) / 2,  # REQUIRED without sources
            min_steps_per_wvl=ctx["min_steps_per_wvl"],
            override_structures=[ctx["refine_box"]],
        ),
        structures=all_structures,
        run_time=ctx["run_time"],
        medium=td.Medium(permittivity=1.0),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True),
        monitors=[ctx["fld_mnt"]],
    )
    return sim

from tidy3d.plugins.mode import ModeSolver


def build_mode_source(*, ctx, sim_ref, num_modes: int = 1, mode_index: int = 0):
    """
    Build the input ModeSource from a reference simulation.
    This matches your Part2 logic:
      - plane_left at x = -0.8
      - GaussianPulse(freq0=freq_mid, fwidth=fwidth)
      - direction="+"
    """
    # keep same definition as your notebook
    wavelength_max = ctx["wavelength_max"]
    wavelength_min = ctx["wavelength_min"]

    freq_max = td.C_0 / wavelength_min
    freq_min = td.C_0 / wavelength_max
    fwidth = abs(freq_max - freq_min)
    freq_mid = 0.5 * (freq_min + freq_max)

    # infer output spacing from Part1 definition: y_pos_out = y_pos_out_orig * scale_factor
    y_pos_out_scaled = ctx["y_pos_out_orig"] * ctx["scale_factor"]
    wg_spacing = 2.0 * y_pos_out_scaled  # distance between output waveguide centers (scaled)

    # mode plane size (same structure as your code)
    # old: mode_size_horizontal = (0, 1.8 * wg_spacing + wg_width, max([Lz, lz, 3]))
    # new: z-span: keep the old fallback ">=3"
    mode_size_horizontal = (0, 1.8 * wg_spacing + ctx["wg_width"], 3)

    plane_left = td.Box(
        center=(-0.8, 0, 0),
        size=mode_size_horizontal,
    )

    mode_solver_left = ModeSolver(
        simulation=sim_ref,
        plane=plane_left,
        freqs=[freq_mid],
        mode_spec=td.ModeSpec(num_modes=num_modes),
    )

    mode_src = mode_solver_left.to_source(
        source_time=td.GaussianPulse(freq0=freq_mid, fwidth=fwidth),
        direction="+",
        mode_index=mode_index,
    )

    return mode_src, freq_mid, fwidth


def make_terminal_monitor(*, ctx, sim_ref, direction: str, freqs, name: str, num_modes: int = 1, freq_mid=None):
    """
    Make terminal ModeMonitor (terminal_1 / terminal_2) using ModeSolver,
    then updated_copy(center=...) exactly like your Part2.
    """
    # y position scaled exactly like your Part2
    y_pos_out_scaled = ctx["y_pos_out_orig"] * ctx["scale_factor"]

    # terminal centers (exactly your Part2 logic)
    center = [0.0, 0.0, 0.0]
    if name == "terminal_1":
        center[0] = ctx["design_x"] + 0.8
        center[1] = y_pos_out_scaled
    elif name == "terminal_2":
        center[0] = ctx["design_x"] + 0.8
        center[1] = -y_pos_out_scaled
    else:
        raise ValueError(f"Unknown terminal name: {name}")
    center[2] = 0.0

    # Build a "right" plane that is compatible with your terminal center update
    # (plane provides size/orientation; final position is set by updated_copy)
    y_pos_out_scaled = ctx["y_pos_out_orig"] * ctx["scale_factor"]
    wg_spacing = 2.0 * y_pos_out_scaled
    mode_size_horizontal = (0, 1.8 * wg_spacing + ctx["wg_width"], 3)

    if direction != "right":
        # Your current pipeline only uses "right". Keep guard to avoid silent behavior changes.
        raise ValueError(f"Only direction='right' is supported in the refactor; got {direction}")

    plane_right = td.Box(
        center=(ctx["design_x"] + 0.8, 0, 0),
        size=mode_size_horizontal,
    )

    if freq_mid is None:
        # compute it the same way as build_mode_source
        freq_max = td.C_0 / ctx["wavelength_min"]
        freq_min = td.C_0 / ctx["wavelength_max"]
        freq_mid = 0.5 * (freq_min + freq_max)

    mode_solver = ModeSolver(
        simulation=sim_ref,
        plane=plane_right,
        freqs=[freq_mid],
        mode_spec=td.ModeSpec(num_modes=num_modes),
    )

    terminal = mode_solver.to_monitor(freqs=freqs, name=name)
    terminal = terminal.updated_copy(center=center)
    return terminal


def build_flux_monitors(*, terminal_1, terminal_2, freq_min, freq_max, fwidth, Nf: int = 501):
    """
    Build FluxMonitor for each terminal, same as your Part2.
    """
    freqs_flux = np.linspace(freq_min - fwidth / 10, freq_max + fwidth / 10, Nf)

    flux_mnt_tl1 = td.FluxMonitor(
        center=terminal_1.center,
        size=terminal_1.size,
        name="flux_1",
        freqs=list(freqs_flux),
    )

    flux_mnt_tl2 = td.FluxMonitor(
        center=terminal_2.center,
        size=terminal_2.size,
        name="flux_2",
        freqs=list(freqs_flux),
    )

    return [flux_mnt_tl1, flux_mnt_tl2], freqs_flux


def make_sim_full(*, params, beta, ctx, mode_src, terminal_1, terminal_2, flux_mnts):
    """
    Full simulation = Part1 base sim + Part2 source + Part2 monitors.
    This mirrors your Part2 make_sim().
    """
    sim_base = make_sim_base(params, ctx=ctx, beta=beta)

    output_monitors = [terminal_1, terminal_2]
    monitors_all = tuple(list(sim_base.monitors) + list(flux_mnts) + output_monitors)

    return sim_base.updated_copy(
        sources=[mode_src],
        monitors=monitors_all,
    )
