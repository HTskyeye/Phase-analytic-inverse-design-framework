import numpy as np
import matplotlib.pylab as plt
from src.tidy3d.simulation import build_part1_context, make_sim_base

ctx = build_part1_context(scale_factor=0.70)
params = np.random.random((ctx["nx"] * ctx["ny"],))

sim = make_sim_base(params, ctx=ctx, beta=1)
ax = sim.plot_eps(z=0, monitor_alpha=0.0)
plt.show()
