#!/usr/bin/env python3
r"""
Vector-valued architecture landscape for the multi-issuer compute-pricing paper
(§sec:arch-potential, §sec:bifurcation).  EVERYTHING here is derived from the
paper's OWN equations:

  * fitness      f_a = gamma*s_a + c_a,   c_a = alpha_a + beta^mig(psitilde_a - phi_a)   [eq:arch-fitness]
  * replicator   ds_a = s_a (f_a - fbar) dt                                          [eq:share]
  * potential    Phi(s) = (gamma/2)||s||^2 + c^T s   (eq:potential at omega=0; congestion
                 -1/2*sum omega_ab s_a s_b deforms the interior only, edges/tip invariant)

Because dF = gamma*I is symmetric, the replicator is a potential game (Sandholm):
Phi is a strictly convex quadratic, so on the simplex the |A| vertices are strict
maxima (lock-in basins), the edge points are the index-1 separating saddles, and
the full-support interior rest point is a SOURCE (a repeller), not a saddle.

Panel (a): the 3-family simplex (NVIDIA, TPU, ASIC) -- Phi level sets, the three
corner basins (integrated from eq:share), the interior source, the two NVIDIA-edge
saddles, and the most-probable escape paths.

Panel (b): the CONDITIONAL successor split P(A=a | T_tip<=T) [eq:competing-risks]. Edges
share the common maturation clock, so the marginal tip timing is the headline IG(18.5,24)
of arch_bifurcation.py (NOT an independent min of edges); the successor is a mark on that
single tip -- leapfrogs win early tips, the steady challenger wins late ones.
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import invgauss

t_wall = time.time()

# ----------------------------------------------------------------------------------
# PANEL (a) parameters -- gamma and the standalone fitnesses c_a (NVIDIA the incumbent)
# ----------------------------------------------------------------------------------
gamma = 0.5
# families in order N=NVIDIA, T=TPU, A=ASIC; NVIDIA's dominance is the gamma*s_a moat,
# not a large standalone c (challengers even carry slightly higher c_a).
c = np.array([0.10, 0.03, 0.00])            # c_N, c_T, c_A
names = ["NVIDIA", "TPU", "ASIC"]

# simplex vertices in the plane: N top, T bottom-left, A bottom-right
h = np.sqrt(3) / 2
V = np.array([[0.5, h], [0.0, 0.0], [1.0, 0.0]])     # rows: N, T, A


def bary2cart(s):                                     # s: (...,3) -> (...,2)
    return s @ V


def Phi(s):
    return 0.5 * gamma * (s ** 2).sum(-1) + s @ c


# ---- barycentric lattice over the simplex (for the 3-D surface + basin colors)
nsub = 80                                             # subdivisions per edge
pts_b = []                                            # barycentric lattice points
for i in range(nsub + 1):
    for j in range(nsub + 1 - i):
        k = nsub - i - j
        pts_b.append((i / nsub, j / nsub, k / nsub))  # (s_N, s_T, s_A)
pts_b = np.array(pts_b)
pts_xy = bary2cart(pts_b)
z_phi = Phi(pts_b)

# lattice triangulation (both orientations of each cell)
def lat_index():
    idx, out = {}, []
    row = 0
    for i in range(nsub + 1):
        for j in range(nsub + 1 - i):
            idx[(i, j)] = row; row += 1
    for i in range(nsub):
        for j in range(nsub - i):
            out.append([idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]])
            if j < nsub - i - 1:
                out.append([idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]])
    return np.array(out)
tris = lat_index()
triang = mtri.Triangulation(pts_xy[:, 0], pts_xy[:, 1], triangles=tris)

# ---- basin per lattice point: integrate eq:share to its corner
S = pts_b.copy()
dt = 0.08
for _ in range(900):
    f = gamma * S + c
    fbar = (S * f).sum(-1, keepdims=True)
    S = S + dt * S * (f - fbar)
    S = np.clip(S, 0, None)
    S = S / S.sum(-1, keepdims=True)
basin = S.argmax(-1)                                  # 0=N,1=T,2=A per point
basin_face = basin[tris].max(-1) * 0 + np.round(basin[tris].mean(-1)).astype(int)  # per-face majority-ish

# ---- special points (all closed-form from eq:potential)
cbar = c.mean()
s_src = 1.0 / 3 + (cbar - c) / gamma                  # interior equal-fitness rest point
# NVIDIA-edge saddles: on edge N-x (third family share 0), s_N* = 1/2 - (c_N - c_x)/(2 gamma)
def edge_saddle(other):                               # other=1(TPU) or 2(ASIC)
    sNe = 0.5 - (c[0] - c[other]) / (2 * gamma)
    s = np.zeros(3); s[0] = sNe; s[other] = 1 - sNe
    return s
sad_T, sad_A = edge_saddle(1), edge_saddle(2)
s_op = np.array([0.955, 0.025, 0.020])                # contestable operating point: at the vertex (two-variable synthesis)

# ----------------------------------------------------------------------------------
# PANEL (b): the CONDITIONAL successor split, consistent-by-construction with the headline.
#
# The challenger edges share the common maturation clock (non-orthogonal frontiers: one
# foundry, shared process-node and portability cadence), so the marginal tip timing is the
# common-clock first passage = the calibrated headline IG(18.5, 24) of fig:arch-bif(c), NOT
# the independent min of two edges (which would overstate it ~4x). The successor is the
# IDIOSYNCRATIC argmin, modeled as a mark on the single tip: P(A=leapfrog | tip at tau).
# Only a high-dispersion leapfrog can tip EARLY (before the clock brings the mean to the
# saddle-node), so the mark declines in tenor -- early tips are leapfrogs, late tips steady.
# ----------------------------------------------------------------------------------
MU_HEAD, LAM_HEAD = 18.5, 24.34                          # = headline tip law (arch_bifurcation.py)
Tg = np.linspace(0.05, 30, 600)
head = invgauss(mu=MU_HEAD / LAM_HEAD, scale=LAM_HEAD)
f_head, F_head = head.pdf(Tg), head.cdf(Tg)

THALF, WIDTH = 10.0, 4.0                                # mark crossover tenor / width [ILLUS]
def p_leapfrog(tau):                                    # P(A=ASIC leapfrog | tip at tau), declining
    return 1.0 / (1.0 + np.exp((tau - THALF) / WIDTH))
pL = p_leapfrog(Tg)

# conditional shares P(A=a | tip<=T) = int_0^T p_a(u) f(u) du / F(T)  -- marginal stays = headline
num_A = np.concatenate([[0.0], np.cumsum(0.5 * (pL[1:] * f_head[1:] + pL[:-1] * f_head[:-1]) * np.diff(Tg))])
i0 = int(np.argmax(F_head > 1e-4))
shareA = num_A / np.clip(F_head, 1e-12, None)           # ASIC (leapfrog) conditional share
shareA[:i0] = shareA[i0]
share = {"ASIC": shareA, "TPU": 1.0 - shareA}

# ==================================================================================
# FIGURE
# ==================================================================================
fig = plt.figure(figsize=(11.8, 4.6))
a0 = fig.add_subplot(1, 2, 1, projection="3d")
a0.computed_zorder = False        # painter's order: floor -> surface -> paths/markers on top
a1 = fig.add_subplot(1, 2, 2)

# ---- (a) ternary landscape as a 3-D surface --------------------------------------
pastel = np.array([[0.66, 0.86, 0.72],                # N  green
                   [0.70, 0.81, 0.94],                # T  blue
                   [0.99, 0.84, 0.66]])               # A  orange
z0 = z_phi.min() - 0.075                              # floor level for projections

# manual Lambert shading so the relief reads (matplotlib drops shade with facecolors)
p3 = np.stack([pts_xy[:, 0], pts_xy[:, 1], z_phi], -1)[tris]      # (F,3,3)
nrm = np.cross(p3[:, 1] - p3[:, 0], p3[:, 2] - p3[:, 0])
nrm[nrm[:, 2] < 0] *= -1
nrm = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
light = np.array([-0.35, -0.45, 0.82]); light = light / np.linalg.norm(light)
lam = np.clip(nrm @ light, 0, 1)
shade_f = (0.60 + 0.40 * lam)[:, None]
face_rgb = np.clip(pastel[basin_face] * shade_f, 0, 1)

# floor first: Phi level sets + triangle outline (the old 2-D view, projected)
a0.tricontour(triang, z_phi, levels=12, zdir="z", offset=z0,
              colors="0.55", linewidths=0.5, zorder=1)
tri_edge = np.array([V[0], V[1], V[2], V[0]])
a0.plot(tri_edge[:, 0], tri_edge[:, 1], z0, color="0.25", lw=1.1, zorder=1)

# the surface
surf = a0.plot_trisurf(triang, z_phi, shade=False, antialiased=False,
                       edgecolor="none", linewidth=0.0, zorder=2)
surf.set_facecolors(face_rgb)

# surface boundary (the three simplex edges, at height Phi)
for k0, k1 in ((0, 1), (1, 2), (2, 0)):
    tt = np.linspace(0, 1, 120)[:, None]
    sb = (1 - tt) * np.eye(3)[k0] + tt * np.eye(3)[k1]
    pb = bary2cart(sb)
    a0.plot(pb[:, 0], pb[:, 1], Phi(sb), color="0.25", lw=1.0, zorder=3)

# corner peaks + labels
peak_lab = {0: (0.06, 0.02, 0.035, "left"), 1: (-0.06, -0.02, 0.030, "right"),
            2: (0.12, -0.05, 0.000, "left")}
for k, nm in enumerate(names):
    zk = Phi(np.eye(3)[k])
    a0.scatter(*V[k], zk + 0.004, color="k", edgecolors="white", linewidths=0.6,
               s=30, depthshade=False, zorder=9)
    dx, dy, dz, ha = peak_lab[k]
    a0.text(V[k][0] + dx, V[k][1] + dy, zk + dz, nm, ha=ha, fontsize=9, fontweight="bold")

# interior source: a white disc lying IN the bowl (tangent to the surface, reads 3-D)
th_d = np.linspace(0, 2 * np.pi, 72)
u0 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)          # sum-zero basis of the simplex plane
w0 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
r_src = 0.055
ring_b = s_src[None, :] + r_src * (np.cos(th_d)[:, None] * u0 + np.sin(th_d)[:, None] * w0)
ring_xy = bary2cart(ring_b)
ring_z = Phi(ring_b) + 0.004
disc = Poly3DCollection([np.column_stack([ring_xy, ring_z])],
                        facecolor="white", edgecolor="k", linewidth=1.2, zorder=8)
a0.add_collection3d(disc)
psrc = bary2cart(s_src)
a0.text(psrc[0] + 0.04, psrc[1] - 0.22, Phi(s_src) - 0.012, "source", ha="center", fontsize=7.5)
for s in (sad_T, sad_A):
    p = bary2cart(s)
    a0.scatter(*p, Phi(s) + 0.006, marker="o", color="0.15", s=26, depthshade=False, zorder=8)
pT = bary2cart(sad_T)
a0.text(pT[0] - 0.05, pT[1] + 0.03, Phi(sad_T) + 0.045, "saddles", fontsize=7.5, color="0.25")
# escape paths: barycentric lines op -> saddle, lifted above the surface and pulled
# slightly inward so they clear the boundary rim; cone arrowheads (view-scaled 3-D mesh)
rng_vis = np.array([1.04, h + 0.10, (z_phi.max() + 0.05) - z0])   # axis spans
asp_vis = np.array([1.0, 0.92, 0.62])                             # box aspect
def to_vis(p):  return p / rng_vis * asp_vis
def to_dat(v):  return v * rng_vis / asp_vis

def cone_head(ax, apex, direction, L=0.066, R=0.024, color="tab:red", nseg=24):
    """3-D cone with apex at `apex`, axis along `direction` (data coords);
    built in view-scaled space so it renders round despite axis anisotropy."""
    a_v = to_vis(np.asarray(apex, float))
    d_v = to_vis(np.asarray(apex, float)) - to_vis(np.asarray(apex, float) - np.asarray(direction, float))
    d_v = d_v / np.linalg.norm(d_v)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(d_v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d_v, tmp); u /= np.linalg.norm(u)
    w = np.cross(d_v, u)
    base = a_v - L * d_v
    th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ring_v = base[None, :] + R * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)
    ring = to_dat(ring_v)
    apex_d = np.asarray(apex, float)
    faces = [[apex_d, ring[i], ring[(i + 1) % nseg]] for i in range(nseg)]
    faces.append([ring[i] for i in range(nseg)])                  # base disk
    pc = Poly3DCollection(faces, facecolor=color, edgecolor=color, linewidth=0.15, zorder=9)
    ax.add_collection3d(pc)

lift, pull = 0.022, 0.040
cen = np.array([1.0, 1.0, 1.0]) / 3
for s_end in (sad_T, sad_A):
    tt = np.linspace(0, 1, 90)[:, None]
    taper = np.sin(np.pi * tt) ** 0.7                             # 0 at both ends, max mid-path
    sp = (1 - tt) * s_op + tt * s_end
    sp = (1 - pull * taper) * sp + (pull * taper) * cen           # inward off the rim, anchored ends
    pp = bary2cart(sp)
    zz = Phi(sp) + 0.007 + (lift - 0.007) * taper[:, 0]           # lifted mid-path, hugs both ends
    a0.plot(pp[:-8, 0], pp[:-8, 1], zz[:-8], color="tab:red", lw=2.2, alpha=0.95, zorder=8)
    apex = np.array([pp[-1, 0], pp[-1, 1], zz[-1]])
    dirn = np.array([pp[-1, 0] - pp[-16, 0], pp[-1, 1] - pp[-16, 1], zz[-1] - zz[-16]])
    cone_head(a0, apex, dirn)

pop = bary2cart(s_op)
a0.scatter(*pop, Phi(s_op) + 0.008, marker="D", color="tab:blue", s=60, depthshade=False, zorder=9)
a0.text(pop[0] - 0.30, pop[1] + 0.03, Phi(s_op) + 0.055,
        "now ($s^{c}\\!\\approx\\!1$)", ha="center", fontsize=7.5, color="tab:blue")

a0.view_init(elev=28, azim=-26)
a0.set_box_aspect((1.0, 0.92, 0.62))
a0.set_axis_off()
a0.set_xlim(-0.02, 1.02); a0.set_ylim(-0.05, h + 0.05); a0.set_zlim(z0, z_phi.max() + 0.05)
a0.set_title("(a) architecture landscape: the potential $\\Phi$ as a surface, and edge escapes",
             fontsize=10, y=0.96)

# ---- (b) conditional successor split (marginal = headline, by construction) ------
a1.stackplot(Tg, share["TPU"], share["ASIC"],
             colors=["tab:blue", "tab:red"], alpha=0.9,
             labels=["TPU (structural catch-up)", "ASIC (leapfrog)"])
a1.axvline(3, color="0.15", ls=":", lw=1.1)
a1.text(3.4, 0.09, "traded $\\leq$3yr", fontsize=7, color="0.15")
a1.set_xlim(Tg[0], 30); a1.set_ylim(0, 1)
a1.set_xlabel("tenor $T$ (years)", fontsize=9)
a1.set_ylabel("successor share  $\\mathbb{P}(A=a\\mid T_{\\mathrm{tip}}\\leq T)$", fontsize=9)
a1.set_title("(b) which successor, given a tip by $T$", fontsize=10)
a1.legend(fontsize=7.5, loc="lower right", framealpha=0.92)

plt.tight_layout()
plt.savefig("fig_arch_simplex.pdf")
plt.close()

# ---- console summary (sanity) ----------------------------------------------------
print("interior source (bary):", np.round(s_src, 3), " inside:", bool((s_src >= 0).all()))
print("edge saddle N-T  s_N* =", round(0.5 - (c[0] - c[1]) / (2 * gamma), 3),
      "| N-A  s_N* =", round(0.5 - (c[0] - c[2]) / (2 * gamma), 3))
frac = [np.mean(basin == k) for k in range(3)]
print("basin area share  N/T/A:", np.round(frac, 3))
print(f"panel(b): marginal = headline IG({MU_HEAD},{LAM_HEAD}); "
      f"P(tip) 3/5/10y = {F_head[np.argmin(abs(Tg-3))]:.3f}/"
      f"{F_head[np.argmin(abs(Tg-5))]:.3f}/{F_head[np.argmin(abs(Tg-10))]:.3f} (matches fig:arch-bif c)")
for T in (3, 5, 10, 20):
    iT = np.argmin(abs(Tg - T))
    print(f"  {T:>2}y: ASIC(leapfrog) {share['ASIC'][iT]*100:3.0f}%  TPU(steady) {share['TPU'][iT]*100:3.0f}%")
print(f"figure: fig_arch_simplex.pdf   [wall {time.time()-t_wall:.2f}s]")
