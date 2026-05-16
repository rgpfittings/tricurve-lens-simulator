"""
Streamlit prototype conversion of EihabGPTricurveSim25.

Run:
    pip install streamlit numpy scipy matplotlib
    streamlit run tricurve_streamlit_prototype.py

This first version ports the core numerical engine and plots:
    - Tangential topography map
    - Fluorescein / clearance simulation
    - Tear profile along selected axis
    - Cornea/lens cross-section along selected axis

Not yet ported:
    - Mouse hover tooltips
    - Any file export or reporting
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator


@dataclass
class TopographyInputs:
    r_flat: float = 8.13
    r_steep: float = 7.93
    e_flat: float = 0.70
    e_steep: float = 0.24
    flat_axis_deg: float = 11.0
    spec_sphere_d: float = 0.0
    spec_cyl_d: float = 0.0
    spec_axis_deg: float = 180.0
    lens_bvp_d: float = 0.0


@dataclass
class LensParams:
    boz_r: float = 8.00
    boz_d: float = 5.32
    pc1_r: float = 8.32
    pc1_d: float = 6.90
    pc2_r: float = 9.475
    pc2_d: float = 9.00
    is_toric: bool = False
    toricity_mm: float = 0.50


@dataclass
class AppSettings:
    map_diameter_mm: float = 12.0
    grid_step_mm: float = 0.05
    n_k: float = 1.3375
    eccentricity_scale: float = 0.50
    max_penetration_um: float = 0.0
    hidden_edge_width_mm: float = 0.12
    peripheral_blend_width_mm: float = 0.55
    min_pc1_pc2_blend_mm: float = 0.05
    min_peripheral_blend_mm: float = 0.15
    edge_curve_radius_mm: float = 1000.0
    boz_pc1_blend_start_fraction: float = 0.70
    cross_section_separation_scale: float = 3.0


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def circle_sag(r: np.ndarray | float, r_c: float, z_c: float, radius: float):
    r = np.asarray(r)
    arg = radius**2 - (r - r_c) ** 2
    if np.any(arg < -1e-12):
        raise ValueError("Invalid circle sag geometry.")
    arg = np.maximum(arg, 0)
    return z_c - np.sqrt(arg)


def circle_slope(r: float, r_c: float, radius: float):
    arg = radius**2 - (r - r_c) ** 2
    if arg <= 0:
        raise ValueError("Invalid circle slope geometry.")
    return (r - r_c) / np.sqrt(arg)


def circle_center_from_point_slope_radius(r0: float, z0: float, m0: float, radius: float):
    n = np.array([-m0, 1.0], dtype=float)
    n = n / np.linalg.norm(n)
    c_plus = np.array([r0, z0]) + radius * n
    c_minus = np.array([r0, z0]) - radius * n
    if c_plus[1] > z0:
        return float(c_plus[0]), float(c_plus[1])
    return float(c_minus[0]), float(c_minus[1])


def circle_second_derivative_from_slope_radius(slope: float, radius: float):
    return (1 + slope**2) ** 1.5 / radius


def quintic_c2_patch(r, r_a, r_b, z_a, m_a, zpp_a, z_b, m_b, zpp_b):
    r = np.asarray(r)
    h = r_b - r_a
    if h <= 0:
        raise ValueError("quintic_c2_patch requires r_b > r_a.")

    c0 = z_a
    c1 = m_a
    c2 = zpp_a / 2

    rhs1 = z_b - (c0 + c1 * h + c2 * h**2)
    rhs2 = m_b - (c1 + 2 * c2 * h)
    rhs3 = zpp_b - (2 * c2)

    a = np.array(
        [
            [h**3, h**4, h**5],
            [3 * h**2, 4 * h**3, 5 * h**4],
            [6 * h, 12 * h**2, 20 * h**3],
        ],
        dtype=float,
    )
    c3, c4, c5 = np.linalg.solve(a, np.array([rhs1, rhs2, rhs3], dtype=float))

    x = r - r_a
    return c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5


# -----------------------------------------------------------------------------
# Cornea and lens model
# -----------------------------------------------------------------------------

def build_corneal_sag_grid(topo: TopographyInputs, settings: AppSettings):
    radius = settings.map_diameter_mm / 2
    x = np.arange(-radius, radius + settings.grid_step_mm / 2, settings.grid_step_mm)
    y = np.arange(-radius, radius + settings.grid_step_mm / 2, settings.grid_step_mm)
    xq, yq = np.meshgrid(x, y)
    rho = np.hypot(xq, yq)

    theta = np.arctan2(yq, xq) - np.deg2rad(topo.flat_axis_deg)

    e_flat_eff = settings.eccentricity_scale * topo.e_flat
    e_steep_eff = settings.eccentricity_scale * topo.e_steep
    k_flat = -(e_flat_eff**2)
    k_steep = -(e_steep_eff**2)

    c2 = np.cos(theta) ** 2
    s2 = np.sin(theta) ** 2

    r_dir = 1 / (c2 / topo.r_flat + s2 / topo.r_steep)
    k_dir = c2 * k_flat + s2 * k_steep

    c = 1 / r_dir
    inside = 1 - (1 + k_dir) * (c**2) * (rho**2)
    inside = np.where(inside < 0, np.nan, inside)

    z = (c * rho**2) / (1 + np.sqrt(inside))
    z = np.where(rho > radius, np.nan, z)
    return xq, yq, z


def build_tricurve_lens_profile(lens: LensParams, settings: AppSettings, steep: bool = False):
    dr = 0.01

    bcr = lens.boz_r
    pc1r = lens.pc1_r
    pc2r = lens.pc2_r
    if steep and lens.is_toric:
        bcr -= lens.toricity_mm
        pc1r -= lens.toricity_mm
        pc2r -= lens.toricity_mm

    if min(bcr, pc1r, pc2r) <= 0:
        raise ValueError("Toricity too large: one or more steep radii become <= 0.")

    r0 = lens.boz_d / 2
    r1 = r0 + (lens.pc1_d - lens.boz_d) / 2
    r5 = r1 + (lens.pc2_d - lens.pc1_d) / 2

    edge_w = settings.hidden_edge_width_mm
    pc1_pc2_blend = settings.min_pc1_pc2_blend_mm
    min_pure_pc2 = 0.10
    outer_blend_req = settings.peripheral_blend_width_mm
    min_outer_blend = settings.min_peripheral_blend_mm

    available_outer = r5 - r1
    outer_blend_max = available_outer - edge_w - pc1_pc2_blend - min_pure_pc2
    if outer_blend_max <= 0:
        raise ValueError("Not enough room for a valid outer geometry. Increase TD or reduce edge/blend widths.")

    outer_blend = min(outer_blend_req, outer_blend_max)
    if outer_blend < min_outer_blend and outer_blend_max >= min_outer_blend:
        outer_blend = min_outer_blend

    pure_pc2 = available_outer - edge_w - pc1_pc2_blend - outer_blend
    if pure_pc2 < min_pure_pc2:
        pure_pc2 = min_pure_pc2
        outer_blend = available_outer - edge_w - pc1_pc2_blend - pure_pc2

    if outer_blend <= 0:
        raise ValueError("Not enough room for a valid PC2-to-edge transition.")

    r2 = r1 + pc1_pc2_blend
    r3 = r2 + pure_pc2
    r4 = r3 + outer_blend
    r_blend_start = settings.boz_pc1_blend_start_fraction * r0

    # Parent circles
    r_c0, z_c0 = 0.0, bcr

    z0 = circle_sag(r0, r_c0, z_c0, bcr)
    m0 = circle_slope(r0, r_c0, bcr)
    r_c1, z_c1 = circle_center_from_point_slope_radius(r0, z0, m0, pc1r)

    z1 = circle_sag(r1, r_c1, z_c1, pc1r)
    m1 = circle_slope(r1, r_c1, pc1r)
    r_c2, z_c2 = circle_center_from_point_slope_radius(r1, z1, m1, pc2r)

    z3 = circle_sag(r3, r_c2, z_c2, pc2r)
    m3 = circle_slope(r3, r_c2, pc2r)
    r_ce, z_ce = circle_center_from_point_slope_radius(r3, z3, m3, settings.edge_curve_radius_mm)

    r_all = np.arange(0, r5 + dr / 2, dr)
    z_boz = circle_sag(r_all, r_c0, z_c0, bcr)
    z_pc1 = circle_sag(r_all, r_c1, z_c1, pc1r)
    z_pc2 = circle_sag(r_all, r_c2, z_c2, pc2r)
    z_edge = circle_sag(r_all, r_ce, z_ce, settings.edge_curve_radius_mm)

    z_all = np.full_like(r_all, np.nan, dtype=float)

    # BOZ -> PC1 transition
    z_a = circle_sag(r_blend_start, r_c0, z_c0, bcr)
    m_a = circle_slope(r_blend_start, r_c0, bcr)
    zpp_a = circle_second_derivative_from_slope_radius(m_a, bcr)
    z_b = circle_sag(r0, r_c1, z_c1, pc1r)
    m_b = circle_slope(r0, r_c1, pc1r)
    zpp_b = circle_second_derivative_from_slope_radius(m_b, pc1r)

    mask = r_all <= r_blend_start
    z_all[mask] = z_boz[mask]
    mask = (r_all > r_blend_start) & (r_all < r0)
    z_all[mask] = quintic_c2_patch(r_all[mask], r_blend_start, r0, z_a, m_a, zpp_a, z_b, m_b, zpp_b)

    # Pure PC1
    mask = (r_all >= r0) & (r_all < r1)
    z_all[mask] = z_pc1[mask]

    # PC1 -> PC2 transition
    z_a = circle_sag(r1, r_c1, z_c1, pc1r)
    m_a = circle_slope(r1, r_c1, pc1r)
    zpp_a = circle_second_derivative_from_slope_radius(m_a, pc1r)
    z_b = circle_sag(r2, r_c2, z_c2, pc2r)
    m_b = circle_slope(r2, r_c2, pc2r)
    zpp_b = circle_second_derivative_from_slope_radius(m_b, pc2r)
    mask = (r_all >= r1) & (r_all < r2)
    z_all[mask] = quintic_c2_patch(r_all[mask], r1, r2, z_a, m_a, zpp_a, z_b, m_b, zpp_b)

    # Pure PC2
    mask = (r_all >= r2) & (r_all < r3)
    z_all[mask] = z_pc2[mask]

    # PC2 -> edge transition
    z_a = circle_sag(r3, r_c2, z_c2, pc2r)
    m_a = circle_slope(r3, r_c2, pc2r)
    zpp_a = circle_second_derivative_from_slope_radius(m_a, pc2r)
    z_b = circle_sag(r4, r_ce, z_ce, settings.edge_curve_radius_mm)
    m_b = circle_slope(r4, r_ce, settings.edge_curve_radius_mm)
    zpp_b = circle_second_derivative_from_slope_radius(m_b, settings.edge_curve_radius_mm)
    mask = (r_all >= r3) & (r_all < r4)
    z_all[mask] = quintic_c2_patch(r_all[mask], r3, r4, z_a, m_a, zpp_a, z_b, m_b, zpp_b)

    # Pure edge
    mask = (r_all >= r4) & (r_all <= r5)
    z_all[mask] = z_edge[mask]

    if np.any(~np.isfinite(z_all)):
        raise ValueError("Lens geometry invalid. Check ordered parameters.")

    return r_all, z_all, r5


def calculate_clearance(xq, yq, z_cornea, lens: LensParams, topo: TopographyInputs, settings: AppSettings):
    r_grid = np.hypot(xq, yq)

    r_flat, z_flat, lens_radius = build_tricurve_lens_profile(lens, settings, steep=False)
    z_flat_interp = PchipInterpolator(r_flat, z_flat, extrapolate=False)(r_grid)

    if lens.is_toric:
        r_steep, z_steep, lens_radius_steep = build_tricurve_lens_profile(lens, settings, steep=True)
        lens_radius = min(lens_radius, lens_radius_steep)
        z_steep_interp = PchipInterpolator(r_steep, z_steep, extrapolate=False)(r_grid)
        theta = np.arctan2(yq, xq) - np.deg2rad(topo.flat_axis_deg)
        w_flat = np.cos(theta) ** 2
        w_steep = np.sin(theta) ** 2
        z_lens0 = w_flat * z_flat_interp + w_steep * z_steep_interp
    else:
        z_lens0 = z_flat_interp

    z_lens0 = np.where(r_grid > lens_radius, np.nan, z_lens0)

    clearance0_um = (z_cornea - z_lens0) * 1000
    target_min_um = -settings.max_penetration_um
    min0_um = np.nanmin(clearance0_um)
    off_mm = (min0_um - target_min_um) / 1000
    z_lens_seated = z_lens0 + off_mm

    clearance_um = (z_cornea - z_lens_seated) * 1000
    clearance_um = np.where(clearance_um < target_min_um - 0.05, target_min_um, clearance_um)
    return clearance_um, z_lens_seated, lens_radius, r_flat, z_flat


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def medmont_clearance_cmap():
    x_anchor = np.array([0, 5, 10, 20, 30, 50, 75, 100], dtype=float) / 100
    g_anchor = np.array([0, 0.02, 0.08, 0.28, 0.45, 0.72, 0.88, 1.00], dtype=float)
    colors = [(x, (0, g, 0)) for x, g in zip(x_anchor, g_anchor)]
    return LinearSegmentedColormap.from_list("medmont_green", colors, N=256)


def plot_fluorescein_map(xq, yq, clearance_um):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(
        clearance_um,
        extent=[xq[0, 0], xq[0, -1], yq[0, 0], yq[-1, 0]],
        origin="lower",
        cmap=medmont_clearance_cmap(),
        vmin=0,
        vmax=100,
    )
    ax.set_aspect("equal")
    ax.set_title("FL Simulation")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Clearance (µm)")
    cbar.set_ticks([0, 10, 20, 30, 50, 100])
    return fig


def plot_tangential_map(topo: TopographyInputs, settings: AppSettings):
    xq, yq, z = build_corneal_sag_grid(topo, settings)
    dx = settings.grid_step_mm
    rho = np.hypot(xq, yq)
    outside = rho > settings.map_diameter_mm / 2

    # IMPORTANT:
    # np.gradient returns derivatives in array-axis order: row/y first, column/x second.
    # MATLAB's gradient output order differs, so using zx, zy = np.gradient(...)
    # swaps the x/y derivatives and rotates the topography colour pattern by 90 degrees.
    zy, zx = np.gradient(z, dx, dx)

    # Second derivatives:
    # gradient(zx) -> [dZx/dy, dZx/dx] = [Zxy, Zxx]
    # gradient(zy) -> [dZy/dy, dZy/dx] = [Zyy, Zxy]
    zxy1, zxx = np.gradient(zx, dx, dx)
    zyy, zxy2 = np.gradient(zy, dx, dx)
    zxy = 0.5 * (zxy1 + zxy2)

    denom_n = np.sqrt(1 + zx**2 + zy**2)
    e2 = zxx / denom_n
    f2 = zxy / denom_n
    g2 = zyy / denom_n

    E = 1 + zx**2
    F = zx * zy
    G = 1 + zy**2

    ux = np.divide(xq, rho, out=np.ones_like(xq), where=rho != 0)
    uy = np.divide(yq, rho, out=np.zeros_like(yq), where=rho != 0)

    num = e2 * ux**2 + 2 * f2 * ux * uy + g2 * uy**2
    den = E * ux**2 + 2 * F * ux * uy + G * uy**2

    k_tan = num / den
    r_tan_mm = 1 / np.abs(k_tan)
    dioptres = 1000 * (settings.n_k - 1) / r_tan_mm
    dioptres = np.where(outside, np.nan, dioptres)

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    im = ax.imshow(
        dioptres,
        extent=[xq[0, 0], xq[0, -1], yq[0, 0], yq[-1, 0]],
        origin="lower",
        cmap="jet",
        vmin=35,
        vmax=50,
    )
    ax.set_aspect("equal")
    ax.set_title("Topography")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Tangential curvature (D)")
    return fig


def sample_line(field, xq, yq, x_line, y_line):
    xv = xq[0, :]
    yv = yq[:, 0]
    interpolator = RegularGridInterpolator((yv, xv), field, bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([y_line, x_line])
    return interpolator(pts)


def plot_tear_profile(xq, yq, clearance_um, lens_radius, axis_deg):
    r_line = min(6.0, lens_radius)
    s = np.linspace(-r_line, r_line, 1200)
    x_line = s * np.cos(np.deg2rad(axis_deg))
    y_line = s * np.sin(np.deg2rad(axis_deg))
    c_line = sample_line(clearance_um, xq, yq, x_line, y_line)

    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    ax.plot(s, c_line, linewidth=1.5)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.axhline(20, linestyle=":", linewidth=1)
    ax.grid(True)
    ax.set_xlim(-r_line, r_line)
    ax.set_xlabel(f"Distance along {axis_deg:.1f}° axis (mm)")
    ax.set_ylabel("Clearance (µm)")
    ax.set_title("Tear Profile")
    return fig


def plot_cross_section(xq, yq, z_cornea, z_lens_seated, lens_radius, axis_deg, settings: AppSettings):
    r_line = min(settings.map_diameter_mm / 2, lens_radius)
    s = np.linspace(-r_line, r_line, 1200)
    x_line = s * np.cos(np.deg2rad(axis_deg))
    y_line = s * np.sin(np.deg2rad(axis_deg))

    z_cor_line = sample_line(z_cornea, xq, yq, x_line, y_line)
    z_len_line = sample_line(z_lens_seated, xq, yq, x_line, y_line)
    z_len_plot = z_cor_line - settings.cross_section_separation_scale * (z_cor_line - z_len_line)

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(s, -z_cor_line * 1000, label="Cornea", linewidth=1.0)
    ax.plot(s, -z_len_plot * 1000, label="Lens", linewidth=1.0)
    ax.grid(True)
    ax.set_xlabel(f"Distance along {axis_deg:.1f}° axis (mm)")
    ax.set_ylabel("Sag (µm)")
    ax.set_title("Cornea / Lens Cross-Section")
    ax.legend(loc="best")
    ax.set_yticklabels([])
    return fig




# -----------------------------------------------------------------------------
# Optics / order summary
# -----------------------------------------------------------------------------

def normalize_axis180(axis):
    axis = axis % 180
    if axis <= 0:
        axis += 180
    return axis


def principal_powers_to_minus_cyl(Faxis, Fperp, Aaxis):
    if Faxis >= Fperp:
        S = Faxis
        C = Fperp - Faxis
        Aout = Aaxis % 180
    else:
        S = Fperp
        C = Faxis - Fperp
        Aout = (Aaxis + 90) % 180

    if Aout == 0:
        Aout = 180
    return S, C, Aout


def vertex_convert_rx(S, C, A, vertex_mm):
    d = vertex_mm / 1000.0
    F1 = S
    F2 = S + C

    F1v = F1 / (1 - d * F1)
    F2v = F2 / (1 - d * F2)

    return principal_powers_to_minus_cyl(F1v, F2v, A)


def rx_to_power_vectors(S, C, A):
    A = A % 180
    M = S + C / 2
    J0 = (-C / 2) * np.cos(np.deg2rad(2 * A))
    J45 = (-C / 2) * np.sin(np.deg2rad(2 * A))
    return M, J0, J45


def power_vectors_to_minus_cyl(M, J0, J45):
    mag = np.hypot(J0, J45)
    C = -2 * mag
    S = M - C / 2

    if mag < 1e-10:
        A = 180
    else:
        A = 0.5 * np.rad2deg(np.arctan2(J45, J0))
        if A <= 0:
            A += 180

    return S, C, A


def format_rx_string(S, C, A):
    S = round(S * 4) / 4
    C = round(C * 4) / 4
    A = round(A) % 180
    if A == 0:
        A = 180

    if abs(C) < 0.125:
        return f"{S:+.2f} DS"
    return f"{S:+.2f} / {C:+.2f} x {A:03d}"


def signed_axis_delta180(observed_axis, reference_axis):
    return ((observed_axis - reference_axis + 90) % 180) - 90


def compute_order_summary(
    topo: TopographyInputs,
    lens: LensParams,
    settings: AppSettings,
    has_orx=False,
    orx_sphere_d=0.0,
    orx_cyl_d=0.0,
    orx_axis_deg=180.0,
    lens_locate_axis_deg=None,
):
    """
    Python port of the MATLAB computeOrderOpticsSummary/updateOrderBox logic.

    Assumptions kept from the MATLAB prototype:
        - vertex distance = 12 mm
        - tear lens effective index = 1.3375
        - spherical lens = front BVP only
        - toric lens without ORx = SPE bitoric assumption
        - toric lens with ORx = CPE bitoric compensated front toric order
    """

    vertex_distance_mm = 12.0
    n_eff = 1.3375

    flat_axis = normalize_axis180(topo.flat_axis_deg)

    # Spectacle Rx at corneal plane
    spec_s, spec_c, spec_a = vertex_convert_rx(
        topo.spec_sphere_d,
        topo.spec_cyl_d,
        topo.spec_axis_deg,
        vertex_distance_mm,
    )
    Mspec, J0spec, J45spec = rx_to_power_vectors(spec_s, spec_c, spec_a)

    # Tear lens from back surface versus cornea
    Kflat = 1000 * (n_eff - 1) / topo.r_flat
    Ksteep = 1000 * (n_eff - 1) / topo.r_steep

    BCflat = lens.boz_r
    BCsteep = lens.boz_r - lens.toricity_mm if lens.is_toric else lens.boz_r

    if BCsteep <= 0:
        raise ValueError("Toricity too large: steep BOZR becomes <= 0.")

    FBCflat = 1000 * (n_eff - 1) / BCflat
    FBCsteep = 1000 * (n_eff - 1) / BCsteep

    tear_flat = FBCflat - Kflat
    tear_steep = FBCsteep - Ksteep

    tear_s, tear_c, tear_a = principal_powers_to_minus_cyl(tear_flat, tear_steep, flat_axis)
    Mtear, J0tear, J45tear = rx_to_power_vectors(tear_s, tear_c, tear_a)

    # Current trial lens air powers
    if lens.is_toric:
        # SPE bitoric assumption:
        #   Dioptric difference between base curves ~= dioptric difference
        #   between the meridional lens BVPs.
        #
        # The entered Lens BVP is treated as the flat-meridian/labelled SPE power.
        # The steeper base-curve meridian is assigned extra minus power equal to
        # the base-curve dioptric difference, using the tear/air keratometric index.
        #
        # Example: 8.00 / 7.50 mm gives
        #   delta_bc_d = 1000*(1.3375 - 1)*(1/7.50 - 1/8.00) ~= +2.81 D
        #   lens_air_flat_trial  = entered BVP
        #   lens_air_steep_trial = entered BVP - 2.81 D
        delta_bc_d = FBCsteep - FBCflat
        lens_air_flat_trial = topo.lens_bvp_d
        lens_air_steep_trial = topo.lens_bvp_d - delta_bc_d
    else:
        lens_air_flat_trial = topo.lens_bvp_d
        lens_air_steep_trial = topo.lens_bvp_d

    trial_lens_air_s, trial_lens_air_c, trial_lens_air_a = principal_powers_to_minus_cyl(
        lens_air_flat_trial,
        lens_air_steep_trial,
        flat_axis,
    )
    Mlens_trial, J0lens_trial, J45lens_trial = rx_to_power_vectors(
        trial_lens_air_s,
        trial_lens_air_c,
        trial_lens_air_a,
    )

    # Current on-eye system
    Mcurrent = Mlens_trial + Mtear
    J0current = J0lens_trial + J0tear
    J45current = J45lens_trial + J45tear
    current_sys_s, current_sys_c, current_sys_a = power_vectors_to_minus_cyl(
        Mcurrent,
        J0current,
        J45current,
    )

    # Defaults
    Mtarget = Mcurrent
    J0target = J0current
    J45target = J45current

    lens_air_flat = lens_air_flat_trial
    lens_air_steep = lens_air_steep_trial
    lens_air_s = trial_lens_air_s
    lens_air_c = trial_lens_air_c
    lens_air_a = trial_lens_air_a
    lens_air_order_a = trial_lens_air_a

    locating_axis = np.nan
    rotation_delta = np.nan

    # ORx-driven CPE bitoric
    if lens.is_toric and has_orx:
        if lens_locate_axis_deg is None:
            lens_locate_axis_deg = flat_axis

        locating_axis = normalize_axis180(lens_locate_axis_deg)

        orx_s_c, orx_c_c, orx_a_c = vertex_convert_rx(
            orx_sphere_d,
            orx_cyl_d,
            orx_axis_deg,
            vertex_distance_mm,
        )
        Morx, J0orx, J45orx = rx_to_power_vectors(orx_s_c, orx_c_c, orx_a_c)

        # Desired final on-eye system = current system + ORx
        Mtarget = Mcurrent + Morx
        J0target = J0current + J0orx
        J45target = J45current + J45orx

        # Required lens air power = desired system - tear lens
        Mlens_req = Mtarget - Mtear
        J0lens_req = J0target - J0tear
        J45lens_req = J45target - J45tear

        lens_air_s, lens_air_c, lens_air_a = power_vectors_to_minus_cyl(
            Mlens_req,
            J0lens_req,
            J45lens_req,
        )

        lens_air_flat = lens_air_s
        lens_air_steep = lens_air_s + lens_air_c

        if abs(lens_air_c) >= 0.125:
            rotation_delta = signed_axis_delta180(locating_axis, flat_axis)
            lens_air_order_a = normalize_axis180(lens_air_a - rotation_delta)
        else:
            lens_air_order_a = 180

    target_sys_s, target_sys_c, target_sys_a = power_vectors_to_minus_cyl(
        Mtarget,
        J0target,
        J45target,
    )

    # Predicted residual = spec Rx at cornea - target system
    Mres = Mspec - Mtarget
    J0res = J0spec - J0target
    J45res = J45spec - J45target
    res_s, res_c, res_a = power_vectors_to_minus_cyl(Mres, J0res, J45res)

    return {
        "flat_axis": flat_axis,
        "locating_axis": locating_axis,
        "rotation_delta": rotation_delta,
        "spec": (spec_s, spec_c, spec_a),
        "tear_flat": tear_flat,
        "tear_steep": tear_steep,
        "tear": (tear_s, tear_c, tear_a),
        "current_system": (current_sys_s, current_sys_c, current_sys_a),
        "target_system": (target_sys_s, target_sys_c, target_sys_a),
        "lens_air_flat": lens_air_flat,
        "lens_air_steep": lens_air_steep,
        "lens_air": (lens_air_s, lens_air_c, lens_air_a),
        "lens_air_order_axis": lens_air_order_a,
        "residual": (res_s, res_c, res_a),
    }


def build_order_lines(topo, lens, settings, has_orx=False, orx_sphere_d=0.0, orx_cyl_d=0.0, orx_axis_deg=180.0, lens_locate_axis_deg=None):
    O = compute_order_summary(
        topo,
        lens,
        settings,
        has_orx=has_orx,
        orx_sphere_d=orx_sphere_d,
        orx_cyl_d=orx_cyl_d,
        orx_axis_deg=orx_axis_deg,
        lens_locate_axis_deg=lens_locate_axis_deg,
    )

    spec_s, spec_c, spec_a = O["spec"]
    target_s, target_c, target_a = O["target_system"]
    res_s, res_c, res_a = O["residual"]

    if not lens.is_toric:
        return [
            "Design: Spherical",
            f"Back surface: {lens.boz_r:.2f}:{lens.boz_d:.2f} / {lens.pc1_r:.2f}:{lens.pc1_d:.2f} / {lens.pc2_r:.2f}:{lens.pc2_d:.2f}",
            f"Lens BVP in air: {topo.lens_bvp_d:+.2f} D",
            f"Spec @ cornea: {format_rx_string(spec_s, spec_c, spec_a)}",
            f"Tear lens: flat {O['tear_flat']:+.2f} D, steep {O['tear_steep']:+.2f} D",
            f"Estimated on-eye system: {format_rx_string(target_s, target_c, target_a)}",
            f"Estimated residual: {format_rx_string(res_s, res_c, res_a)}",
        ]

    flat_bc = lens.boz_r
    steep_bc = lens.boz_r - lens.toricity_mm
    flat_pc1 = lens.pc1_r
    steep_pc1 = lens.pc1_r - lens.toricity_mm
    flat_pc2 = lens.pc2_r
    steep_pc2 = lens.pc2_r - lens.toricity_mm

    flat_line = f"Back flat: {flat_bc:.2f}:{lens.boz_d:.2f} / {flat_pc1:.2f}:{lens.pc1_d:.2f} / {flat_pc2:.2f}:{lens.pc2_d:.2f}"
    steep_line = f"Back steep: {steep_bc:.2f}:{lens.boz_d:.2f} / {steep_pc1:.2f}:{lens.pc1_d:.2f} / {steep_pc2:.2f}:{lens.pc2_d:.2f}"
    tor_line = f"Back toricity: {lens.toricity_mm:.2f} mm"
    spec_line = f"Spec @ cornea: {format_rx_string(spec_s, spec_c, spec_a)}"
    tear_line = f"Tear lens: flat {O['tear_flat']:+.2f} D, steep {O['tear_steep']:+.2f} D"

    if has_orx:
        current_s, current_c, current_a = O["current_system"]
        lens_air_s, lens_air_c, lens_air_a = O["lens_air"]
        return [
            "Design: CPE bitoric",
            f"Back flat axis: {round(O['flat_axis']):03d}",
            f"Lens locating axis: {round(O['locating_axis']):03d}",
            flat_line,
            steep_line,
            tor_line,
            f"Entered ORx: {format_rx_string(orx_sphere_d, orx_cyl_d, orx_axis_deg)}",
            spec_line,
            tear_line,
            f"Current on-eye system: {format_rx_string(current_s, current_c, current_a)}",
            f"Compensated front toric order: {format_rx_string(lens_air_s, lens_air_c, O['lens_air_order_axis'])}",
            f"Predicted final on-eye system: {format_rx_string(target_s, target_c, target_a)}",
            f"Predicted residual: {format_rx_string(res_s, res_c, res_a)}",
        ]

    return [
        "Design: SPE bitoric",
        f"Axis (flat meridian): {round(O['flat_axis']):03d}",
        flat_line,
        steep_line,
        tor_line,
        f"Lens BVP in air: flat {O['lens_air_flat']:+.2f} D, steep {O['lens_air_steep']:+.2f} D",
        f"Nominal labelled SPE power: {topo.lens_bvp_d:+.2f} DS",
        spec_line,
        tear_line,
        f"Estimated on-eye system: {format_rx_string(target_s, target_c, target_a)}",
        f"Estimated residual: {format_rx_string(res_s, res_c, res_a)}",
    ]


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Tricurve Lens Simulator", layout="wide")
    st.title("Tricurve Lens Simulator")

    settings = AppSettings()

    with st.sidebar:
        st.header("Topography")
        topo = TopographyInputs(
            r_flat=st.number_input("Flat r (mm)", value=8.13, min_value=4.0, max_value=12.0, step=0.01),
            r_steep=st.number_input("Steep r (mm)", value=7.93, min_value=4.0, max_value=12.0, step=0.01),
            e_flat=st.number_input("Flat e", value=0.70, min_value=0.0, step=0.01),
            e_steep=st.number_input("Steep e", value=0.24, min_value=0.0, step=0.01),
            flat_axis_deg=st.number_input("Flat axis (deg)", value=11.0, min_value=0.0, max_value=180.0, step=1.0),
            spec_sphere_d=st.number_input("Spectacle sphere (D)", value=0.0, step=0.25),
            spec_cyl_d=st.number_input("Spectacle cyl (D)", value=0.0, step=0.25),
            spec_axis_deg=st.number_input("Spectacle axis (deg)", value=180.0, min_value=0.0, max_value=180.0, step=1.0),
            lens_bvp_d=st.number_input("Lens BVP (D)", value=0.0, step=0.25),
        )

        st.header("Lens")
        lens_type = st.selectbox("Lens type", ["Spherical", "Toric"])
        lens = LensParams(
            boz_r=st.number_input("BOZr", value=8.00, min_value=4.0, max_value=12.0, step=0.01),
            boz_d=st.number_input("BOZD", value=5.32, min_value=1.0, max_value=12.0, step=0.01),
            pc1_r=st.number_input("PC1r", value=8.32, min_value=4.0, max_value=20.0, step=0.01),
            pc1_d=st.number_input("PC1D", value=6.90, min_value=1.0, max_value=14.0, step=0.01),
            pc2_r=st.number_input("PC2r", value=9.475, min_value=4.0, max_value=30.0, step=0.01),
            pc2_d=st.number_input("PC2D / TD", value=9.00, min_value=1.0, max_value=18.0, step=0.01),
            is_toric=lens_type == "Toric",
            toricity_mm=st.number_input("Toricity (mm)", value=0.50, min_value=0.01, step=0.01) if lens_type == "Toric" else 0.0,
        )

        has_orx = False
        orx_sphere_d = 0.0
        orx_cyl_d = 0.0
        orx_axis_deg = 180.0
        lens_locate_axis_deg = topo.flat_axis_deg

        if lens_type == "Toric":
            st.header("Over-refraction / CPE")
            has_orx = st.checkbox("Use ORx to calculate CPE bitoric order", value=False)
            if has_orx:
                orx_sphere_d = st.number_input("ORx sphere (D)", value=0.0, step=0.25)
                orx_cyl_d = st.number_input("ORx cyl (D)", value=0.0, step=0.25)
                orx_axis_deg = st.number_input("ORx axis (deg)", value=180.0, min_value=0.0, max_value=180.0, step=1.0)
                lens_locate_axis_deg = st.number_input("Lens locating axis (deg)", value=float(topo.flat_axis_deg), min_value=0.0, max_value=180.0, step=1.0)

        st.header("Display")
        axis_deg = st.slider("Axis", min_value=0.0, max_value=180.0, value=0.0, step=1.0)

    if lens.pc1_d < lens.boz_d:
        st.error("PC1D must be greater than or equal to BOZD.")
        return
    if lens.pc2_d <= lens.pc1_d:
        st.error("PC2D / TD must be greater than PC1D.")
        return

    try:
        xq, yq, z_cornea = build_corneal_sag_grid(topo, settings)
        clearance_um, z_lens_seated, lens_radius, lens_r, lens_z = calculate_clearance(
            xq, yq, z_cornea, lens, topo, settings
        )
    except Exception as exc:
        st.error(str(exc))
        return

    col_topo, col_fl = st.columns([1, 2])
    with col_topo:
        st.pyplot(plot_tangential_map(topo, settings), clear_figure=True)
    with col_fl:
        st.pyplot(plot_fluorescein_map(xq, yq, clearance_um), clear_figure=True)

    col_cross, col_tear = st.columns(2)
    with col_cross:
        st.pyplot(plot_cross_section(xq, yq, z_cornea, z_lens_seated, lens_radius, axis_deg, settings), clear_figure=True)
    with col_tear:
        st.pyplot(plot_tear_profile(xq, yq, clearance_um, lens_radius, axis_deg), clear_figure=True)

    st.subheader("Order summary")
    try:
        order_lines = build_order_lines(
            topo,
            lens,
            settings,
            has_orx=has_orx,
            orx_sphere_d=orx_sphere_d,
            orx_cyl_d=orx_cyl_d,
            orx_axis_deg=orx_axis_deg,
            lens_locate_axis_deg=lens_locate_axis_deg,
        )
        st.code("\n".join(order_lines), language=None)
    except Exception as exc:
        st.error(f"Order summary error: {exc}")

    with st.expander("Debug values"):
        st.write(
            {
                "lens_radius_mm": lens_radius,
                "min_clearance_um": float(np.nanmin(clearance_um)),
                "max_clearance_um": float(np.nanmax(clearance_um)),
            }
        )


if __name__ == "__main__":
    main()
