Web VPython 3.2

# Rainbow order visualizer -- Web VPython (glowscript.org) single-file build.
# Merges rainbow_app.py + rainbow_optics.py since Web VPython only allows
# "import vpython" and "import random" (no os/sys, no local multi-file
# imports). See rainbow_app.py + rainbow_optics.py for the two-file version
# that runs locally via a real Python interpreter.
#
# Two modes, toggled live: droplet ray-tracing (the actual refracted /
# internally-reflected light path through a droplet, at 3 sample
# wavelengths) and sky dome (where each order's bow actually sits relative
# to the sun / antisolar point). Orders 1-2 are the ordinarily-visible
# primary/secondary rainbow; orders 3+ are real but effectively invisible
# in nature, rendered here at full strength so you can see them, with a
# "realistic brightness" toggle to dim them back down.
# Physics validated against Konnen 2015 / Suzuki & Suzuki 2010 / Lock 1987.

from vpython import ( canvas, sphere, curve, vector, color, label, arrow, rate, radio, checkbox, slider, wtext, sqrt, sin, cos, asin, acos, pi, round, )

# vpython re-exports these math names for convenience (standard VPython
# idiom); they are NOT otherwise available as bare globals. radians()/
# degrees() aren't among them though, so define those two ourselves.
def radians(deg):
    return deg * pi / 180.0


def degrees(rad):
    return rad * 180.0 / pi


VISIBLE_MIN_NM = 400.0
VISIBLE_MAX_NM = 700.0


# Refractive index of water, Cauchy 2-term fit to visible-range data
# (fit to n(400nm)=1.343, n(700nm)=1.331; matches n(589nm)=1.3334 vs the
# real sodium-D value of 1.3330).
def water_index(wavelength_nm):
    lam_um = wavelength_nm / 1000.0
    return 1.325184 + 0.00285 / lam_um ** 2


def _scale_component(c, factor, gamma):
    return (c * factor) ** gamma if c > 0 else 0.0


# Approximate visible-spectrum wavelength -> (r,g,b) in [0,1].
# Classic Dan Bruton algorithm, public domain.
def wavelength_to_rgb(wavelength_nm):
    w = wavelength_nm
    if 380 <= w < 440:
        r, g, b = -(w - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / (490 - 440), 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        r, g, b = (w - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / (645 - 580), 0.0
    elif 645 <= w <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0

    if 380 <= w < 420:
        factor = 0.3 + 0.7 * (w - 380) / (420 - 380)
    elif 420 <= w < 700:
        factor = 1.0
    elif 700 <= w <= 780:
        factor = 0.3 + 0.7 * (780 - w) / (780 - 700)
    else:
        factor = 0.0

    gamma = 0.8
    return (_scale_component(r, factor, gamma), _scale_component(g, factor, gamma), _scale_component(b, factor, gamma))


# Descartes ray for order k (k internal reflections, p = k+1 chords).
# Returns (i, r, D, theta_scatter) all in radians: i = angle of incidence,
# r = angle of refraction, D = total ray deviation, theta_scatter =
# scattering angle folded into [0, pi] measured from the forward direction.
def rainbow_geometry(k, n):
    p = k + 1
    cos_i = sqrt((n ** 2 - 1) / (p ** 2 - 1))
    i = acos(cos_i)
    r = asin(sin(i) / n)
    D = 2 * (i - r) + k * (pi - 2 * r)
    theta = D % (2 * pi)
    if theta > pi:
        theta = 2 * pi - theta
    return i, r, D, theta


# Where order-k appears in the sky for a given wavelength. Returns
# (side, angle_deg): side is 'antisolar' (classic rainbow side) or 'solar'
# (buried in glare); angle_deg is measured from the antisolar point or the
# sun respectively. Rule: theta_scatter < 90 deg -> solar side (still
# mostly forward-scattered), theta_scatter > 90 deg -> antisolar side.
def sky_position(k, wavelength_nm):
    n = water_index(wavelength_nm)
    _, _, _, theta = rainbow_geometry(k, n)
    theta_deg = degrees(theta)
    if theta_deg > 90.0:
        return "antisolar", 180.0 - theta_deg
    else:
        return "solar", theta_deg


# Degree of polarization P(k), percent. k=1,2 from Suzuki & Suzuki (2010);
# k=3..7 and the k->infinity asymptote from Konnen (2015) Table 1 / Eq. 7,
# using the average of the two listed 5th-order rows (5b/5c: 76.2%/76.8%).
_POLARIZATION_TABLE_PCT = { 1: 94.0, 2: 90.0, 3: 78.1, 4: 77.0, 5: 76.5, 6: 76.1, 7: 75.9, }
_POLARIZATION_ASYMPTOTE_PCT = 75.3  # k -> infinity, Konnen Eq. 7


def polarization_percent(k):
    if k in _POLARIZATION_TABLE_PCT:
        return _POLARIZATION_TABLE_PCT[k]
    return _POLARIZATION_ASYMPTOTE_PCT


# Only the primary and secondary rainbow are ordinarily visible to the
# naked eye; k>=3 requires ideal dark conditions and, in practice, has
# only ever been captured photographically (Konnen 2015).
def naked_eye_visible(k):
    return k <= 2


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    m = sqrt(_dot(a, a))
    return (a[0] / m, a[1] / m, a[2] / m)


def _scale_add(a, s, b):
    return (a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2])


def _reflect(d, normal):
    dn = _dot(d, normal)
    return (d[0] - 2 * dn * normal[0], d[1] - 2 * dn * normal[1], d[2] - 2 * dn * normal[2])


# d: unit incident direction of travel. normal_outward: the sphere's
# outward normal at the surface point. eta = n_from/n_to. Returns the
# refracted unit direction, or None on total internal reflection.
def _refract(d, normal_outward, eta):
    if _dot(d, normal_outward) < 0:
        n_used = normal_outward
    else:
        n_used = (-normal_outward[0], -normal_outward[1], -normal_outward[2])
    cos_i = -_dot(d, n_used)
    sin2_t = eta ** 2 * (1 - cos_i ** 2)
    if sin2_t >= 1.0:
        return None
    cos_t = sqrt(1 - sin2_t)
    coef = eta * cos_i - cos_t
    return (eta * d[0] + coef * n_used[0], eta * d[1] + coef * n_used[1], eta * d[2] + coef * n_used[2])


# Trace the actual Descartes rainbow ray of order k through a spherical
# droplet, in the plane spanned by incident_dir and an arbitrary
# perpendicular. Returns a dict with points (entry, k reflections, exit),
# exit_dir, and theta_scatter_deg (for cross-checking against
# rainbow_geometry()).
def trace_ray_path(k, n, radius=1.0, incident_dir=None):
    if incident_dir is None:
        incident_dir = (1.0, 0.0, 0.0)
    i_angle, _, _, _ = rainbow_geometry(k, n)
    return trace_ray_at_incidence(k, n, i_angle, radius=radius, incident_dir=incident_dir)


# Like trace_ray_path, but for an arbitrary angle of incidence i_angle
# (radians) rather than the exact Descartes rainbow-ray angle. Tracing a
# handful of these around the Descartes angle is what shows the caustic:
# nearby rays bunch up near the same exit direction, which is why that
# particular angle produces a rainbow.
def trace_ray_at_incidence(k, n, i_angle, radius=1.0, incident_dir=None):
    if incident_dir is None:
        incident_dir = (1.0, 0.0, 0.0)
    incident_dir = _norm(incident_dir)

    # Build an orthonormal basis (u, v) spanning the plane of incidence,
    # with u = incident_dir.
    u = incident_dir
    # pick any vector not parallel to u
    if abs(u[0]) < 0.9:
        ref = (0.0, 1.0, 0.0)
    else:
        ref = (0.0, 0.0, 1.0)
    v = (ref[0] - _dot(ref, u) * u[0], ref[1] - _dot(ref, u) * u[1], ref[2] - _dot(ref, u) * u[2])
    v = _norm(v)

    # Entry point: P0 = R*(-cos(i)*u + sin(i)*v)
    entry = (radius * (-cos(i_angle) * u[0] + sin(i_angle) * v[0]), radius * (-cos(i_angle) * u[1] + sin(i_angle) * v[1]), radius * (-cos(i_angle) * u[2] + sin(i_angle) * v[2]))
    normal0 = _norm(entry)
    d = _refract(incident_dir, normal0, 1.0 / n)

    points = [entry]
    current = entry
    for _ in range(k):
        t = -2 * _dot(current, d)
        nxt = _scale_add(current, t, d)
        points.append(nxt)
        normal = _norm(nxt)
        d = _reflect(d, normal)
        current = nxt

    t = -2 * _dot(current, d)
    exit_point = _scale_add(current, t, d)
    points.append(exit_point)
    normal = _norm(exit_point)
    exit_dir = _refract(d, normal, n / 1.0)
    if exit_dir is None:
        exit_dir = d  # shouldn't happen for a valid Descartes ray

    cos_theta = max(-1.0, min(1.0, _dot(exit_dir, incident_dir)))
    theta_check = degrees(acos(cos_theta))

    return { "points": points, "exit_dir": exit_dir, "theta_scatter_deg": theta_check, }


DROPLET_RADIUS = 3.0
SKY_RADIUS = 10.0
MAX_ORDER_LIMIT = 13  # Lock (1987): over a dozen rainbows observed in a single droplet
DISPERSION_WAVELENGTHS_NM = [460.0, 550.0, 640.0]  # blue, green, red samples
SKY_SWEEP_WAVELENGTHS_NM = [400 + i * (300.0 / 14) for i in range(15)]

# Fan-ray fractions of the adaptive max spread (see fan_deltas_deg below).
# Kept to 2 symmetric pairs (4 rays) to stay visually light.
FAN_FRACTIONS = [0.35, 0.8]


# Adaptively-scaled incidence-angle offsets (degrees) for the fan of rays
# drawn around the Descartes ray of a given order. i_descartes climbs
# toward 90 deg as k grows, so a fixed +/-6 deg spread would push past the
# 90 deg domain boundary at high k (dropping rays) and land on a much
# steeper part of the deviation curve (swinging exit angle 5-10 deg
# instead of ~1 deg). Scaling the max spread down as i_descartes -> 90 deg
# keeps every order's fan symmetric, in-domain, and visually similar.
def fan_deltas_deg(i_descartes_rad):
    i_deg = degrees(i_descartes_rad)
    max_delta = min(6.0, 0.6 * (90.0 - i_deg))
    deltas = []
    for frac in FAN_FRACTIONS:
        deltas.append(-frac * max_delta)
        deltas.append(frac * max_delta)
    return deltas

ORDER_NAMES = {1: "primary", 2: "secondary", 3: "tertiary", 4: "quaternary", 5: "quinary", 6: "senary", 7: "septenary", 8: "octonary"}


def order_name(k):
    return ORDER_NAMES.get(k, "order " + str(k))

scene = canvas(title="Rainbow Order Visualizer", width=1200, height=800, background=vector(0.05, 0.05, 0.1))
scene.range = SKY_RADIUS * 1.3
scene.forward = vector(-0.4, -0.35, -1)

state = {"mode": "droplet", "max_order": 4, "realistic": False, "labels": True, "show_fan": True}
_drawn = []


def _clear():
    global _drawn
    for obj in _drawn:
        obj.visible = False
    _drawn = []


def _opacity_for(k):
    if naked_eye_visible(k):
        return 1.0
    return 0.15 if state["realistic"] else 0.65


def draw_droplet_mode():
    droplet = sphere(pos=vector(0, 0, 0), radius=DROPLET_RADIUS, color=vector(0.6, 0.8, 1.0), opacity=0.15, shininess=0.9)
    _drawn.append(droplet)

    # context: a handful of parallel incident sunlight rays
    for frac in (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
        b = frac * DROPLET_RADIUS
        start = vector(-DROPLET_RADIUS * 2.2, b, 0)
        end = vector(-sqrt(max(DROPLET_RADIUS ** 2 - b ** 2, 0.0)), b, 0)
        ray = curve(pos=[start, end], color=vector(1, 1, 0.6), radius=0.01 * DROPLET_RADIUS, opacity=0.35)
        _drawn.append(ray)

    for k in range(1, state["max_order"] + 1):
        opacity = _opacity_for(k)
        last_exit_pt = None

        if state["show_fan"]:
            n_ref = water_index(550.0)
            i_ref, _, _, _ = rainbow_geometry(k, n_ref)
            fan_color = vector(0.75, 0.75, 0.75)
            for delta_deg in fan_deltas_deg(i_ref):
                i_angle = i_ref + radians(delta_deg)
                if not (0.0 < i_angle < pi / 2):
                    continue
                traced = trace_ray_at_incidence(k, n_ref, i_angle, radius=DROPLET_RADIUS)
                pts = [vector(*p) for p in traced["points"]]
                exit_dir = vector(*traced["exit_dir"])
                pts.append(pts[-1] + exit_dir * (DROPLET_RADIUS * 3.0))
                c = curve(pos=pts, color=fan_color, radius=0.005 * DROPLET_RADIUS, opacity=opacity * 0.25)
                _drawn.append(c)

        for wl in DISPERSION_WAVELENGTHS_NM:
            n = water_index(wl)
            traced = trace_ray_path(k, n, radius=DROPLET_RADIUS)
            pts = [vector(*p) for p in traced["points"]]
            exit_dir = vector(*traced["exit_dir"])
            extended = pts[-1] + exit_dir * (DROPLET_RADIUS * 3.0)
            pts.append(extended)
            rgb = wavelength_to_rgb(wl)
            c = curve(pos=pts, color=vector(*rgb), radius=0.02 * DROPLET_RADIUS, opacity=opacity)
            _drawn.append(c)
            last_exit_pt = extended

        if state["labels"] and last_exit_pt is not None:
            name = order_name(k)
            vis = "visible" if naked_eye_visible(k) else "not naturally visible"
            pct = polarization_percent(k)
            label_text = "k=" + str(k) + " (" + name + ")\n" + str(int(round(pct))) + "% polarized, " + vis
            lbl = label(pos=last_exit_pt, text=label_text, height=12, box=False, opacity=0, line=False, color=vector(*wavelength_to_rgb(550)))
            _drawn.append(lbl)

    origin_marker = sphere(pos=vector(0, 0, 0), radius=0.02 * DROPLET_RADIUS, color=color.white, opacity=0)
    _drawn.append(origin_marker)
    scene.range = DROPLET_RADIUS * 4.5
    scene.center = vector(DROPLET_RADIUS * 0.5, 0, 0)


def draw_sky_mode():
    sun_arrow = arrow(pos=vector(0, 0, -SKY_RADIUS * 0.15), axis=vector(0, 0, -SKY_RADIUS * 0.5), color=vector(1, 0.9, 0.3), shaftwidth=SKY_RADIUS * 0.02)
    _drawn.append(sun_arrow)
    sun_lbl = label(pos=vector(0, 0, -SKY_RADIUS * 0.7), text="Sun", height=14, box=False, opacity=0, color=vector(1, 0.9, 0.3))
    _drawn.append(sun_lbl)

    anti_marker = sphere(pos=vector(0, 0, SKY_RADIUS), radius=SKY_RADIUS * 0.04, color=color.white, opacity=0.8)
    _drawn.append(anti_marker)
    anti_lbl = label(pos=vector(0, 0, SKY_RADIUS * 1.12), text="Antisolar point\n(shadow of your head)", height=14, box=False, opacity=0, color=color.white)
    _drawn.append(anti_lbl)

    observer = sphere(pos=vector(0, 0, 0), radius=SKY_RADIUS * 0.03, color=color.orange)
    _drawn.append(observer)

    for k in range(1, state["max_order"] + 1):
        opacity = _opacity_for(k)
        top_point = None
        top_side = None
        for wl in SKY_SWEEP_WAVELENGTHS_NM:
            side, angle_deg = sky_position(k, wl)
            angle_rad = radians(angle_deg)
            ring_r = SKY_RADIUS * sin(angle_rad)
            z = SKY_RADIUS * cos(angle_rad)
            if side == "solar":
                z = -z
            n_pts = 48
            pts = []
            for j in range(n_pts + 1):
                phi = 2 * pi * j / n_pts
                pts.append(vector(ring_r * cos(phi), ring_r * sin(phi), z))
            rgb = wavelength_to_rgb(wl)
            c = curve(pos=pts, color=vector(*rgb), radius=SKY_RADIUS * 0.004, opacity=opacity)
            _drawn.append(c)
            if wl == SKY_SWEEP_WAVELENGTHS_NM[len(SKY_SWEEP_WAVELENGTHS_NM) // 2]:
                top_point = vector(0, ring_r, z)
                top_side = side

        if state["labels"] and top_point is not None:
            name = order_name(k)
            vis = "visible" if naked_eye_visible(k) else "not naturally visible"
            pct = polarization_percent(k)
            label_text = "k=" + str(k) + " (" + name + ", " + top_side + " side)\n" + str(int(round(pct))) + "% polarized, " + vis
            lbl = label(pos=top_point, text=label_text, height=12, box=False, opacity=0, line=True, color=color.white, xoffset=10, yoffset=10)
            _drawn.append(lbl)

    scene.range = SKY_RADIUS * 1.4
    scene.center = vector(0, 0, 0)


def redraw():
    _clear()
    if state["mode"] == "droplet":
        draw_droplet_mode()
    else:
        draw_sky_mode()


def on_mode(r):
    state["mode"] = r.text
    # radio group callback receives the clicked radio button; determine
    # which one is checked
    for rb in mode_radios:
        if rb.checked:
            state["mode"] = rb.text
    redraw()


def on_order_slider(s):
    state["max_order"] = int(s.value)
    order_text.text = " max order: " + str(int(s.value))
    redraw()


def on_realistic(c):
    state["realistic"] = c.checked
    redraw()


def on_labels(c):
    state["labels"] = c.checked
    redraw()


def on_fan(c):
    state["show_fan"] = c.checked
    redraw()


scene.append_to_caption("\n")
scene.append_to_caption("Mode:  ")
mode_radios = [ radio(bind=on_mode, text="droplet", checked=True), ]
scene.append_to_caption("  ")
mode_radios.append(radio(bind=on_mode, text="sky"))
scene.append_to_caption("\n\n")

scene.append_to_caption("Highest rainbow order to show (1-13): ")
order_slider = slider(bind=on_order_slider, min=1, max=MAX_ORDER_LIMIT, value=state["max_order"], step=1, length=300)
order_text = wtext(text=" max order: " + str(state["max_order"]))
scene.append_to_caption("\n\n")

realistic_cb = checkbox(bind=on_realistic, text="Realistic brightness (dim to show why they're normally invisible)", checked=state["realistic"])
scene.append_to_caption("\n")
labels_cb = checkbox(bind=on_labels, text="Show labels", checked=state["labels"])
scene.append_to_caption("\n")
fan_cb = checkbox(bind=on_fan, text="Show ray fan (droplet mode: nearby rays bunching up = the caustic that makes a rainbow)", checked=state["show_fan"])
scene.append_to_caption("\n\n")

scene.append_to_caption("Orders 1-2 (primary/secondary) are the rainbow you actually see. Orders 3+ are real light paths computed from Descartes ray theory, but in nature they sit either right next to the sun (glare-buried) or are simply too faint against the sky -- this app renders them at full strength so you can actually see where they are.\n")

redraw()

while True:
    rate(30)
