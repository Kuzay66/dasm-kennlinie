import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator

# -------------------------------------------------
# Seitenkonfiguration
# -------------------------------------------------
st.set_page_config(page_title="DASM Kennlinie", layout="wide")
st.title("Drehstromasynchronmaschine (DASM)")
st.subheader("Drehzahl–Drehmoment–Kennlinie")

# -------------------------------------------------
# Styles: kleine, kursiv wirkende Formelzeichen + Sub/Sup
# -------------------------------------------------
st.markdown(
    """
    <style>
      .fx { font-size: 0.90rem; font-style: italic; line-height: 1.1; margin-top: 0.25rem; }
      .fx sub { font-size: 0.72em; vertical-align: sub; }
      .fx sup { font-size: 0.72em; vertical-align: super; }
      .u { font-style: normal; }

      @media (max-width: 700px) {
        .fx { font-size: 0.85rem; }
      }
    </style>
    """,
    unsafe_allow_html=True
)

def labeled_selectbox(parent, label_html, options, key, index=0):
    c1, c2 = parent.columns([1.35, 2.0], gap="small")
    with c1:
        st.markdown(f'<div class="fx">{label_html}</div>', unsafe_allow_html=True)
    with c2:
        return st.selectbox("", options, index=index, key=key, label_visibility="collapsed")

def labeled_number(parent, label_html, value, step, key):
    c1, c2 = parent.columns([1.35, 2.0], gap="small")
    with c1:
        st.markdown(f'<div class="fx">{label_html}</div>', unsafe_allow_html=True)
    with c2:
        return st.number_input("", value=value, step=step, key=key, label_visibility="collapsed")

# -------------------------------------------------
# Sidebar – Netz
# -------------------------------------------------
with st.sidebar:
    st.header("Netz")
    f_net = labeled_selectbox(
        st,
        'Netzfrequenz&nbsp;f&nbsp;<span class="u">[Hz]</span>',
        [50, 60],
        key="FNET",
        index=0
    )

# -------------------------------------------------
# Sidebar – Motor
# -------------------------------------------------
with st.sidebar:
    st.header("Motor")

    ns_options = [3000, 1500, 1000, 750] if f_net == 50 else [3600, 1800, 1200, 900]
    n_s = labeled_selectbox(
        st,
        'Synchrondrehzahl&nbsp;n<sub>s</sub>&nbsp;<span class="u">[min<sup>−1</sup>]</span>',
        ns_options,
        key="NS",
        index=1
    )

    n_N_default = int(float(n_s) * 0.96)
    n_N = labeled_number(
        st,
        'Nenndrehzahl&nbsp;n<sub>N</sub>&nbsp;<span class="u">[min<sup>−1</sup>]</span>',
        value=n_N_default,
        step=10,
        key="NN"
    )

    if n_N >= n_s:
        st.sidebar.error("Hinweis: Es muss gelten n_N < n_s.")
        n_N = n_N_default

    M_A = labeled_number(
        st,
        'Anlaufmoment&nbsp;M<sub>A</sub>&nbsp;<span class="u">[Nm]</span>',
        value=60.0,
        step=1.0,
        key="MA"
    )
    M_S = labeled_number(
        st,
        'Sattelmoment&nbsp;M<sub>S</sub>&nbsp;<span class="u">[Nm]</span>',
        value=35.0,
        step=1.0,
        key="MS"
    )
    M_K = labeled_number(
        st,
        'Kippmoment&nbsp;M<sub>K</sub>&nbsp;<span class="u">[Nm]</span>',
        value=120.0,
        step=1.0,
        key="MK"
    )
    M_N = labeled_number(
        st,
        'Nennmoment&nbsp;M<sub>N</sub>&nbsp;<span class="u">[Nm]</span>',
        value=50.0,
        step=1.0,
        key="MN"
    )

# -------------------------------------------------
# Sidebar – Last (nur konstant)
# -------------------------------------------------
with st.sidebar:
    st.header("Last")

    _ = labeled_selectbox(
        st,
        'Lasttyp',
        ["konstant"],
        key="LOADTYPE",
        index=0
    )

    M_L0 = labeled_number(
        st,
        'Lastmoment&nbsp;M<sub>L</sub>&nbsp;<span class="u">[Nm]</span>',
        value=40.0,
        step=1.0,
        key="ML"
    )

# -------------------------------------------------
# Kennlinienberechnung
# -------------------------------------------------
n_grid = np.linspace(0.0, float(n_s), 800)

pts_n = np.array([0.0, 0.25 * n_s, 0.60 * n_s, float(n_N), float(n_s)], dtype=float)
pts_M = np.array([float(M_A), float(M_S), float(M_K), float(M_N), 0.0], dtype=float)

motor_curve = PchipInterpolator(pts_n, pts_M)
M_motor = motor_curve(n_grid)

M_load = np.full_like(n_grid, float(M_L0))

# -------------------------------------------------
# Arbeitspunkt (erster Schnittpunkt)
# -------------------------------------------------
diff = M_motor - M_load
cross_idxs = np.where(diff[:-1] * diff[1:] <= 0)[0]

if len(cross_idxs) > 0:
    i = int(cross_idxs[0])
    x0, x1 = n_grid[i], n_grid[i + 1]
    y0, y1 = diff[i], diff[i + 1]

    if abs(y1 - y0) < 1e-12:
        n_AP = x0
    else:
        n_AP = x0 - y0 * (x1 - x0) / (y1 - y0)

    M_AP = float(motor_curve(np.array([n_AP]))[0])
else:
    idx = int(np.argmin(np.abs(diff)))
    n_AP = float(n_grid[idx])
    M_AP = float(M_motor[idx])

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=n_grid, y=M_motor,
    name="Motor",
    line=dict(color="black", width=3)
))

fig.add_trace(go.Scatter(
    x=n_grid, y=M_load,
    name="Last",
    line=dict(color="red", width=3)
))

points = {
    "M<sub>A</sub>": (0.0, float(M_A)),
    "M<sub>S</sub>": (0.25 * n_s, float(M_S)),
    "M<sub>K</sub>": (0.60 * n_s, float(M_K)),
    "M<sub>N</sub>": (float(n_N), float(M_N)),
}

for label_html, (x0, y0) in points.items():
    fig.add_trace(go.Scatter(
        x=[x0], y=[y0],
        mode="markers+text",
        text=[label_html],
        textposition="top right",
        textfont=dict(size=12, color="black"),
        marker=dict(size=10, color="black"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[x0, x0], y=[0, y0],
        mode="lines",
        line=dict(dash="dot", width=1, color="rgba(0,0,0,0.45)"),
        showlegend=False
    ))

fig.add_trace(go.Scatter(
    x=[n_AP], y=[M_AP],
    mode="markers",
    marker=dict(size=13, color="green"),
    name="Arbeitspunkt"
))

fig.update_layout(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_title="Drehzahl n [min⁻¹]",
    yaxis_title="Drehmoment M [Nm]",
    # FIX: Achsen nicht zoombar/verschiebbar
    xaxis=dict(range=[0, float(n_s)], fixedrange=True),
    yaxis=dict(fixedrange=True),
    # FIX: kein Drag-Zoom / Pan
    dragmode=False,
    margin=dict(l=120, r=20, t=40, b=70),
    legend=dict(
        orientation="h",
        y=1.12,
        font=dict(color="black", size=12)
    ),
    font=dict(color="black"),
    height=520
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        # FIX: Modebar aus + Zoom/Doubleclick aus
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "responsive": True
    }
)
