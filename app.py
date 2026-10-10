import streamlit as st

import ui
from engine import AvellanedaStoikovSimulator, run_batch

st.set_page_config(page_title="Quoting engine", layout="wide", initial_sidebar_state="expanded")
st.markdown(ui.css(), unsafe_allow_html=True)

STRATEGIES = ["Avellaneda and Stoikov (inventory skew)", "Symmetric around the mid (benchmark)"]
VIEWS = ["Live", "Analytics", "Model"]


# ----------------------------------------------------------------- session state
def new_sim():
    s = st.session_state
    return AvellanedaStoikovSimulator(
        gamma=s.gamma, sigma=s.sigma, k=s.k, A=s.A,
        seed=int(s.seed) or None,                       # 0 means a new random path every session
        skew=(s.strategy == STRATEGIES[0]), full_spread=s.full_spread,
    )


def on_toggle():
    if st.session_state.sim.done:        # finished session: Play starts a fresh one
        st.session_state.sim = new_sim()
        st.session_state.running = True
    else:
        st.session_state.running = not st.session_state.running


def on_step():
    st.session_state.running = False
    st.session_state.sim.step()


def on_reset():
    st.session_state.sim = new_sim()
    st.session_state.running = True


if "sim" not in st.session_state:
    st.session_state.sim = AvellanedaStoikovSimulator()
    st.session_state.running = True

# ----------------------------------------------------------------------- sidebar
with st.sidebar:
    st.button("Pause" if st.session_state.running else "Play", on_click=on_toggle,
              type="primary", width="stretch")
    c1, c2 = st.columns(2)
    c1.button("Step", on_click=on_step, width="stretch", help="Advance one tick")
    c2.button("New session", on_click=on_reset, width="stretch")

    st.markdown('<div class="tm pt sp">Strategy</div>', unsafe_allow_html=True)
    strategy = st.radio("Strategy", STRATEGIES, key="strategy", label_visibility="collapsed")
    full_spread = st.checkbox("Full spread term, γσ²(T−t)", value=True, key="full_spread",
                              help="The Avellaneda and Stoikov spread is γσ²(T−t) + (2/γ)ln(1+γ/k). "
                                   "Untick it to use only the second term.")

    st.markdown('<div class="tm pt sp">Model</div>', unsafe_allow_html=True)
    gamma = st.slider("Risk aversion γ", 0.0, 0.5, 0.1, 0.01, key="gamma",
                      help="Higher values skew the quotes harder against inventory.")
    sigma = st.slider("Volatility σ", 0.5, 5.0, 2.0, 0.1, key="sigma")
    k = st.slider("Depth decay k", 0.5, 3.0, 1.5, 0.1, key="k",
                  help="How fast the fill probability falls as a quote moves away from the mid.")
    A = st.slider("Arrival rate A", 50.0, 300.0, 140.0, 10.0, key="A")

    st.markdown('<div class="tm pt sp">Risk monitor</div>', unsafe_allow_html=True)
    inv_limit = st.slider("Inventory limit", 3, 30, 10, 1,
                          help="Display only. The algorithm has no hard limit, this raises the breach flag.")

    st.markdown('<div class="tm pt sp">Feed</div>', unsafe_allow_html=True)
    speed = st.slider("Speed, ticks per second", 2, 60, 20)
    window = st.slider("Chart window, ticks", 100, 600, 300, 50)
    st.number_input("Seed", min_value=0, value=0, step=1, key="seed",
                    help="Applies to the next new session. 0 gives a new random path every time.")

# Push the sidebar values into the running simulation, so changes apply from the next tick
sim = st.session_state.sim
sim.gamma, sim.sigma, sim.k, sim.A = gamma, sigma, k, A
sim.skew = strategy == STRATEGIES[0]
sim.full_spread = full_spread

# Refresh timing: slow speeds tick once per refresh, fast speeds batch several ticks per refresh
if speed < 4:
    interval, ticks_per_refresh = 1.0 / speed, 1
else:
    interval, ticks_per_refresh = 0.5, int(round(speed * 0.5))

# ---------------------------------------------------------------------- header
c_brand, c_view = st.columns([4, 1.6], vertical_alignment="center")
c_brand.markdown(ui.brand_html(), unsafe_allow_html=True)
view = c_view.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")

PLOT_CFG = {"displayModeBar": False}


# ------------------------------------------------------------------------ views
def live():
    run_every = interval if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def live_view():
        sim = st.session_state.sim
        running = st.session_state.running
        if running:
            for _ in range(ticks_per_refresh):
                if not sim.step():
                    break

        status = "done" if sim.done else ("live" if running else "paused")
        st.markdown(ui.meta_html(sim, status, "Inventory skew" if sim.skew else "Symmetric benchmark"),
                    unsafe_allow_html=True)
        st.markdown(ui.kpi_html(sim, inv_limit), unsafe_allow_html=True)

        left, right = st.columns([2.4, 1], gap="medium")
        with left:
            st.markdown(ui.legend_html(), unsafe_allow_html=True)
            st.plotly_chart(ui.live_chart(sim, window, inv_limit), theme=None, width="stretch", config=PLOT_CFG)
        with right:
            # Fills from the latest refresh flash in the ladder and the trade list
            new_after = sim.t - ticks_per_refresh if running else sim.t
            hit_sides = {"ask" if f["side"] == "SELL" else "bid" for f in sim.fills if f["tick"] > new_after}
            st.markdown('<div class="tm pt">Price ladder</div>', unsafe_allow_html=True)
            st.markdown(ui.ladder_html(sim, hit_sides), unsafe_allow_html=True)
            st.markdown('<div class="tm note">You: our resting quote, size 1, with its edge against theo. '
                        'A dashed outline means the quote has crossed theo. Market sizes are simulated.</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="tm pt sp">Trade list</div>', unsafe_allow_html=True)
            st.markdown(ui.blotter_html(sim.fills, new_after), unsafe_allow_html=True)

        if sim.done and running:
            st.session_state.running = False
            st.rerun()      # full rerun so the auto refresh timer is switched off

    live_view()


@st.cache_data(show_spinner="Simulating sessions")
def monte_carlo(n, gamma, sigma, k, A, full):
    return run_batch(n_sessions=n, gamma=gamma, sigma=sigma, k=k, A=A, full_spread=full, seed=1)


def analytics():
    st.markdown('<div class="tm pt">Monte Carlo: inventory skew against symmetric quoting</div>', unsafe_allow_html=True)
    n = st.select_slider("Sessions", options=[100, 200, 400, 800, 1600], value=400)
    res = monte_carlo(n, gamma, sigma, k, A, full_spread)
    st.markdown(f'<div class="tm note">Model parameters come from the sidebar. {n} sessions of '
                f'{sim.steps - 1} ticks. In every session both strategies see the same price path and the same '
                f'fill draws, so the difference is paired.</div>', unsafe_allow_html=True)
    st.plotly_chart(ui.mc_figure(res), theme=None, width="stretch", config=PLOT_CFG)
    st.markdown(ui.mc_table_html(res), unsafe_allow_html=True)

    st.markdown('<div class="tm pt sp">Fill quality in the current live session</div>', unsafe_allow_html=True)
    fig = ui.markout_figure(sim)
    if fig is None:
        st.markdown('<div class="tm note">Not enough fills yet. Let the live session run a little longer.</div>',
                    unsafe_allow_html=True)
    else:
        st.plotly_chart(fig, theme=None, width="stretch", config=PLOT_CFG)
    st.markdown('<div class="tm note">Markout is how far the mid moved in our favour after a fill. Here the mid '
                'does not react to our fills, so the curve stays near the edge. Against real flow it would fall '
                'with the horizon (adverse selection).</div>', unsafe_allow_html=True)


def model():
    st.markdown('<div class="tm pt">Quoting rule</div>', unsafe_allow_html=True)
    st.latex(r"r = s - q\,\gamma\,\sigma^{2}\,(T-t)")
    st.latex(r"\delta^{a} + \delta^{b} = \gamma\,\sigma^{2}\,(T-t) + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{k}\right)")
    st.latex(r"\lambda(\delta) = A\,e^{-k\delta}")
    st.markdown('<div class="tm note">s is the mid, q our inventory, γ risk aversion, σ volatility, T−t the time left, '
                'and δ the distance of a quote from the mid. Quotes sit half the spread either side of r. '
                'Fills arrive with probability λ(δ)·dt per tick.</div>', unsafe_allow_html=True)

    rows = [
        ("Mid price", "Driftless random walk with constant σ.",
         "Volatility clusters, prices jump, and the mid drifts with order flow."),
        ("Fills", "Poisson arrivals with intensity A·exp(−kδ), independent of the next price move.",
         "Fills are toxic. Informed flow trades just before the mid moves against you, so the edge shown here is "
         "overstated. The markout chart on the Analytics view stays flat by construction."),
        ("Size and queue", "One unit on each side, always at the top, filled in full.",
         "Queue position, partial fills, size limits and cancel and replace latency."),
        ("Costs", "None.", "Maker and taker fees, rebates, tick size, financing and hedging costs."),
        ("Horizon", "Fixed T. Inventory risk fades toward the end of the session.",
         "Continuous trading, usually with a stationary version of the control problem."),
        ("Parameters", "A and k are known.", "A and k are estimated from trade data and move through the day."),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    st.markdown('<div class="tm pt sp">What the simulation assumes, and what a desk will push on</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tm blot txt"><table><tr><th>Component</th><th>This simulation</th>'
                f'<th>What a desk will push on</th></tr>{body}</table></div>', unsafe_allow_html=True)


{"Live": live, "Analytics": analytics, "Model": model}[view]()