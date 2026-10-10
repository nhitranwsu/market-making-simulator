"""Presentation layer for the quoting engine: theme, HTML panels and Plotly figures.

Everything here returns strings or figures, so app.py stays about state and layout only.
Colors follow one rule everywhere: blue is the bid side (buys, long), red is the ask side (sells, short),
amber is the model's own fair value (reservation price), green and red text only ever mean gain and loss
and always come with a sign.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C = dict(
    bg="#0d0d0d", panel="#1a1a19", panel2="#222221", line="#2c2c2a", axis="#383835",
    ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    bid="#3987e5", ask="#e66767", res="#c98500", vol="#9085e9",
    gain="#0ca30c", loss="#e66767", aqua="#199e70",
)
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

_CSS_BODY = """
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { padding: 1.2rem 1.6rem 2.5rem; max-width: 100%; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stMarkdown p { font-size: 13px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); }

.tm, .tm * { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; box-sizing: border-box; }
.tm { color: var(--ink); font-size: 14px; font-variant-numeric: tabular-nums; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }

/* top bar */
.brand { display: flex; align-items: baseline; gap: 14px; padding: 12px 16px; background: var(--panel);
       border: 1px solid var(--line); border-radius: 6px; }
.brand b { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }
.brand span { color: var(--ink2); font-size: 13px; }

/* status strip */
.meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 26px; padding: 10px 2px 8px; font-size: 13px; }
.meta .kv em { font-style: normal; color: var(--muted); margin-right: 7px; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 3px 11px 3px 9px; border-radius: 999px;
        border: 1px solid var(--axis); font-weight: 600; font-size: 12px; color: var(--ink); }
.pill i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.pill.live i { background: var(--loss); animation: pulse 1.3s infinite; }
.pill.done i { background: var(--gain); }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(230,103,103,.65); }
  70% { box-shadow: 0 0 0 7px rgba(230,103,103,0); }
  100% { box-shadow: 0 0 0 0 rgba(230,103,103,0); }
}
.prog { height: 2px; background: var(--line); margin-bottom: 10px; }
.prog i { display: block; height: 100%; background: var(--ink2); }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--line);
        border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-bottom: 12px; }
.kpi { background: var(--panel); padding: 11px 15px 12px; min-width: 0; }
.kpi .l { color: var(--muted); font-size: 12px; }
.kpi .v { font-size: 27px; font-weight: 600; line-height: 1.2; margin-top: 3px; white-space: nowrap; }
.kpi .v.two { font-size: 18px; line-height: 1.35; margin-top: 4px; }
.kpi .s { color: var(--ink2); font-size: 12px; line-height: 1.35; margin-top: 3px; }
.sq { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 8px; }
.sq.b { background: var(--bid); } .sq.a { background: var(--ask); }
.lim { height: 3px; background: var(--axis); border-radius: 2px; margin-top: 7px; overflow: hidden; }
.lim i { display: block; height: 100%; background: var(--ink2); }
.lim.breach i { background: var(--loss); }

/* section titles */
.pt { color: var(--ink2); font-size: 13px; font-weight: 500; margin: 0 0 6px; }
.pt.sp { margin-top: 14px; }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; line-height: 1.45; }

/* price ladder */
.lad { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; font-size: 14px; }
.lad .hd, .lad .lr { display: grid; grid-template-columns: 1fr 76px 1fr; align-items: center; height: 26px; padding: 0 10px; }
.lad .hd { height: 28px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--line); }
.lad .l { text-align: right; padding-right: 10px; position: relative; height: 100%; display: flex; align-items: center; justify-content: flex-end; }
.lad .r { text-align: left; padding-left: 10px; position: relative; height: 100%; display: flex; align-items: center; }
.lad .c { text-align: center; color: var(--ink2); }
.lad .dp { position: absolute; top: 3px; bottom: 3px; opacity: .38; }
.lad .l .dp { right: 0; background: var(--bid); border-radius: 2px 0 0 2px; }
.lad .r .dp { left: 0; background: var(--ask); border-radius: 0 2px 2px 0; }
.lad .t { position: relative; }
.lad .lr.theo { background: rgba(240,239,236,.08); border-top: 1px solid var(--axis); border-bottom: 1px solid var(--axis);
                font-weight: 600; color: var(--ink); }
.lad .lr.theo .c { color: var(--ink); }
.lad .lr.theo .l, .lad .lr.res .l { color: var(--muted); font-weight: 400; font-size: 12px; }
.lad .lr.res .c, .lad .lr.res .r { color: var(--res); }
.lad .lr.res { font-size: 13px; }
.you { display: inline-flex; gap: 8px; align-items: center; padding: 1px 8px; border: 1.5px solid var(--ink);
       border-radius: 4px; background: rgba(240,239,236,.10); font-weight: 600; font-size: 13px; white-space: nowrap; }
.you.cross { border-style: dashed; border-color: var(--res); }
.you.hit { animation: hit 1.1s ease-out; }
@keyframes hit { 0% { background: rgba(240,239,236,.85); color: #0d0d0d; } 100% { background: rgba(240,239,236,.10); } }

/* blotter and stats tables */
.blot { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.blot table { width: 100%; border-collapse: collapse; font-size: 13px; }
.blot th { color: var(--muted); font-weight: 500; font-size: 12px; text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
.blot td { text-align: right; padding: 0 9px; height: 25px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.blot tr:last-child td { border-bottom: 0; }
.blot th:first-child, .blot td:first-child { text-align: left; }
.blot td.sd { text-align: left; }
.blot .g { display: inline-block; width: 14px; font-size: 10px; }
.blot .g.b { color: var(--bid); } .blot .g.a { color: var(--ask); }
.blot .empty { text-align: center; color: var(--muted); height: 60px; }
.blot tr.new td { animation: flashrow 1.1s ease-out; }
@keyframes flashrow { 0% { background: rgba(240,239,236,.22); } 100% { background: transparent; } }
.blot.stats td { height: 34px; font-size: 14px; }
.blot.stats td:first-child { color: var(--ink2); }
.blot.txt th, .blot.txt td { text-align: left; vertical-align: top; height: auto; padding: 10px 14px; line-height: 1.5; font-size: 13px; white-space: normal; }
.blot.txt td:first-child { color: var(--ink); font-weight: 500; white-space: nowrap; }
.blot.txt td { color: var(--ink2); }
"""


def css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in C.items()) + "}"
    font = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
    return "<style>" + font + root + _CSS_BODY + "</style>"


def _sg(x):
    return "gain" if x >= 0 else "loss"


def _rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


# --------------------------------------------------------------------- HTML panels
def brand_html():
    return ('<div class="tm brand"><b>Quoting engine</b>'
            '<span>Avellaneda and Stoikov inventory control, simulated order flow</span></div>')


def meta_html(sim, status, strategy_label):
    text = {"live": "Live", "paused": "Paused", "done": "Complete"}[status]
    pct = 100.0 * sim.t / (sim.steps - 1)
    seed = "random" if sim.seed is None else sim.seed
    return (f'<div class="tm meta"><span class="pill {status}"><i></i>{text}</span>'
            f'<span class="kv"><em>Session time</em>{sim.times[sim.t]:.3f} of {sim.T:.3f}</span>'
            f'<span class="kv"><em>Tick</em>{sim.t} of {sim.steps - 1}</span>'
            f'<span class="kv"><em>Strategy</em>{strategy_label}</span>'
            f'<span class="kv"><em>Seed</em>{seed}</span></div>'
            f'<div class="tm prog"><i style="width:{pct:.1f}%"></i></div>')


def _tile(label, value, sub="", vcls="", extra=""):
    return (f'<div class="kpi"><div class="l">{label}</div><div class="v {vcls}">{value}</div>'
            f'<div class="s">{sub}</div>{extra}</div>')


def kpi_html(sim, inv_limit):
    t = sim.t
    prev = max(t - 1, 0)
    mid = float(sim.prices[t])
    bid, ask, res = float(sim.bid_quotes[t]), float(sim.ask_quotes[t]), float(sim.reservation_prices[t])
    pnl = sim.pnl()
    total = float(pnl[-1])
    spread_pnl, inv_pnl = sim.spread_pnl(), sim.inventory_pnl()
    buys = sum(1 for f in sim.fills if f["side"] == "BUY")
    sells = len(sim.fills) - buys
    vol = sim.realized_vol()[-1]
    q = sim.q
    breach = abs(q) > inv_limit
    dm = mid - float(sim.prices[prev])
    avg_edge = spread_pnl / len(sim.fills) if sim.fills else None
    peak = float(np.max(pnl))

    tiles = [
        _tile("P&L", f'<span class="{_sg(total)}">{total:+.2f}</span>',
              f'edge {spread_pnl:+.2f}, inventory {inv_pnl:+.2f}'),
        _tile("Position", f'{q:+d}' if q else "0",
              '<span class="loss">Over limit of ±%d</span>' % inv_limit if breach else f'limit ±{inv_limit}',
              extra=f'<div class="lim {"breach" if breach else ""}"><i style="width:{min(100, abs(q) / inv_limit * 100):.0f}%"></i></div>'),
        _tile("Mid", f'{mid:.2f}',
              f'<span class="{_sg(dm)}">{"▲" if dm >= 0 else "▼"} {abs(dm):.2f}</span> last tick'),
        _tile("Bid and ask",
              f'<div><i class="sq b"></i>{bid:.2f}</div><div><i class="sq a"></i>{ask:.2f}</div>',
              f'spread {ask - bid:.2f}, skew {res - mid:+.2f}', vcls="two"),
        _tile("Fills", f'{len(sim.fills)}', f'{buys} buys, {sells} sells'),
        _tile("Edge per fill", f'{avg_edge:+.2f}' if avg_edge is not None else "n/a", "average distance from mid"),
        _tile("Realized vol", "n/a" if np.isnan(vol) else f'{vol:.2f}', f'input σ {sim.sigma:.2f}'),
        _tile("Max drawdown", f'{sim.max_drawdown():.2f}', f'peak P&L {peak:+.2f}'),
    ]
    return '<div class="tm kpis">' + "".join(tiles) + "</div>"


def ladder_html(sim, hit_sides, levels=8, tick=0.25):
    """Price ladder: market depth (illustrative) with theo, the reservation price and our two quotes."""
    t = sim.t
    mid, res = float(sim.prices[t]), float(sim.reservation_prices[t])
    bid, ask = float(sim.bid_quotes[t]), float(sim.ask_quotes[t])
    mkt = sim.order_book(levels, tick)
    max_size = max(r["size"] for r in mkt)
    items = [dict(kind="mkt", **r) for r in mkt]
    items += [dict(kind="you", side="ask", price=ask), dict(kind="you", side="bid", price=bid),
              dict(kind="theo", price=mid), dict(kind="res", price=res)]
    items.sort(key=lambda r: -r["price"])

    out = ['<div class="tm lad"><div class="hd"><span class="l">Bid size</span><span class="c">Price</span>'
           '<span class="r">Ask size</span></div>']
    for r in items:
        p = r["price"]
        if r["kind"] == "theo":
            out.append(f'<div class="lr theo"><span class="l">Theo</span><span class="c">{p:.2f}</span><span class="r"></span></div>')
        elif r["kind"] == "res":
            out.append(f'<div class="lr res"><span class="l">Reservation</span><span class="c">{p:.2f}</span>'
                       f'<span class="r">skew {res - mid:+.2f}</span></div>')
        elif r["kind"] == "you":
            edge = (ask - mid) if r["side"] == "ask" else (mid - bid)
            cls = "you" + (" hit" if r["side"] in hit_sides else "") + (" cross" if edge < 0 else "")
            cell = f'<span class="{cls}">You <span>{edge:+.2f}</span></span>'
            if r["side"] == "ask":
                out.append(f'<div class="lr"><span class="l"></span><span class="c">{p:.2f}</span><span class="r">{cell}</span></div>')
            else:
                out.append(f'<div class="lr"><span class="l">{cell}</span><span class="c">{p:.2f}</span><span class="r"></span></div>')
        else:
            w = max(6, int(100 * r["size"] / max_size))
            bar = f'<i class="dp" style="width:{w}%"></i>'
            if r["side"] == "ask":
                out.append(f'<div class="lr"><span class="l"></span><span class="c">{p:.2f}</span>'
                           f'<span class="r">{bar}<span class="t">{r["size"]}</span></span></div>')
            else:
                out.append(f'<div class="lr"><span class="l">{bar}<span class="t">{r["size"]}</span></span>'
                           f'<span class="c">{p:.2f}</span><span class="r"></span></div>')
    out.append("</div>")
    return "".join(out)


def blotter_html(fills, new_after_tick, n=10):
    head = ("<tr><th>Tick</th><th>Side</th><th>Qty</th><th>Price</th><th>Theo</th><th>Edge</th><th>Position</th></tr>")
    if not fills:
        body = '<tr><td colspan="7" class="empty">Waiting for the first fill</td></tr>'
    else:
        body = ""
        for f in reversed(fills[-n:]):
            buy = f["side"] == "BUY"
            glyph = f'<span class="g {"b" if buy else "a"}">{"▲" if buy else "▼"}</span>'
            body += (f'<tr class="{"new" if f["tick"] > new_after_tick else ""}"><td>{f["tick"]}</td>'
                     f'<td class="sd">{glyph}{"Buy" if buy else "Sell"}</td><td>{f["qty"]}</td>'
                     f'<td>{f["price"]:.2f}</td><td>{f["mid"]:.2f}</td>'
                     f'<td class="{_sg(f["edge"])}">{f["edge"]:+.2f}</td><td>{f["inventory"]:+d}</td></tr>')
    return f'<div class="tm blot"><table>{head}{body}</table></div>'


# ------------------------------------------------------------------------- figures
def _style(fig, height, margin):
    fig.update_layout(
        height=height, paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
        font=dict(family=FONT, size=12, color=C["ink2"]), margin=margin,
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["axis"], font=dict(family=FONT, size=12, color=C["ink"])),
        legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom",
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C["axis"], tickfont=dict(size=12, color=C["muted"]))
    fig.update_yaxes(gridcolor=C["line"], gridwidth=1, zeroline=False, linecolor=C["axis"],
                     tickfont=dict(size=12, color=C["muted"]))


def _panel_label(fig, text, yaxis):
    fig.add_annotation(xref="paper", yref=f"{yaxis} domain", x=0, y=1.0, text=text, showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=13, color=C["ink2"]))


def live_chart(sim, window, inv_limit):
    n = sim.n
    lo = max(0, n - window)
    sl = slice(lo, n)
    t = sim.times[sl]
    mid_now = float(sim.prices[sim.t])

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                        row_heights=[0.5, 0.15, 0.2, 0.15])

    # 1. quotes and fills
    fig.add_trace(go.Scatter(x=t, y=sim.prices[sl], mode="lines", name="Mid",
                             line=dict(color=C["ink"], width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sim.reservation_prices[sl], mode="lines", name="Reservation",
                             line=dict(color=C["res"], width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sim.bid_quotes[sl], mode="lines", name="Bid quote",
                             line=dict(color=C["bid"], width=1.25)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=sim.ask_quotes[sl], mode="lines", name="Ask quote",
                             line=dict(color=C["ask"], width=1.25),
                             fill="tonexty", fillcolor="rgba(240,239,236,0.05)"), row=1, col=1)
    buys = [f for f in sim.fills if f["tick"] >= lo and f["side"] == "BUY"]
    sells = [f for f in sim.fills if f["tick"] >= lo and f["side"] == "SELL"]
    ring = dict(width=2, color=C["panel"])
    fig.add_trace(go.Scatter(x=[f["time"] for f in buys], y=[f["price"] for f in buys], mode="markers",
                             name="Buy fill", marker=dict(symbol="triangle-up", size=12, color=C["bid"], line=ring)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[f["time"] for f in sells], y=[f["price"] for f in sells], mode="markers",
                             name="Sell fill", marker=dict(symbol="triangle-down", size=12, color=C["ask"], line=ring)),
                  row=1, col=1)
    fig.add_annotation(xref="paper", yref="y", x=1.0, y=mid_now, text=f"{mid_now:.2f}", showarrow=False,
                       xanchor="left", xshift=6, bgcolor=C["ink"], borderpad=3,
                       font=dict(size=12, color=C["bg"], family=FONT))

    # 2. position, shaded blue when long and red when short
    inv = sim.inventory[sl]
    fig.add_trace(go.Scatter(x=t, y=np.where(inv > 0, inv, 0), mode="lines", line=dict(width=0, shape="hv"),
                             fill="tozeroy", fillcolor=_rgba(C["bid"], 0.30), hoverinfo="skip", showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.where(inv < 0, inv, 0), mode="lines", line=dict(width=0, shape="hv"),
                             fill="tozeroy", fillcolor=_rgba(C["ask"], 0.30), hoverinfo="skip", showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=inv, mode="lines", name="Position",
                             line=dict(color=C["ink"], width=1.5, shape="hv"), showlegend=False), row=2, col=1)
    for y in (inv_limit, -inv_limit):
        fig.add_hline(y=y, line_dash="dot", line_color=C["muted"], line_width=1, row=2, col=1)
    span = max(inv_limit * 1.25, float(np.nanmax(np.abs(inv))) * 1.15 + 1)
    fig.update_yaxes(range=[-span, span], zeroline=True, zerolinecolor=C["axis"], tickformat="d", row=2, col=1)

    # 3. P&L, with cumulative edge so the gap between the two lines is the inventory P&L
    pnl = sim.pnl()[sl]
    fig.add_trace(go.Scatter(x=t, y=np.where(pnl >= 0, pnl, 0), mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_rgba(C["gain"], 0.22), hoverinfo="skip", showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.where(pnl < 0, pnl, 0), mode="lines", line=dict(width=0),
                             fill="tozeroy", fillcolor=_rgba(C["loss"], 0.22), hoverinfo="skip", showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=sim.edge_cum[sl], mode="lines", name="Cumulative edge",
                             line=dict(color=C["muted"], width=1.5, dash="dash"), showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=pnl, mode="lines", name="P&L", line=dict(color=C["ink"], width=2), showlegend=False),
                  row=3, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor=C["axis"], row=3, col=1)

    # 4. realized volatility against the sigma the quotes assume
    fig.add_trace(go.Scatter(x=t, y=sim.realized_vol()[sl], mode="lines", name="Realized vol",
                             line=dict(color=C["vol"], width=2), showlegend=False), row=4, col=1)
    fig.add_hline(y=sim.sigma, line_dash="dash", line_color=C["muted"], line_width=1, row=4, col=1,
                  annotation_text="input σ", annotation_position="top right",
                  annotation_font=dict(size=11, color=C["muted"]))

    _style(fig, 920, dict(l=56, r=74, t=36, b=48))
    fig.update_xaxes(range=[lo * sim.dt, (lo + window) * sim.dt], tickformat=".3f",
                     showspikes=True, spikemode="across", spikesnap="cursor", spikecolor=C["axis"], spikethickness=1)
    fig.update_xaxes(title_text="Session time", title_font=dict(size=12, color=C["muted"]), row=4, col=1)
    fig.update_yaxes(tickformat=".2f", row=1, col=1)
    fig.update_yaxes(tickformat=".1f", row=3, col=1)
    fig.update_yaxes(tickformat=".1f", row=4, col=1)
    fig.update_layout(hovermode="x unified")
    for label, ax in (("Quotes and fills", "y"), ("Position", "y2"),
                      ("P&L (solid) and cumulative edge (dashed)", "y3"), ("Realized volatility, 50 tick window", "y4")):
        _panel_label(fig, label, ax)
    return fig


def markout_figure(sim):
    rows = [(h, m, n) for h, m, n in sim.markouts() if n >= 3 and not np.isnan(m)]
    if len(rows) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r[0] for r in rows], y=[r[1] for r in rows], mode="lines+markers",
                             name="Average markout", line=dict(color=C["aqua"], width=2),
                             marker=dict(size=9, color=C["aqua"], line=dict(width=2, color=C["panel"])),
                             customdata=[r[2] for r in rows],
                             hovertemplate="%{x} ticks after the fill: %{y:+.3f} (%{customdata} fills)<extra></extra>"))
    fig.add_hline(y=0, line_color=C["axis"], line_width=1)
    _style(fig, 300, dict(l=56, r=20, t=34, b=48))
    fig.update_xaxes(title_text="Ticks after the fill", title_font=dict(size=12, color=C["muted"]))
    fig.update_yaxes(tickformat="+.2f")
    _panel_label(fig, "Average markout per fill (price units)", "y")
    fig.update_layout(showlegend=False)
    return fig


def mc_figure(res):
    names = {"as": "Avellaneda and Stoikov", "sym": "Symmetric quoting"}
    colors = {"as": C["aqua"], "sym": C["vol"]}
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.07)
    for col, key in ((1, "pnl"), (2, "final_q")):
        allv = np.concatenate([res["as"][key], res["sym"][key]])
        if key == "final_q":
            bins = dict(start=np.floor(allv.min()) - 0.5, end=np.ceil(allv.max()) + 0.5, size=1)
        else:
            lo, hi = float(allv.min()), float(allv.max())
            bins = dict(start=lo, end=hi, size=max((hi - lo) / 32, 1e-9))
        fig.add_trace(go.Histogram(x=res["as"][key], name=names["as"], legendgroup="as", showlegend=(col == 1),
                                   histnorm="percent", xbins=bins, opacity=0.85,
                                   marker=dict(color=colors["as"], line=dict(color=C["panel"], width=1))),
                      row=1, col=col)
        fig.add_trace(go.Histogram(x=res["sym"][key], name=names["sym"], legendgroup="sym", showlegend=(col == 1),
                                   histnorm="percent", xbins=bins,
                                   marker=dict(color="rgba(0,0,0,0)", line=dict(color=colors["sym"], width=1.75))),
                      row=1, col=col)
    _style(fig, 360, dict(l=56, r=20, t=64, b=48))
    fig.update_layout(barmode="overlay", hovermode="closest", legend=dict(y=1.13))
    fig.update_yaxes(title_text="Share of sessions (%)", title_font=dict(size=12, color=C["muted"]), col=1)
    fig.update_xaxes(title_text="Final P&L", title_font=dict(size=12, color=C["muted"]), col=1)
    fig.update_xaxes(title_text="Final inventory", title_font=dict(size=12, color=C["muted"]), col=2)
    fig.add_annotation(xref="x domain", yref="y domain", x=0, y=1.0, text="Distribution of final P&L", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=13, color=C["ink2"]))
    fig.add_annotation(xref="x2 domain", yref="y2 domain", x=0, y=1.0, text="Distribution of final inventory",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=13, color=C["ink2"]))
    return fig


def mc_table_html(res):
    a, s = res["as"], res["sym"]

    def stat(r):
        p = r["pnl"]
        return dict(mean=p.mean(), std=p.std(ddof=1), ratio=p.mean() / p.std(ddof=1), p5=np.percentile(p, 5),
                    fin=r["final_q"].std(ddof=1), absq=r["mean_abs_q"].mean(), fills=r["fills"].mean())

    A, S = stat(a), stat(s)
    diff = a["pnl"] - s["pnl"]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    rows = [
        ("Mean P&L per session", f'{A["mean"]:.2f}', f'{S["mean"]:.2f}', f'{A["mean"] - S["mean"]:+.2f}'),
        ("Standard deviation of P&L", f'{A["std"]:.2f}', f'{S["std"]:.2f}', f'{A["std"] / S["std"] - 1:+.0%}'),
        ("Mean P&L divided by its standard deviation", f'{A["ratio"]:.2f}', f'{S["ratio"]:.2f}', f'{A["ratio"] - S["ratio"]:+.2f}'),
        ("5th percentile of P&L", f'{A["p5"]:.2f}', f'{S["p5"]:.2f}', f'{A["p5"] - S["p5"]:+.2f}'),
        ("Standard deviation of final inventory", f'{A["fin"]:.2f}', f'{S["fin"]:.2f}', f'{A["fin"] / S["fin"] - 1:+.0%}'),
        ("Average absolute inventory over the session", f'{A["absq"]:.2f}', f'{S["absq"]:.2f}', f'{A["absq"] / S["absq"] - 1:+.0%}'),
        ("Fills per session", f'{A["fills"]:.0f}', f'{S["fills"]:.0f}', f'{A["fills"] / S["fills"] - 1:+.0%}'),
        ("Paired mean P&L difference, 95% interval", "", "", f'{diff.mean():+.2f} ± {1.96 * se:.2f}'),
    ]
    head = "<tr><th>Metric</th><th>Avellaneda and Stoikov</th><th>Symmetric quoting</th><th>Difference</th></tr>"
    body = "".join(f"<tr><td>{m}</td><td>{x}</td><td>{y}</td><td>{d}</td></tr>" for m, x, y, d in rows)
    return f'<div class="tm blot stats"><table>{head}{body}</table></div>'
