import numpy as np


def half_spread(gamma, k, sigma, time_left, full=True):
    """Half of the optimal quoted spread in Avellaneda and Stoikov (2008).

    full=True:   delta_a + delta_b = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)
    full=False:  only the second term (the simplified version used in the first draft of this project)
    """
    if gamma == 0.0:                      # risk neutral limit
        total = 2.0 / k
    else:
        total = (2.0 / gamma) * np.log(1.0 + gamma / k)
        if full:
            total += gamma * sigma ** 2 * time_left
    return total / 2.0


class AvellanedaStoikovSimulator:
    """Single asset market maker that can be advanced one tick at a time.

    Live use:   sim.step() moves the market forward by one tick (returns False when the session is over).
    Batch use:  sim.simulate() runs a whole session and returns the arrays.
    gamma, sigma, k, A, skew and full_spread can be changed between ticks and apply from the next tick.

    skew=True   quotes around the reservation price (inventory skew, the Avellaneda-Stoikov strategy)
    skew=False  quotes symmetrically around the mid with the same spread (the benchmark in the paper)
    seed=None   a fresh random path on every reset, an int gives a reproducible path
    """

    def __init__(self, s0=100.0, T=1.0, dt=0.001, sigma=2.0, gamma=0.1, k=1.5, A=140.0,
                 seed=None, skew=True, full_spread=True):
        self.s0 = s0
        self.T = T
        self.dt = dt
        self.sigma = sigma
        self.gamma = gamma
        self.k = k
        self.A = A
        self.seed = seed
        self.skew = skew
        self.full_spread = full_spread
        self.steps = int(round(T / dt))
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        n = self.steps
        self.times = np.arange(n) * self.dt
        self.prices = np.full(n, np.nan)
        self.reservation_prices = np.full(n, np.nan)
        self.bid_quotes = np.full(n, np.nan)
        self.ask_quotes = np.full(n, np.nan)
        self.inventory = np.full(n, np.nan)
        self.cash = np.full(n, np.nan)
        self.edge_cum = np.full(n, np.nan)   # cumulative edge: sum over fills of the distance from mid
        self.fills = []   # one dict per fill: tick, time, side, qty, price, mid, edge, inventory (after the fill)

        self.t = 0        # index of the latest tick
        self.q = 0        # current inventory
        self.c = 0.0      # current cash

        self.prices[0] = self.s0
        self._post_quotes(0)
        self.inventory[0] = 0
        self.cash[0] = 0.0
        self.edge_cum[0] = 0.0

    @property
    def n(self):
        """Number of ticks that exist so far."""
        return self.t + 1

    @property
    def done(self):
        return self.t >= self.steps - 1

    # ------------------------------------------------------------ model maths
    def _post_quotes(self, i):
        s = self.prices[i]
        time_left = self.T - i * self.dt
        # Reservation price: the mid shifted against our inventory (no shift for the symmetric benchmark)
        r = s - self.q * self.gamma * (self.sigma ** 2) * time_left if self.skew else s
        half = half_spread(self.gamma, self.k, self.sigma, time_left, self.full_spread)
        self.reservation_prices[i] = r
        self.bid_quotes[i] = r - half
        self.ask_quotes[i] = r + half

    # ------------------------------------------------------------- live stepping
    def step(self):
        """Advance one tick. Returns True if a tick happened, False if the session is already over."""
        if self.done:
            return False

        i = self.t + 1
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.prices[i] = self.prices[i - 1] + self.sigma * dW
        s = self.prices[i]

        self._post_quotes(i)
        ask, bid = self.ask_quotes[i], self.bid_quotes[i]

        # Probability that someone hits / lifts our quote this tick: A * exp(-k * distance from mid) * dt
        prob_ask = min(1.0, self.A * np.exp(-self.k * (ask - s)) * self.dt)
        prob_bid = min(1.0, self.A * np.exp(-self.k * (s - bid)) * self.dt)

        edge_now = 0.0
        if self.rng.random() < prob_ask:
            self.q -= 1
            self.c += ask
            edge = float(ask - s)         # positive = we sold above the mid
            edge_now += edge
            self.fills.append({"tick": i, "time": self.times[i], "side": "SELL", "qty": 1,
                               "price": float(ask), "mid": float(s), "edge": edge, "inventory": int(self.q)})
        if self.rng.random() < prob_bid:
            self.q += 1
            self.c -= bid
            edge = float(s - bid)         # positive = we bought below the mid
            edge_now += edge
            self.fills.append({"tick": i, "time": self.times[i], "side": "BUY", "qty": 1,
                               "price": float(bid), "mid": float(s), "edge": edge, "inventory": int(self.q)})

        self.inventory[i] = self.q
        self.cash[i] = self.c
        self.edge_cum[i] = self.edge_cum[i - 1] + edge_now
        self.t = i
        return True

    # ------------------------------------------------------------------ views
    def pnl(self):
        n = self.n
        return self.cash[:n] + self.inventory[:n] * self.prices[:n]

    def max_drawdown(self):
        """Largest peak to trough fall in P&L so far."""
        pnl = self.pnl()
        return float(np.max(np.maximum.accumulate(pnl) - pnl))

    def spread_pnl(self):
        """P&L from quoting: the sum of edge over every fill."""
        return float(self.edge_cum[self.t])

    def inventory_pnl(self):
        """The rest of the P&L: mark to market gain or loss on the inventory we are holding."""
        return float(self.pnl()[-1] - self.spread_pnl())

    def realized_vol(self, window=50):
        """Rolling realized volatility per unit of session time, one value per tick (nan until enough data)."""
        n = self.n
        out = np.full(n, np.nan)
        r = np.diff(self.prices[:n])
        if len(r) >= window:
            rolling = np.lib.stride_tricks.sliding_window_view(r, window).std(axis=1) / np.sqrt(self.dt)
            out[window:] = rolling
        return out

    def markouts(self, horizons=(0, 5, 10, 25, 50, 100)):
        """Average markout per fill: how far the mid moved in our favour h ticks after the fill.

        Returns a list of (h, mean markout, number of fills with enough future data). At h = 0 the markout
        is the edge. In this simulation the mid is independent of our fills, so the markout stays close to the
        edge at every horizon. In a real market, fills are followed by adverse moves and the curve falls.
        """
        out = []
        for h in horizons:
            vals = []
            for f in self.fills:
                j = f["tick"] + h
                if j > self.t:
                    continue
                m = self.prices[j]
                vals.append((m - f["price"]) if f["side"] == "BUY" else (f["price"] - m))
            out.append((h, float(np.mean(vals)) if vals else float("nan"), len(vals)))
        return out

    def order_book(self, levels=8, tick=0.25):
        """Illustrative market depth around the mid, one dict per price level.

        The model has no real book, so the sizes are random (a new draw every tick gives the flicker of a live
        book). Only our own quotes, drawn separately, are real model output.
        """
        s = float(self.prices[self.t])
        rng = np.random.default_rng(self.t + 12345)
        base = round(s / tick) * tick
        rows = []
        for i in range(1, levels + 1):
            rows.append({"price": round(base + i * tick, 2), "side": "ask",
                         "size": int(rng.uniform(0.6, 1.4) * (10 + 6 * i))})
            rows.append({"price": round(base - i * tick, 2), "side": "bid",
                         "size": int(rng.uniform(0.6, 1.4) * (10 + 6 * i))})
        return rows

    # ------------------------------------------------------------ batch mode
    def simulate(self):
        """Run a full session in one go (same return signature as the original engine)."""
        self.reset()
        while self.step():
            pass
        return (self.prices, self.reservation_prices, self.bid_quotes,
                self.ask_quotes, self.inventory, self.cash)


def run_batch(n_sessions=400, s0=100.0, T=1.0, dt=0.001, sigma=2.0, gamma=0.1, k=1.5, A=140.0,
              full_spread=True, seed=1):
    """Monte Carlo of Avellaneda-Stoikov quoting against symmetric quoting on identical random paths.

    Both strategies see the same price path and the same fill draws in every session, so the difference
    between them is paired and not just two independent samples. Returns {"as": {...}, "sym": {...}} where each
    entry has per session arrays: pnl, final_q, mean_abs_q (time average of |inventory|) and fills.
    """
    steps = int(round(T / dt))
    rng = np.random.default_rng(seed)
    dW = rng.normal(0.0, np.sqrt(dt), size=(n_sessions, steps - 1))
    prices = np.concatenate([np.full((n_sessions, 1), s0), s0 + sigma * np.cumsum(dW, axis=1)], axis=1)
    u = rng.random((n_sessions, steps - 1, 2))

    out = {}
    for name, skew in (("as", True), ("sym", False)):
        q = np.zeros(n_sessions)
        c = np.zeros(n_sessions)
        fills = np.zeros(n_sessions)
        abs_q = np.zeros(n_sessions)
        for i in range(1, steps):
            s = prices[:, i]
            time_left = T - i * dt
            r = s - q * gamma * sigma ** 2 * time_left if skew else s
            h = half_spread(gamma, k, sigma, time_left, full_spread)
            ask, bid = r + h, r - h
            p_ask = np.minimum(1.0, A * np.exp(-k * (ask - s)) * dt)
            p_bid = np.minimum(1.0, A * np.exp(-k * (s - bid)) * dt)
            fa = u[:, i - 1, 0] < p_ask
            fb = u[:, i - 1, 1] < p_bid
            q = q - fa + fb
            c = c + ask * fa - bid * fb
            fills += fa + fb
            abs_q += np.abs(q)
        out[name] = {"pnl": c + q * prices[:, -1], "final_q": q,
                     "mean_abs_q": abs_q / (steps - 1), "fills": fills}
    return out


if __name__ == "__main__":
    sim = AvellanedaStoikovSimulator(seed=1)
    s, r, bids, asks, q, c = sim.simulate()
    final_pnl = c[-1] + (q[-1] * s[-1])
    print(f"Session: P&L {final_pnl:.2f}, final inventory {int(q[-1])}, fills {len(sim.fills)}")
    res = run_batch(n_sessions=300)
    for name in ("as", "sym"):
        p = res[name]["pnl"]
        print(f"{name:>3}: mean P&L {p.mean():7.2f}  std {p.std():6.2f}  std final inv {res[name]['final_q'].std():5.2f}")
