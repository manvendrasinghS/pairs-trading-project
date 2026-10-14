"""Block 0: universe, download, cleaning.

Builds data/prices.csv (adjusted closes, days x tickers) and the two views
derived from it: log prices for the relationship tests, simple returns for P&L.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252

# Liquid names grouped by shared economic driver, with cross-sector pairs left
# in so the screen gets a fair chance to produce false positives.
UNIVERSE: dict[str, str] = {
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "JPM": "banks", "BAC": "banks", "WFC": "banks", "C": "banks",
    "V": "payments", "MA": "payments",
    "MSFT": "tech", "AAPL": "tech", "ORCL": "tech", "IBM": "tech",
    "HD": "retail", "LOW": "retail", "TGT": "retail", "WMT": "retail",
    "KO": "beverages", "PEP": "beverages",
    "VZ": "telecom", "T": "telecom",
    "XLE": "etf_sector", "XLF": "etf_sector", "XLK": "etf_sector",
}

# SPY/IVV/VOO track the same index, GLD/IAU the same metal. If the screen misses
# these, the screen is broken. (It does miss them: see diagnose_pair.)
POSITIVE_CONTROLS: dict[str, str] = {
    "SPY": "control_sp500", "IVV": "control_sp500", "VOO": "control_sp500",
    "GLD": "control_gold", "IAU": "control_gold",
}


def all_tickers(include_controls: bool = True) -> list[str]:
    """Sorted ticker list, so pair order and tie-breaks are deterministic."""
    tickers = set(UNIVERSE)
    if include_controls:
        tickers |= set(POSITIVE_CONTROLS)
    return sorted(tickers)


def sector_of(ticker: str) -> str:
    return UNIVERSE.get(ticker) or POSITIVE_CONTROLS.get(ticker, "unknown")


def candidate_pairs(tickers: list[str]) -> list[tuple[str, str]]:
    """All n(n-1)/2 unordered pairs. Kept unfiltered: the width of the search is
    a number the study has to report."""
    return list(itertools.combinations(tickers, 2))


def ensure_ssl_trust(cache_path: str = "data/ca_bundle.pem") -> str | None:
    """Export the macOS keychains to a PEM and point the SSL env vars at it.

    Needed behind a TLS-inspecting proxy: yfinance validates through curl_cffi,
    which ignores the keychain, so a corporate root CA shows up as "possibly
    delisted". Fails soft (returns None) and is a no-op off macOS.
    """
    if sys.platform != "darwin":
        return None
    if os.environ.get("SSL_CERT_FILE"):
        return os.environ["SSL_CERT_FILE"]

    bundle = pathlib.Path(cache_path)
    try:
        if not bundle.exists() or bundle.stat().st_size == 0:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            pem_chunks = []
            for keychain in (
                "/System/Library/Keychains/SystemRootCertificates.keychain",  # public roots
                "/Library/Keychains/System.keychain",                         # corporate roots
            ):
                out = subprocess.run(
                    ["security", "find-certificate", "-a", "-p", keychain],
                    capture_output=True, text=True, timeout=30,
                )
                if out.returncode == 0 and "BEGIN CERTIFICATE" in out.stdout:
                    pem_chunks.append(out.stdout)
            if not pem_chunks:
                return None
            bundle.write_text("\n".join(pem_chunks))

        # requests, urllib and curl_cffi respectively.
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[var] = str(bundle.resolve())
        return str(bundle.resolve())
    except Exception:
        return None


def download_prices(
    tickers: list[str],
    start: str = "2012-01-01",
    end: str | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Daily adjusted closes, date index and one column per ticker.

    auto_adjust=True matters: un-adjusted prices jump on splits and dividends,
    and the strategy reads that as the spread blowing out.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=False,      # easier on Yahoo's rate limiter
            )
            if not raw.empty:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(2 * (attempt + 1))
    else:
        raise ValueError(f"Download failed after {retries} attempts: {last_error}")

    if raw.empty:
        raise ValueError("Yahoo returned no data. Check tickers/dates/connection.")

    # Many tickers -> MultiIndex columns; one ticker -> flat columns.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def clean_prices(prices: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    """Drop thin tickers, then incomplete days.

    Dropping rather than forward-filling: a filled value invents a flat day on
    one leg while the other moved, which fabricates a spread move. Tickers go
    first, else one short history deletes years of good data.
    """
    coverage = prices.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print(f"  [clean] dropped {len(dropped)} thin ticker(s): {dropped}")

    cleaned = prices[keep].dropna(how="any")
    return cleaned


def to_log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Log prices: scale-free, and make beta an elasticity."""
    if (prices <= 0).any().any():
        raise ValueError("Non-positive price found; cannot take a log.")
    return np.log(prices)


def to_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple returns, the only input to P&L.

    r_portfolio = w_A*r_A + w_B*r_B holds for these and is false for log
    returns, so the two views stay in separate functions.
    """
    return prices.pct_change().dropna(how="all")


def build_dataset(
    start: str = "2012-01-01",
    end: str | None = None,
    include_controls: bool = True,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Download and clean the master price matrix.

    Starting in 2012 keeps 2008-09 out of the formation windows, where it would
    let the screen pick pairs that co-moved only because everything did.
    """
    bundle = ensure_ssl_trust()
    if bundle:
        print(f"Using CA bundle for HTTPS: {bundle}")

    tickers = all_tickers(include_controls=include_controls)
    print(f"Downloading {len(tickers)} tickers from {start} ...")
    prices = download_prices(tickers, start=start, end=end)
    prices = clean_prices(prices, min_coverage=min_coverage)
    return prices


def sanity_checks(prices: pd.DataFrame) -> None:
    """Checks for the data bugs that would otherwise pass silently."""
    print("=" * 68)
    print("SANITY CHECKS")
    print("=" * 68)
    print(f"Tickers: {prices.shape[1]}   Days: {prices.shape[0]}")
    print(f"Date range: {prices.index.min().date()} -> {prices.index.max().date()}")
    n_pairs = len(candidate_pairs(list(prices.columns)))
    print(f"Candidate pairs to be screened: {n_pairs}")

    assert (prices > 0).all().all(), "Non-positive price found — data bug!"
    assert not prices.isna().any().any(), "NaNs survived cleaning — alignment bug!"
    assert prices.index.is_unique, "Duplicate dates in index — data bug!"

    # A >50% day in a mega-cap is usually an un-adjusted split.
    rets = to_simple_returns(prices)
    extreme = rets.abs() > 0.50
    if extreme.any().any():
        hits = [(d.date(), t) for d, t in zip(*np.where(extreme))][:5]
        print(f"  WARNING: suspicious >50% daily moves (possible split issue): {hits}")
    else:
        print("  No >50% daily moves — split/dividend adjustment looks sane.")

    # Same-underlying funds must be ~perfectly return-correlated; if not, the
    # columns or dates are mis-aligned somewhere.
    for a, b in [("SPY", "IVV"), ("GLD", "IAU")]:
        if a in rets.columns and b in rets.columns:
            corr = rets[a].corr(rets[b])
            flag = "OK" if corr > 0.97 else "SUSPICIOUS"
            print(f"  Control corr({a},{b}) = {corr:.4f}  <- {flag} (expect > 0.97)")

    print("=" * 68)


if __name__ == "__main__":
    px = build_dataset()
    sanity_checks(px)

    out_path = "data/prices.csv"
    px.to_csv(out_path)
    print(f"\nSaved price matrix -> {out_path}  ({px.shape[0]} days x {px.shape[1]} tickers)")
