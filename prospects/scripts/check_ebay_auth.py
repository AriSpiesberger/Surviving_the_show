"""Diagnostic: probe eBay credentials and the Browse API.

Reports:
  - which env vars are loaded from .env / shell
  - which token-mint flow will be used (app vs refresh-token)
  - result of a single token mint attempt
  - result of a single Browse API search ("Aaron Judge auto" smoke test)

Usage:
    python -m prospects.scripts.check_ebay_auth
"""
from __future__ import annotations

import os
from pathlib import Path

_env = Path(__file__).resolve().parents[2] / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from prospects.market.ebay_client import EbayBrowseClient


def _mask(s: str | None, keep: int = 6) -> str:
    if not s:
        return "<missing>"
    if len(s) <= keep * 2:
        return s[:2] + "…" + s[-2:]
    return f"{s[:keep]}…{s[-keep:]}  ({len(s)} chars)"


def main():
    env = os.environ.get("EBAY_ENV", "production")
    cid = os.environ.get("EBAY_CLIENT_ID")
    csec = os.environ.get("EBAY_CLIENT_SECRET")
    bear = os.environ.get("EBAY_BEARER_TOKEN")
    refr = os.environ.get("EBAY_USER_REFRESH_TOKEN")

    print("Env state:")
    print(f"  EBAY_ENV                : {env}")
    print(f"  EBAY_CLIENT_ID          : {_mask(cid)}")
    print(f"  EBAY_CLIENT_SECRET      : {_mask(csec)}")
    print(f"  EBAY_BEARER_TOKEN       : {_mask(bear)}")
    print(f"  EBAY_USER_REFRESH_TOKEN : {_mask(refr)}")

    if bear:
        flow = "preloaded bearer token"
    elif refr and cid and csec:
        flow = "user refresh-token grant (auto-mints user access tokens)"
    elif cid and csec:
        flow = "client_credentials grant (mints app tokens)"
    else:
        flow = "INSUFFICIENT — need either bearer token, refresh+app, or just app creds"
    print(f"\nResolved auth flow: {flow}")

    if "INSUFFICIENT" in flow:
        return

    client = EbayBrowseClient()
    print(f"\nToken URL : {client.token_url}")
    print(f"Browse URL: {client.browse_url}")

    print("\nAttempting token mint...")
    try:
        tok = client._ensure_token()
        print(f"  ✓ minted token ({_mask(tok, 4)})")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        print("  (likely cause: production keyset not yet enabled, or wrong creds)")
        return

    print("\nAttempting Browse API search...")
    try:
        results = client.search("Aaron Judge auto", limit=3, max_pages=1)
        print(f"  ✓ returned {len(results)} listing(s)")
        for r in results[:3]:
            price = f"${r.price_usd:.2f}" if r.price_usd is not None else "$?"
            print(f"    {price:>8}  {r.title[:80]}")
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
