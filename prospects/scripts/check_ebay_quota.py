"""Query eBay's Developer Analytics API for current rate-limit usage.

Endpoint: GET /developer/analytics/v1_beta/rate_limit/
Scope:    https://api.ebay.com/oauth/api_scope/commerce.developer.analytics.readonly

Reports calls remaining, period limits, and time until reset.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

# Load .env
_env = Path(__file__).resolve().parents[2] / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

import base64
from urllib.parse import urlencode

CLIENT_ID = os.environ.get("EBAY_CLIENT_ID")
CLIENT_SECRET = os.environ.get("EBAY_CLIENT_SECRET")
ENV = os.environ.get("EBAY_ENV", "production")

BASE = ("https://api.sandbox.ebay.com" if ENV.lower().startswith("sand")
        else "https://api.ebay.com")
TOKEN_URL = f"{BASE}/identity/v1/oauth2/token"
RATE_URL = f"{BASE}/developer/analytics/v1_beta/rate_limit/"

# The rate_limit endpoint requires a specific scope
SCOPE = "https://api.ebay.com/oauth/api_scope/commerce.developer.analytics.readonly"


def mint_token() -> str:
    if not (CLIENT_ID and CLIENT_SECRET):
        raise SystemExit("Missing EBAY_CLIENT_ID / EBAY_CLIENT_SECRET in .env")
    basic = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post(
        TOKEN_URL,
        headers={"Authorization": f"Basic {basic}",
                 "Content-Type": "application/x-www-form-urlencoded"},
        data=urlencode({"grant_type": "client_credentials", "scope": SCOPE}),
        timeout=30,
    )
    if not r.ok:
        raise SystemExit(f"Token mint failed ({r.status_code}): {r.text}")
    return r.json()["access_token"]


def main():
    token = mint_token()
    r = requests.get(
        RATE_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={"api_name": "Buy", "api_context": "Browse"},
        timeout=30,
    )
    if not r.ok:
        print(f"Rate-limit query failed ({r.status_code}): {r.text}")
        return
    data = r.json()
    rates = data.get("rateLimits", [])
    if not rates:
        print("No rate-limit data returned. Full response:")
        print(data)
        return
    print(f"{'API':<25} {'Resource':<25} {'Limit':>10} {'Remaining':>10} {'Reset':>20}")
    print("-" * 95)
    for entry in rates:
        api = entry.get("apiName", "")
        ctx = entry.get("apiContext", "")
        for res in entry.get("resources", []):
            res_name = res.get("name", "")
            for rate in res.get("rates", []):
                limit = rate.get("limit")
                remaining = rate.get("remaining")
                reset = rate.get("reset")
                window = rate.get("timeWindow")
                print(f"{api+'/'+ctx:<25} {res_name:<25} {limit:>10} {remaining:>10} "
                      f"{reset or '':<20}")
                if window:
                    print(f"  (window: {window}s)")


if __name__ == "__main__":
    main()
