"""
eBay Browse API client.

Two ways to authenticate (set env vars or pass at construction):

  1. EBAY_BEARER_TOKEN — paste an already-minted token (Application OR User).
     Easiest for development. Token expires every ~2 hours and you'll need
     to refresh manually.

  2. EBAY_CLIENT_ID + EBAY_CLIENT_SECRET — app credentials for the OAuth2
     client_credentials grant. Client mints Application tokens on demand
     and refreshes automatically. Recommended for unattended runs.

  3. EBAY_USER_REFRESH_TOKEN + EBAY_CLIENT_ID + EBAY_CLIENT_SECRET —
     a long-lived user refresh token (from eBay Sign-In flow) plus app
     creds. Client trades refresh token for fresh User access tokens.

Read-only Browse API works with any of the three.

Endpoints:
  - POST https://api.ebay.com/identity/v1/oauth2/token  (token mint)
  - GET  https://api.ebay.com/buy/browse/v1/item_summary/search  (search)

Scope used: https://api.ebay.com/oauth/api_scope (public Browse access).
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urlencode

import requests


PRODUCTION_HOST = "https://api.ebay.com"
SANDBOX_HOST = "https://api.sandbox.ebay.com"
BROWSE_SCOPE = "https://api.ebay.com/oauth/api_scope"


def _hosts_for(env: str) -> tuple[str, str]:
    env = (env or "production").lower()
    base = SANDBOX_HOST if env.startswith("sand") else PRODUCTION_HOST
    return f"{base}/identity/v1/oauth2/token", f"{base}/buy/browse/v1/item_summary/search"


# Defaults follow env at import time; can be overridden per-instance.
TOKEN_URL, BROWSE_URL = _hosts_for(os.environ.get("EBAY_ENV", "production"))


@dataclass
class ListingSummary:
    item_id: str
    title: str
    price_usd: Optional[float]
    listing_type: str
    seller: str
    seller_feedback: Optional[int]
    item_url: str
    end_time: Optional[str]
    image_url: Optional[str]


class EbayBrowseClient:
    """Lightweight Browse-API client with auto token refresh."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        marketplace: str = "EBAY_US",
        env: Optional[str] = None,
        user_agent: str = "prospects-research/0.1",
    ):
        self.client_id = client_id or os.environ.get("EBAY_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("EBAY_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.environ.get("EBAY_USER_REFRESH_TOKEN")
        self._bearer = bearer_token or os.environ.get("EBAY_BEARER_TOKEN")
        self._bearer_expires_at: float = float("inf") if self._bearer else 0.0
        self.marketplace = marketplace
        self.env = env or os.environ.get("EBAY_ENV", "production")
        self.token_url, self.browse_url = _hosts_for(self.env)
        self.user_agent = user_agent
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _mint_app_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise RuntimeError(
                "No bearer token and no client credentials available. "
                "Set EBAY_BEARER_TOKEN or both EBAY_CLIENT_ID/EBAY_CLIENT_SECRET."
            )
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        data = urlencode({"grant_type": "client_credentials", "scope": BROWSE_SCOPE})
        r = self.session.post(
            self.token_url,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=data,
            timeout=30,
        )
        if not r.ok:
            raise RuntimeError(
                f"Token mint failed ({r.status_code}): {r.text}"
            )
        payload = r.json()
        self._bearer = payload["access_token"]
        # Refresh ~60s before actual expiry
        self._bearer_expires_at = time.time() + max(60, payload.get("expires_in", 7200) - 60)
        return self._bearer

    def _mint_user_token(self) -> str:
        if not (self.client_id and self.client_secret and self.refresh_token):
            raise RuntimeError("User token mint requires CLIENT_ID, CLIENT_SECRET, and REFRESH_TOKEN")
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()
        ).decode()
        data = urlencode({
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "scope": BROWSE_SCOPE,
        })
        r = self.session.post(
            self.token_url,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data=data,
            timeout=30,
        )
        if not r.ok:
            raise RuntimeError(
                f"Token mint failed ({r.status_code}): {r.text}"
            )
        payload = r.json()
        self._bearer = payload["access_token"]
        self._bearer_expires_at = time.time() + max(60, payload.get("expires_in", 7200) - 60)
        return self._bearer

    def _ensure_token(self) -> str:
        if self._bearer and time.time() < self._bearer_expires_at:
            return self._bearer
        if self.refresh_token:
            return self._mint_user_token()
        return self._mint_app_token()

    # ------------------------------------------------------------------
    # Browse API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 50,
        category_ids: Optional[Iterable[str]] = None,
        sort: str = "price",
        max_pages: int = 3,
        sleep_between_pages: float = 0.4,
        fixed_price_only: bool = True,
    ) -> list[ListingSummary]:
        """Return active listings matching the query. Paginates if needed.

        fixed_price_only: when True (default) restricts to Buy-It-Now listings
        so auction snipes don't contaminate price discovery.
        """
        token = self._ensure_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace,
            "User-Agent": self.user_agent,
        }
        results: list[ListingSummary] = []
        offset = 0
        for _ in range(max_pages):
            params = {
                "q": query,
                "limit": str(min(limit, 200)),
                "offset": str(offset),
                "sort": sort,
            }
            if fixed_price_only:
                params["filter"] = "buyingOptions:{FIXED_PRICE}"
            if category_ids:
                params["category_ids"] = ",".join(category_ids)

            # Retry loop for 429 (rate limit) and 5xx (server-side hiccups).
            r = None
            for attempt in range(4):
                r = self.session.get(self.browse_url, headers=headers,
                                     params=params, timeout=30)
                if r.status_code == 401:
                    # Refresh token once mid-batch
                    self._bearer = None
                    token = self._ensure_token()
                    headers["Authorization"] = f"Bearer {token}"
                    continue
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    retry_after = float(r.headers.get("Retry-After", "0") or 0)
                    backoff = max(retry_after, 1.0 * (2 ** attempt))
                    raise_msg = f"HTTP {r.status_code} (attempt {attempt+1}/4); sleeping {backoff:.1f}s"
                    # Surface this so the caller knows what's happening
                    print(f"[ebay] {raise_msg}", flush=True)
                    time.sleep(backoff)
                    continue
                break
            if r is None:
                raise RuntimeError("eBay search failed with no response")
            r.raise_for_status()
            payload = r.json()
            items = payload.get("itemSummaries", [])
            for it in items:
                price = it.get("price") or {}
                seller = it.get("seller") or {}
                results.append(ListingSummary(
                    item_id=it.get("itemId", ""),
                    title=it.get("title", ""),
                    price_usd=_to_float(price.get("value")),
                    listing_type=",".join(it.get("buyingOptions") or []),
                    seller=seller.get("username", ""),
                    seller_feedback=_to_int(seller.get("feedbackScore")),
                    item_url=it.get("itemWebUrl", ""),
                    end_time=it.get("itemEndDate"),
                    image_url=(it.get("image") or {}).get("imageUrl"),
                ))
            total = payload.get("total", 0)
            offset += len(items)
            if not items or offset >= total:
                break
            time.sleep(sleep_between_pages)
        return results


def _to_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
