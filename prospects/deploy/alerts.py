"""Daily alerts + digest email — runs on Railway at 14:00 UTC.

Inputs:
  - holdings.csv             : your owned cards
  - prices_holdings_latest   : today's lowest_buynow for each held player
  - prices_buylist_latest    : today's buy-list prices
  - alerts_state.json        : already-fired 2x triggers (don't re-spam)

Fires:
  - 2x trigger: per held card where lowest_buynow_price >= 2 * buy_price_usd
                (raw, base 1st Bowman Chrome auto)
  - Daily digest: portfolio value at lowest_buynow + top N buy candidates

Email via SendGrid (env: SENDGRID_API_KEY, ALERT_FROM, ALERT_TO).

Holdings CSV schema (user maintains):
  card_id, player_id, name, denominator, grade, buy_date, buy_price_usd,
  ebay_item_id, notes

The 2x trigger requires `denominator == 0` (base) and `grade == "" or "raw"`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _is_raw(grade: str) -> bool:
    g = (grade or "").strip().lower()
    return g in ("", "raw")


def evaluate_holdings(holdings: list[dict], h_prices: list[dict],
                      state: dict) -> tuple[list[dict], list[dict], float]:
    """Returns (new_2x_alerts, all_holdings_valued, portfolio_value)."""
    # Map player_id -> {lowest_buynow_price, lowest_buynow_url, price_median}
    by_pid: dict[str, dict] = {}
    for r in h_prices:
        if not int(r.get("has_market") or 0):
            continue
        if int(r.get("denominator") or 0) != 0:
            continue
        by_pid[r["player_id"]] = r

    new_alerts: list[dict] = []
    valued: list[dict] = []
    portfolio = 0.0
    fired_state = set(state.get("fired_2x_card_ids", []))

    for h in holdings:
        card_id = h.get("card_id") or ""
        pid = h.get("player_id") or ""
        denom = int(h.get("denominator") or 0)
        buy_price = _to_float(h.get("buy_price_usd"))
        grade = h.get("grade") or ""
        market = by_pid.get(pid)
        cur_buynow = _to_float((market or {}).get("lowest_buynow_price"))
        cur_median = _to_float((market or {}).get("price_median"))
        cur_url = (market or {}).get("lowest_buynow_url") or ""

        # Only valuate when the comparison is apples-to-apples: base + raw
        valuation_eligible = denom == 0 and _is_raw(grade)

        if cur_buynow is not None and valuation_eligible:
            portfolio += cur_buynow

        valued.append({
            "card_id": card_id,
            "player_id": pid,
            "name": h.get("name") or (market or {}).get("name") or "",
            "denominator": denom,
            "grade": grade,
            "buy_date": h.get("buy_date") or "",
            "buy_price_usd": buy_price,
            "current_lowest_buynow": cur_buynow,
            "current_median": cur_median,
            "current_lowest_buynow_url": cur_url,
            "multiple": (cur_buynow / buy_price) if (
                cur_buynow and buy_price and buy_price > 0) else None,
        })

        if not valuation_eligible:
            continue
        if buy_price is None or buy_price <= 0:
            continue
        if cur_buynow is None:
            continue
        if cur_buynow < 2.0 * buy_price:
            continue
        if card_id in fired_state:
            continue
        new_alerts.append({
            "card_id": card_id,
            "player_id": pid,
            "name": h.get("name") or (market or {}).get("name") or "",
            "buy_price_usd": buy_price,
            "current_lowest_buynow": cur_buynow,
            "multiple": cur_buynow / buy_price,
            "current_lowest_buynow_url": cur_url,
            "buy_date": h.get("buy_date") or "",
        })

    return new_alerts, valued, portfolio


def render_email_html(snapshot_date: str, new_alerts: list[dict],
                      valued: list[dict], portfolio: float,
                      buy_top: list[dict]) -> tuple[str, str]:
    subject = f"[prospects] {snapshot_date} digest"
    if new_alerts:
        subject = f"[prospects] {len(new_alerts)} 2x ALERT • {snapshot_date}"

    def fmt_money(x):
        return f"${x:,.2f}" if isinstance(x, (int, float)) else "—"

    parts = []
    parts.append(f"<h2>{snapshot_date}</h2>")

    if new_alerts:
        parts.append("<h3>\U0001f7e2 2x alerts (lowest buy-now &ge; 2&times; "
                     "your cost)</h3><ul>")
        for a in new_alerts:
            parts.append(
                f"<li><b>{a['name']}</b> — bought "
                f"{a['buy_date']} @ {fmt_money(a['buy_price_usd'])}, "
                f"current lowest buy-now {fmt_money(a['current_lowest_buynow'])} "
                f"(<b>{a['multiple']:.2f}×</b>) — "
                f"<a href=\"{a['current_lowest_buynow_url']}\">listing</a>"
                f"</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>No new 2x triggers today.</p>")

    parts.append(f"<h3>Portfolio (raw base 1st Bowman Chrome autos)</h3>")
    parts.append(f"<p>Lowest-buynow value: <b>{fmt_money(portfolio)}</b> "
                 f"across {sum(1 for v in valued if v['current_lowest_buynow'])}"
                 f"/{len(valued)} held cards with market.</p>")
    parts.append("<table border=1 cellpadding=4 cellspacing=0>"
                 "<tr><th>Name</th><th>Buy</th><th>Now</th>"
                 "<th>x</th><th>Bought</th></tr>")
    for v in sorted(valued,
                    key=lambda r: (r["multiple"] or 0), reverse=True):
        mult = f"{v['multiple']:.2f}x" if v["multiple"] else "—"
        parts.append(f"<tr><td>{v['name']}</td>"
                     f"<td>{fmt_money(v['buy_price_usd'])}</td>"
                     f"<td>{fmt_money(v['current_lowest_buynow'])}</td>"
                     f"<td>{mult}</td>"
                     f"<td>{v['buy_date']}</td></tr>")
    parts.append("</table>")

    if buy_top:
        parts.append("<h3>Top buy candidates (cheapest current buy-now)</h3>")
        parts.append("<table border=1 cellpadding=4 cellspacing=0>"
                     "<tr><th>Name</th><th>Lowest buy-now</th><th>Median</th>"
                     "<th>Listing</th></tr>")
        for r in buy_top:
            parts.append(
                f"<tr><td>{r.get('name','')}</td>"
                f"<td>{fmt_money(_to_float(r.get('lowest_buynow_price')))}</td>"
                f"<td>{fmt_money(_to_float(r.get('price_median')))}</td>"
                f"<td><a href=\"{r.get('lowest_buynow_url','')}\">link</a>"
                f"</td></tr>")
        parts.append("</table>")

    return subject, "\n".join(parts)


def send_email(subject: str, html: str, to_addr: str, from_addr: str,
               api_key: str) -> None:
    """SendGrid v3 API send. Raises on non-2xx."""
    import requests
    r = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "personalizations": [{"to": [{"email": to_addr}]}],
            "from": {"email": from_addr},
            "subject": subject,
            "content": [{"type": "text/html", "value": html}],
        },
        timeout=30,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"SendGrid {r.status_code}: {r.text}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--holdings", default=os.environ.get("HOLDINGS_PATH",
                                                        "/data/holdings.csv"))
    p.add_argument("--prices-dir", default=os.environ.get("PRICES_DIR",
                                                          "/data/prices"))
    p.add_argument("--state", default=os.environ.get("ALERTS_STATE_PATH",
                                                     "/data/alerts_state.json"))
    p.add_argument("--top-n-buy", type=int, default=15)
    p.add_argument("--dry-run", action="store_true",
                   help="Render to stdout instead of emailing")
    args = p.parse_args()

    _load_env_file(Path(".env"))

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prices_dir = Path(args.prices_dir)

    holdings = _read_csv(Path(args.holdings))
    h_prices = _read_csv(prices_dir / "prices_holdings_latest.csv")
    b_prices = _read_csv(prices_dir / "prices_buylist_latest.csv")

    state_path = Path(args.state)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    new_alerts, valued, portfolio = evaluate_holdings(holdings, h_prices, state)

    buy_top = [
        r for r in b_prices
        if int(r.get("has_market") or 0)
        and int(r.get("denominator") or 0) == 0
        and _to_float(r.get("lowest_buynow_price")) is not None
    ]
    buy_top.sort(key=lambda r: _to_float(r.get("lowest_buynow_price")) or 1e18)
    buy_top = buy_top[: args.top_n_buy]

    subject, html = render_email_html(snapshot_date, new_alerts, valued,
                                      portfolio, buy_top)

    if args.dry_run:
        print(f"SUBJECT: {subject}")
        print("---")
        print(html)
        return 0

    api_key = os.environ.get("SENDGRID_API_KEY")
    to_addr = os.environ.get("ALERT_TO")
    from_addr = os.environ.get("ALERT_FROM")
    if not (api_key and to_addr and from_addr):
        print("ERROR: missing SENDGRID_API_KEY / ALERT_TO / ALERT_FROM",
              file=sys.stderr)
        return 2
    send_email(subject, html, to_addr, from_addr, api_key)
    print(f"[alerts] sent: {subject}")

    # Persist fired-state so we don't re-spam the same card_id every day
    fired = set(state.get("fired_2x_card_ids", []))
    for a in new_alerts:
        if a["card_id"]:
            fired.add(a["card_id"])
    state["fired_2x_card_ids"] = sorted(fired)
    state["last_run"] = snapshot_date
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    print(f"[alerts] state -> {state_path} ({len(fired)} fired card_ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
