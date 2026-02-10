import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import requests
from bs4 import BeautifulSoup

from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

URLS = {
    1: "https://www.pagibigfundservices.com/Magpalistasa4ph/Project/Projects?loc=1",
    2: "https://www.pagibigfundservices.com/Magpalistasa4ph/Project/Projects?loc=2",
}

STATE_FILE = os.getenv("PAGIBIG_STATE_FILE", "pagibig_state.json")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

TZ_NAME = os.getenv("TZ", "Asia/Manila")
TZ = pytz.timezone(TZ_NAME)

BASE_URL = "https://www.pagibigfundservices.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

ASOF_REGEX = re.compile(r"As of\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", re.IGNORECASE)
DETAILS_LINK_REGEX = re.compile(r"/MagpalistaSa4PH/Project/Details/\d+", re.IGNORECASE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------- TELEGRAM ----------------
def tg_send(message: str) -> None:
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise RuntimeError("Missing TELEGRAM_TOKEN or CHAT_ID env vars.")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "disable_web_page_preview": True}
    r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()


# ---------------- STATE ----------------
def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {"locs": {}}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"locs": {}}


def save_state(state: Dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


# ---------------- PARSING ----------------
def parse_asof_date(text: str) -> Optional[datetime]:
    m = ASOF_REGEX.search(text)
    if not m:
        return None
    s = m.group(1).strip()
    try:
        return datetime.strptime(s, "%B %d, %Y")
    except ValueError:
        return None


def fetch_soup(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def extract_details_links(list_soup: BeautifulSoup) -> List[str]:
    """
    From the Projects list page, extract unique Details links.
    """
    links = []
    seen = set()

    for a in list_soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if not DETAILS_LINK_REGEX.search(href):
            continue

        if href.startswith("http"):
            full = href
        else:
            full = BASE_URL.rstrip("/") + (href if href.startswith("/") else "/" + href)

        if full not in seen:
            seen.add(full)
            links.append(full)

    return links


BAD_TITLES = {
    "quantity", "no. of units", "no of units", "paalala", "paumanhin",
    "browse our website", "membership", "savings", "tingnan ang ibang 4ph",
    "tingnan ang ibang 4ph projects", "compute", "30 years term"
}

def extract_project_name_from_details(details_soup: BeautifulSoup) -> str:
    """
    Finds the heading closest BEFORE the 'As of ...' text.
    Avoids headings like 'Quantity' and 'No. of Units'.
    """
    full_text = details_soup.get_text("\n", strip=True)
    m = ASOF_REGEX.search(full_text)

    # If page has no "As of", fall back later
    asof_pos = m.start() if m else None

    candidates = []
    for tag in ["h1", "h2", "h3", "h4"]:
        for h in details_soup.find_all(tag):
            txt = " ".join(h.get_text(" ", strip=True).split())
            if not txt or len(txt) < 3:
                continue

            tnorm = txt.strip().lower()
            if tnorm in BAD_TITLES:
                continue
            # Skip obvious non-names
            if "sqm" in tnorm or "php" in tnorm:
                continue

            idx = full_text.find(txt)
            if idx == -1:
                continue

            # Prefer headings that appear BEFORE "As of"
            if asof_pos is not None and idx > asof_pos:
                continue

            candidates.append((idx, txt))

    if candidates:
        # Pick the closest heading BEFORE "As of"
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    # Fallback: try first meaningful h3/h2 anyway (but filtered)
    for tag in ["h3", "h2", "h1"]:
        for h in details_soup.find_all(tag):
            txt = " ".join(h.get_text(" ", strip=True).split())
            if txt and txt.strip().lower() not in BAD_TITLES and len(txt) >= 3:
                return txt

    if details_soup.title and details_soup.title.string:
        return " ".join(details_soup.title.string.strip().split())

    return "Unknown Project"

def fetch_details_asof(details_url: str) -> Tuple[str, Optional[datetime], str]:
    """
    Fetch details page and return (project_name, asof_dt, asof_str).
    """
    soup = fetch_soup(details_url)
    page_text = soup.get_text("\n", strip=True)

    asof_dt = parse_asof_date(page_text)
    asof_str = asof_dt.strftime("%B %d, %Y") if asof_dt else "Unknown"

    name = extract_project_name_from_details(soup)
    return name, asof_dt, asof_str


# ---------------- LATEST LIVE (what /pagibiglatest should call) ----------------
def get_latest_property_live() -> str:
    """
    LIVE: checks all project Details pages from loc=1 and loc=2,
    then returns the property/properties with the newest 'As of' date.
    """
    overall_latest: Optional[datetime] = None
    winners = []  # list of dicts: {"loc": int, "name": str, "asof": str, "url": str}

    for loc, list_url in URLS.items():
        try:
            list_soup = fetch_soup(list_url)
            detail_urls = extract_details_links(list_soup)

            if not detail_urls:
                logging.warning("No details links found for loc=%s", loc)
                continue

            for durl in detail_urls:
                try:
                    name, asof_dt, asof_str = fetch_details_asof(durl)
                    if asof_dt is None:
                        continue

                    if overall_latest is None or asof_dt > overall_latest:
                        overall_latest = asof_dt
                        winners = [{"loc": loc, "name": name, "asof": asof_str, "url": durl}]
                    elif overall_latest is not None and asof_dt == overall_latest:
                        winners.append({"loc": loc, "name": name, "asof": asof_str, "url": durl})
                except Exception:
                    continue

        except Exception as e:
            logging.exception("Live latest fetch failed for loc=%s: %s", loc, e)

    if overall_latest is None or not winners:
        return "❌ Could not determine the latest property (no per-project 'As of' dates found)."

    latest_asof = winners[0]["asof"]
    lines = [
        "🔥 Latest 4PH Property (LIVE)",
        f"📅 Latest 'As of': {latest_asof}",
        "",
        "🏠 Property/Properties with that latest date:"
    ]

    for i, w in enumerate(winners, start=1):
        lines.append(f"{i}. {w['name']} (loc={w['loc']})")
        lines.append(f"   {w['url']}")

    return "\n".join(lines)


# ---------------- SCAN JOB (scheduled alert every 4 hours) ----------------
def scan_job() -> None:
    """
    Scheduled scan logic:
    - find latest 'As of' per loc (from Details pages)
    - if latest asof moved forward since last saved => notify
    - save snapshot of latest per loc
    """
    state = load_state()
    locs_state = state.setdefault("locs", {})

    for loc, list_url in URLS.items():
        try:
            logging.info("Scanning (details) loc=%s ...", loc)

            list_soup = fetch_soup(list_url)
            detail_urls = extract_details_links(list_soup)

            # find newest for this loc
            latest_dt = None
            latest_name = None
            latest_url = None

            for durl in detail_urls:
                try:
                    name, asof_dt, asof_str = fetch_details_asof(durl)
                    if asof_dt is None:
                        continue
                    if latest_dt is None or asof_dt > latest_dt:
                        latest_dt = asof_dt
                        latest_name = name
                        latest_url = durl
                except Exception:
                    continue

            if latest_dt is None:
                logging.warning("No per-project dates found for loc=%s", loc)
                continue

            latest_asof_str = latest_dt.strftime("%B %d, %Y")

            loc_key = str(loc)
            prev_latest_asof = locs_state.get(loc_key, {}).get("latest_asof")

            should_notify = False
            if prev_latest_asof:
                try:
                    prev_dt = datetime.strptime(prev_latest_asof, "%B %d, %Y")
                    if latest_dt > prev_dt:
                        should_notify = True
                except ValueError:
                    should_notify = True
            else:
                # first run -> don't spam, but you can set to True if you want
                should_notify = False

            if should_notify:
                msg = (
                    f"🆕 Newer 4PH update detected (loc={loc})\n"
                    f"📅 Latest 'As of': {latest_asof_str}\n"
                    f"🏠 Property: {latest_name}\n"
                    f"{latest_url}"
                )
                tg_send(msg)
                logging.info("Notified loc=%s latest=%s", loc, latest_asof_str)
            else:
                logging.info("No newer 'As of' for loc=%s (latest=%s).", loc, latest_asof_str)

            # Save snapshot
            locs_state[loc_key] = {
                "latest_asof": latest_asof_str,
                "latest_name": latest_name or "Unknown",
                "latest_url": latest_url or "",
                "last_checked": datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_state(state)

        except Exception as e:
            logging.exception("Error scanning loc=%s: %s", loc, e)
            try:
                tg_send(f"⚠️ Pag-IBIG scanner error (loc={loc}): {e}")
            except Exception:
                pass

def get_latest_properties_by_loc_live() -> str:
    """
    LIVE: For each loc, checks all project Details pages and returns
    the newest project name + its 'As of' date.
    """
    lines = ["🔥 Latest 4PH Properties (LIVE)", ""]

    for loc, list_url in URLS.items():
        try:
            list_soup = fetch_soup(list_url)
            detail_urls = extract_details_links(list_soup)

            latest_dt = None
            latest_name = None
            latest_url = None
            latest_asof_str = "Unknown"

            for durl in detail_urls:
                try:
                    name, asof_dt, asof_str = fetch_details_asof(durl)
                    if asof_dt is None:
                        continue

                    if latest_dt is None or asof_dt > latest_dt:
                        latest_dt = asof_dt
                        latest_name = name
                        latest_url = durl
                        latest_asof_str = asof_str
                except Exception:
                    continue

            lines.append(f"🏠 loc={loc}")
            if latest_dt is None:
                lines.append("❌ No valid per-project 'As of' found.")
            else:
                lines.append(f"✅ Latest property: {latest_name}")
                lines.append(f"📅 As of: {latest_asof_str}")
                lines.append(f"🔗 {latest_url}")
            lines.append("")

        except Exception as e:
            lines.append(f"🏠 loc={loc}")
            lines.append(f"❌ Live fetch failed: {e}")
            lines.append("")

    return "\n".join(lines)


def main():
    # Send startup ping once
    try:
        tg_send("🤖 Pag-IBIG 4PH scanner started. Checking loc=1 and loc=2 every 4 hours.")
    except Exception as e:
        logging.warning("Startup Telegram ping failed: %s", e)

    sched = BackgroundScheduler(timezone=TZ)
    sched.add_job(scan_job, "interval", hours=4, next_run_time=datetime.now(TZ), id="pagibig_scan")
    sched.start()

    logging.info("Pag-IBIG scheduler running (every 4 hours). TZ=%s", TZ_NAME)
    try:
        while True:
            import time
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
