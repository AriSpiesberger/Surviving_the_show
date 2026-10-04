// SportsCardsPro scraper, run as a Web Worker inside a sportscardspro.com tab
// (Cloudflare blocks plain HTTP clients; a worker isn't throttled like a hidden tab).
//
// Discovers every Bowman Chrome autograph set for the configured years, lists each
// set's base cards (exclude-variants), then pulls each card page for its monthly
// price series (raw / PSA 9 / PSA 10, in cents) and its recent completed sales.
// Rows buffer in IndexedDB and flush to research/card_market/receiver.py on
// 127.0.0.1:8765 whenever it's reachable.
let BASE = '';
let stats = {phase: 'init', sets: 0, setsDone: 0, prods: 0, prodsDone: 0, err: 0, backoff: 0, flushed: 0, lastErr: null, lastFlush: null};
const post = () => postMessage(stats);
const sleep = ms => new Promise(r => setTimeout(r, ms));
let db;

function openDb() {
  return new Promise((res, rej) => {
    const o = indexedDB.open('scp_scrape', 1);
    o.onupgradeneeded = () => o.result.createObjectStore('rows', {autoIncrement: true});
    o.onsuccess = () => res(o.result);
    o.onerror = () => rej(o.error);
  });
}

function add(kind, row) {
  return new Promise((res, rej) => {
    const t = db.transaction('rows', 'readwrite');
    t.objectStore('rows').add({kind, row});
    t.oncomplete = res;
    t.onerror = () => rej(t.error);
  });
}

async function flush() {
  try {
    const c = new AbortController();
    setTimeout(() => c.abort(), 4000);
    await fetch('http://127.0.0.1:8765/', {signal: c.signal});
  } catch (e) { stats.lastFlush = 'unreachable'; return; }
  while (true) {
    const batch = await new Promise(res => {
      const out = [];
      const c = db.transaction('rows').objectStore('rows').openCursor();
      c.onsuccess = e => {
        const cur = e.target.result;
        if (cur && out.length < 300) { out.push([cur.key, cur.value]); cur.continue(); } else res(out);
      };
    });
    if (!batch.length) { stats.lastFlush = 'flushed'; return; }
    const by = {};
    for (const [, v] of batch) (by[v.kind] = by[v.kind] || []).push(v.row);
    for (const kind in by) {
      const r = await fetch('http://127.0.0.1:8765/', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind, rows: by[kind]})});
      if (!r.ok) { stats.lastFlush = 'post ' + r.status; return; }
    }
    await new Promise(res => {
      const t = db.transaction('rows', 'readwrite');
      const s = t.objectStore('rows');
      for (const [k] of batch) s.delete(k);
      t.oncomplete = res;
    });
    stats.flushed += batch.length;
  }
}

async function get(url, json) {
  if (url[0] === '/') url = BASE + url;
  for (let a = 0; a < 5; a++) {
    try {
      const r = await fetch(url, {credentials: 'include'});
      const t = await r.text();
      if (r.status === 200) {
        if (!json) return t;
        try { return JSON.parse(t); } catch (e) {}
      }
      stats.lastErr = r.status + ' ' + url.slice(0, 90);
    } catch (e) { stats.lastErr = String(e).slice(0, 80); }
    stats.backoff++; post();
    await sleep(20000 * (a + 1));
  }
  stats.err++;
  return null;
}

function strip(s) {
  return s.replace(/<[^>]+>/g, ' ').replace(/&amp;/g, '&').replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, ' ').trim();
}

// Completed sales live in tables under div.completed-auctions-<grade>; the class
// appears on both an empty tab header and the populated block, so take the one
// with a <tbody>.
function parseSales(h, cls) {
  let j = h.indexOf('class="' + cls + '"');
  let body = null;
  while (j >= 0) {
    const seg = h.slice(j, j + 200000);
    const end = seg.indexOf('</table>');
    const tb = seg.indexOf('<tbody');
    if (tb >= 0 && tb < end) { body = seg.slice(tb, end); break; }
    j = h.indexOf('class="' + cls + '"', j + 10);
  }
  if (!body) return [];
  return body.split('<tr').slice(1).map(tr => {
    const d = (tr.match(/(\d{4}-\d\d-\d\d)/) || [])[1];
    const p = (tr.match(/js-price[^>]*>\s*\$([\d,\.]+)/) || [])[1];
    const t = (tr.match(/<td class="title"[^>]*>([\s\S]*?)<\/td>/) || [])[1];
    return d && p ? [d, +p.replace(/,/g, ''), t ? strip(t).slice(0, 110) : ''] : null;
  }).filter(Boolean);
}

function parseProd(id, h) {
  const m = h.match(/VGPC\.chart_data\s*=\s*(\{[\s\S]*?\});/);
  let chart = null;
  if (m) {
    try {
      const d = JSON.parse(m[1]);
      chart = {};
      for (const k of ['used', 'graded', 'manualonly'])
        chart[k] = (d[k] || []).filter(x => x[1] > 0).map(x => [new Date(x[0]).toISOString().slice(0, 7), x[1]]);
    } catch (e) {}
  }
  return {id, chart, sales: {
    raw: parseSales(h, 'completed-auctions-used'),
    psa9: parseSales(h, 'completed-auctions-graded'),
    psa10: parseSales(h, 'completed-auctions-manual-only'),
  }};
}

function slug(c) {
  return c.toLowerCase().replace(/'/g, '%27').replace(/[^a-z0-9%&]+/g, '-').replace(/^-|-$/g, '');
}

onmessage = async (e) => {
  const cfg = e.data;
  BASE = cfg.base;
  db = await openDb();
  const sets = {};
  stats.phase = 'discover'; post();
  for (let y = cfg.y0; y <= cfg.y1; y++) for (const t of cfg.templates) {
    const j = await get('/search-products?q=' + encodeURIComponent(y + ' ' + t) + '&type=prices&format=json', true);
    for (const p of (j && j.products) || []) {
      const c = p.consoleName || '';
      if (/^Baseball Cards/.test(c) && /bowman/i.test(c) && /chrome/i.test(c) && /auto/i.test(c)) sets[c] = (sets[c] || 0) + 1;
    }
    stats.discoverYear = y;
  }
  const setNames = Object.keys(sets).sort();
  stats.sets = setNames.length; stats.setNames = setNames; stats.phase = 'sets'; post();
  const prods = [];
  for (const c of setNames) {
    const path = '/console/' + slug(c);
    const j = await get(path + '?format=json&exclude-variants=true', true);
    const ps = (j && j.products) || [];
    await add('setlist', {set: c, path, n: ps.length, products: ps.map(p => ({id: p.id, n: p.productName, p1: p.price1, p2: p.price2, p3: p.price3, pr: p.printRun}))});
    for (const p of ps) prods.push({id: p.id, n: p.productName, c});
    stats.setsDone++; post();
  }
  stats.prods = prods.length; stats.phase = 'products'; post();
  await flush();
  for (const p of prods) {
    const h = await get('/game/' + p.id, false);
    if (h) {
      const row = parseProd(p.id, h);
      row.n = p.n; row.c = p.c;
      await add('product', row);
    }
    stats.prodsDone++;
    if (stats.prodsDone % 20 === 0) { await flush(); post(); }
  }
  await flush();
  stats.phase = 'done'; post();
};
