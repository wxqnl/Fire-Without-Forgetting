"""Stage 2.5 Phase A: verify every refs.bib entry (Crossref by DOI; else arXiv title search + Crossref bibliographic search)."""
import re, json, sys, time, difflib, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
UA = {"User-Agent": "fwf-integrity/0.1 (mailto:none@example.org)"}
def get(url):
    err = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            err = e; time.sleep(4)
    return ""
def clean(s):
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s); s = re.sub(r"[{}\"\\']", "", s)
    return re.sub(r"\s+", " ", s).strip()
def norm(s): return re.sub(r"[^a-z0-9 ]", "", clean(s).lower())
def sim(a, b): return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
bib = open(sys.argv[1]).read()
entries = []
for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", bib, re.S):
    f = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n", m.group(3) + "\n", re.S))
    entries.append((m.group(2).strip(), f))
ns = {"a": "http://www.w3.org/2005/Atom"}
for key, f in entries:
    title, year = clean(f.get("title", "")), f.get("year", "")
    first = clean(f.get("author", "").split(" and ")[0]).split()[-1] if f.get("author") else ""
    if "doi" in f:
        txt = get("https://api.crossref.org/works/" + f["doi"])
        if not txt: print(f"{key} | DOI {f['doi']} | NOT_FOUND"); continue
        m = json.loads(txt)["message"]; t = (m.get("title") or [""])[0]
        au = [a.get("family", "") for a in m.get("author", [])]
        y = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        ok = sim(title, t) > 0.85 and (first.lower() in " ".join(au).lower()) and str(y) == year
        print(f"{key} | DOI | {'VERIFIED' if ok else 'CHECK'} | sim={sim(title,t):.2f} | {t} | {y} | {(m.get('container-title') or [''])[0]} | vol={m.get('volume')} pp={m.get('page')} | first={au[:1]} n_auth={len(au)}/{len(f.get('author','').split(' and '))}")
        time.sleep(0.3); continue
    q = urllib.parse.quote(f'ti:"{title}"')
    txt = get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=3")
    best = None
    if txt:
        for e in ET.fromstring(txt).findall("a:entry", ns):
            t = " ".join(e.find("a:title", ns).text.split()); au = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
            s = sim(title, t)
            if not best or s > best[0]: best = (s, t, e.find("a:id", ns).text.split("/abs/")[-1], e.find("a:published", ns).text[:4], au)
    if best and best[0] > 0.85 and first.lower() in " ".join(best[4]).lower():
        print(f"{key} | arXiv | VERIFIED | sim={best[0]:.2f} | {best[1]} | arXiv:{best[2]} ({best[3]}) | first={best[4][0]} n_auth={len(best[4])}/{len(f.get('author','').split(' and '))} | venue(bib)={clean(f.get('booktitle', f.get('journal','')))} {year}")
    else:
        txt = get("https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": title + " " + first, "rows": 3}))
        hit = None
        for it in (json.loads(txt)["message"]["items"] if txt else []):
            t = (it.get("title") or [""])[0]; au = [a.get("family", "") for a in it.get("author", [])]
            if sim(title, t) > 0.85 and first.lower() in " ".join(au).lower(): hit = (t, it.get("DOI"), (it.get("issued", {}).get("date-parts") or [[None]])[0][0], (it.get("container-title") or [''])[0]); break
        print(f"{key} | xref-search | {'VERIFIED' if hit else 'NOT_FOUND_AUTO'} | {hit if hit else (best[1:4] if best else '')} | bib: {title} / {first} / {year}")
    time.sleep(3.2)
