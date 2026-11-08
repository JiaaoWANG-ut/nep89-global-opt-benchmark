"""HTML assembly of the bilingual report page."""
import html as _h
import re

from .text import METHODS, T

_CJK = "\u3000-\u303f\u4e00-\u9fff\uff00-\uffef"
_CJK_GAP = re.compile(f"(?<=[{_CJK}])\\s+(?=[{_CJK}])")
_CJK_PUNCT_GAP = re.compile("\\s+(?=[\u3001\u3002\uff08\uff09\uff0c\uff1a\uff1b])|(?<=[\u3001\u3002\uff08\uff09\uff0c\uff1a\uff1b])\\s+")

CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,'PingFang SC','Microsoft YaHei','Noto Sans CJK SC',
'Source Han Sans SC','WenQuanYi Micro Hei',sans-serif;
margin:0;color:#222;background:#fafafa;line-height:1.5}
header{background:linear-gradient(120deg,#1f3b73,#2f6f9f);color:#fff;padding:28px 36px}
header h1{margin:0 0 6px;font-size:26px}header p{margin:4px 0;opacity:.92}
header a{color:#ffe08a}nav{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;
padding:8px 36px;z-index:5;font-size:14px}nav a{margin-right:14px;color:#1f3b73;text-decoration:none}
main{max-width:1320px;margin:0 auto;padding:10px 36px 40px}section{margin-top:34px}
h2{border-bottom:2px solid #2f6f9f;padding-bottom:4px;color:#1f3b73}
.cards{display:flex;gap:14px;flex-wrap:wrap}.card{background:#fff;border:1px solid #e3e3e3;border-radius:8px;
padding:12px 18px;min-width:130px}.card b{display:block;font-size:24px;color:#1f3b73}
table{border-collapse:collapse;background:#fff;font-size:13px;margin:10px 0;width:100%}
th,td{border:1px solid #e1e1e1;padding:4px 7px;text-align:right;white-space:nowrap}
th{background:#eef3f8;cursor:pointer;position:sticky;top:0;z-index:1}td.l,th.l{text-align:left;white-space:normal}
td.nw{white-space:nowrap}
tr:nth-child(even) td{background:#fbfbfd}tr.innovative td:nth-child(2){font-weight:600}
.badge{border-radius:4px;padding:1px 6px;font-size:11px;color:#fff}.badge.baseline{background:#4C72B0}
.badge.innovative{background:#DD8452}.small{color:#777;font-size:11px}.ci{color:#888;font-size:11px}
img.fig{max-width:100%;background:#fff;border:1px solid #e3e3e3;border-radius:6px;margin:8px 0}
.two{display:flex;gap:12px;flex-wrap:wrap}.two img{max-width:49%}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
.gallery div{background:#fff;border:1px solid #e3e3e3;border-radius:6px;padding:6px;text-align:center;font-size:12px}
.gallery img{width:100%}.find li{margin:5px 0}.wrap{overflow-x:auto;margin:10px 0}
.wrap.tall{overflow:auto;max-height:85vh}.wrap table{margin:0}
details{margin:12px 0}summary{cursor:pointer;font-weight:600}.desc li{margin:6px 0;font-size:14px;line-height:1.45}
footer{color:#888;font-size:12px;text-align:center;padding:20px}
"""

JS = """
document.querySelectorAll('table.sortable').forEach(function(t){
 t.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
  var b=t.tBodies[0],rows=Array.from(b.rows),asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
  rows.sort(function(x,y){var a=x.cells[i].dataset.v,c=y.cells[i].dataset.v;
   if(a===undefined){a=x.cells[i].innerText;c=y.cells[i].innerText;return asc?a.localeCompare(c):c.localeCompare(a);}
   return asc?(parseFloat(a)-parseFloat(c)):(parseFloat(c)-parseFloat(a));});
  rows.forEach(function(r){b.appendChild(r);});});});});
"""


def esc(s):
    return _h.escape(str(s))


# display-only typography for the ASCII benchmark descriptions (problem.json is hashed, so not edited)
_PRETTY = [(re.compile(r"(?<=\d) A(?=[\s,;:.)\]/^]|$)"), " &Aring;"),
           (re.compile(r"&Aring;\^3"), "&Aring;&sup3;"),
           (re.compile(r"(?<=\d)( ?)x( ?)(?=\d)"), r"\1&times;\2"),
           (re.compile(r" -&gt; "), " &rarr; "),
           (re.compile(r"&gt;="), "&ge;"),
           (re.compile(r"&lt;="), "&le;"),
           (re.compile(r"\b([A-Z][a-z]?)_([nN])\b"), r"\1<sub>\2</sub>"),
           (re.compile(r"\be_tol\b"), "e<sub>tol</sub>"),
           (re.compile(r"\bE_best\b"), "E<sub>best</sub>")]


def prose(s):
    s = esc(s)
    for p, r in _PRETTY:
        s = p.sub(r, s)
    return s


def fmt_ert(v):
    if v is None or v != v:
        return "", -1
    if v == float("inf"):
        return "&infin;", 1e30
    if v >= 1e5:
        return f"{v / 1e3:.0f}k", v
    if v >= 1e4:
        return f"{v / 1e3:.1f}k", v
    return f"{v:.0f}", v


def cell(x, sort=None, cls=""):
    c = f' class="{cls}"' if cls else ""
    if sort is None:
        return f"<td{c}>{x}</td>"
    s = sort if sort == sort else -1e30
    return f'<td{c} data-v="{s:.6g}">{x}</td>'


def table(headers, rows, row_cls=None, left=(1,), tall=False):
    th = "".join(f'<th class="{"l" if i in left else ""}">{h}</th>' for i, h in enumerate(headers))
    body = []
    for k, r in enumerate(rows):
        rc = f' class="{row_cls[k]}"' if row_cls else ""
        body.append(f"<tr{rc}>" + "".join(r) + "</tr>")
    return (f'<div class="wrap{" tall" if tall else ""}"><table class="sortable"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def page(lang, R):
    k = 0 if lang == "en" else 1
    t = lambda key: T[key][k]  # noqa: E731
    other = "zh.html" if lang == "en" else "index.html"
    secs = ["summary", "ranking", "matrix", "profiles", "categories", "stats", "cost", "benchmarks",
            "algorithms", "methods", "downloads"]
    nav = "".join(f'<a href="#{s}">{n}</a>' for s, n in zip(secs, t("nav")))
    out = [f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f"<title>{t('title')}</title><style>{CSS}</style></head><body>"
           f"<header><h1>{t('title')}</h1><p>{t('subtitle').format(**R)}</p>"
           f'<p><a href="{other}">{t("other_lang")}</a> &middot; {t("generated").format(**R)}'
           + (f' &middot; <a href="{R["repo"]}">GitHub</a>' if R.get("repo") else "")
           + f"</p></header><nav>{nav}</nav><main>"]
    # summary
    vals = [R["nb"], R["na"], f"{R['nruns']:,}", f"{R['steps'] / 1e6:.1f} M", f"{R['corehours']:,.0f}"]
    cards = "".join(f"<div class='card'><b>{v}</b>{n}</div>" for v, n in zip(vals, t("cards")))
    finds = "".join(f"<li>{f}</li>" for f in R["findings"][k])
    notes = R.get("notes", ["", ""])[k]
    out.append(f'<section id="summary"><h2>{t("nav")[0]}</h2><div class="cards">{cards}</div>'
               f'<h3>{t("findings")}</h3><ul class="find">{finds}</ul>{notes}</section>')
    # ranking
    rows, rcls = [], []
    for a in R["algs"]:
        ci = f'<span class="ci">[{a["ci_lo"]:.1f}, {a["ci_hi"]:.1f}]</span>'
        sp = f"&times;{a['speedup']:.2f}" if a["speedup"] == a["speedup"] else ""
        rows.append([
            cell(a["rank"], a["rank"]),
            cell(f'{esc(a["name"])} <span class="small">{esc(a["key"])}</span>', cls="l"),
            cell(f'<span class="badge {a["type"]}">{t("type")[a["type"]]}</span>'),
            cell(f'<b>{a["score_t"]:.1f}</b> {ci}', a["score_t"]),
            cell(f'{a["p_best"]:.0%}', a["p_best"]),
            cell(f'{a["gmre"]:.3f}', a["gmre"]),
            cell(f'{a["avg_rank"]:.2f}', a["avg_rank"]),
            cell(f'{a["solved"]}/{R["nt_rows"]}', a["solved"]),
            cell(f'{a["pooled_sr"]:.0%}', a["pooled_sr"]),
            cell(sp, a["speedup"]),
            cell(f'{a["score_all"]:.1f}', a["score_all"]),
            cell(f'{a["score_dev"]:.1f}', a["score_dev"]),
            cell(f'{a["score_wall"]:.1f}', a["score_wall"]),
            cell(f'{a["overhead"]:.2f}', a["overhead"])])
        rcls.append(a["type"])
    uns = (" " + t("unsolved").format(n=len(R["unsolved"]), names=(", ", "、")[k].join(R["unsolved"]))
           if R.get("unsolved") else "")
    out.append(f'<section id="ranking"><h2>{t("nav")[1]}</h2><p>{t("rank_intro").format(**R)}{uns}</p>'
               + table(t("rank_cols"), rows, rcls)
               + f'<img class="fig" src="img/rank_bars.png" alt="ranking"></section>')
    out.append(f'<section id="matrix"><h2>{t("nav")[2]}</h2><p>{t("matrix_intro")}</p>'
               f'<img class="fig" src="img/heatmap.png" alt="heatmap"></section>')
    out.append(f'<section id="profiles"><h2>{t("nav")[3]}</h2><p>{t("profiles_intro")}</p><div class="two">'
               f'<img class="fig" src="img/perf_profile.png" alt="profile">'
               f'<img class="fig" src="img/ecdf.png" alt="ecdf"></div></section>')
    out.append(f'<section id="categories"><h2>{t("nav")[4]}</h2><p>{t("cat_intro")}</p>'
               f'<img class="fig" src="img/categories.png" alt="categories"></section>')
    fr = R.get("friedman")
    st = (f"<p>{t('stats_intro').format(**fr)}</p><img class='fig' src='img/cd.png' alt='cd'>" if fr else "")
    if R.get("sens"):
        rows = [[cell(f"{s['tol']:g}", cls="l"), cell(f"{s['tau']:.2f}", s["tau"]), cell(", ".join(s["top"]), cls="l")]
                for s in R["sens"]]
        rows.insert(0, [cell(f"{R['e_tol']:g} ({t('primary')})", cls="l"), cell("1.00", 1.0),
                        cell(", ".join(R["top3"]), cls="l")])
        st += f"<h3>{t('sens_title')}</h3><p>{t('sens_intro')}</p>" + table(t("sens_cols"), rows, left=(0, 2))
    out.append(f'<section id="stats"><h2>{t("nav")[5]}</h2>{st}</section>')
    out.append(f'<section id="cost"><h2>{t("nav")[6]}</h2><p>{t("cost_intro")}</p>'
               f'<img class="fig" src="img/overhead.png" alt="overhead"></section>')
    # benchmarks
    rows = []
    for b in R["benches"]:
        rows.append([
            cell(esc(b["name"]) + (" *" if b["dev"] else ""), cls="l"),
            cell(t("cat_short").get(b["category"], b["category"]), cls="l"),
            cell(esc(b["title"]), cls="l"), cell(f'{b["n_search"]}/{b["n_total"]}', b["n_total"]),
            cell(f'{b["budget"]:,}', b["budget"]), cell(f'{b["e_tol"]:.2f}', b["e_tol"]),
            cell(f'{b["e_ref"]:.3f}' + (" &dagger;" if b.get("holes") else ""), b["e_ref"]),
            cell(esc(b["found_by"]), cls="l"),
            cell(esc(b["fastest"]), cls="l nw"), cell(*fmt_ert(b["best_ert"])), cell(*fmt_ert(b["bh_ert"])),
            cell(f'{b["n_solved"]}/{R["na"]}', b["n_solved"]),
            cell(f'{b["dmin"]:.2f}', b["dmin"])])
    gal = "".join(f'<div><img src="{b["img"]}" alt="{esc(b["name"])}" loading="lazy"><br><b>{esc(b["name"])}</b>'
                  f'<br>{b["e_ref"]:.3f} eV</div>' for b in R["benches"] if b.get("img"))
    curves = "".join(f'<h4>{t("cat_long").get(c, c)}</h4><img class="fig" src="{p}" alt="{c}" loading="lazy">'
                     for c, p in R["curves"])
    hole = lambda b: t("hole_note").format(n=b["holes"], e=b["hole_e"],  # noqa: E731
                                           algs=(", ", "、")[k].join(b["hole_algs"]))
    desc = "".join(f'<li><b>{esc(b["name"])}</b> &ndash; {esc(b["title"])}. {prose(b.get("description", ""))}'
                   + (f' <i>[{esc(b["guard"])}]</i>' if b.get("guard") else "")
                   + (f' <i>[{hole(b)}]</i>' if b.get("holes") else "") + '</li>'
                   for b in R["benches"] if b.get("description"))
    desc = f'<details><summary>{t("bench_desc")}</summary><ul class="desc">{desc}</ul></details>' if desc else ""
    out.append(f'<section id="benchmarks"><h2>{t("nav")[7]}</h2><p>{t("bench_intro")}</p>'
               + table(t("bench_cols"), rows, left=(0, 1, 2, 7, 8), tall=True) + desc
               + f'<h3>{t("gallery")}</h3><div class="gallery">{gal}</div>'
               f'<h3>{t("curves")}</h3>{curves}</section>')
    # algorithms
    rows = []
    for a in sorted(R["algs"], key=lambda a: (a["type"], a["name"])):
        rows.append([cell(esc(a["name"]), cls="l"), cell(esc(a["key"]), cls="l"),
                     cell(f'<span class="badge {a["type"]}">{t("type")[a["type"]]}</span>'),
                     cell(a["desc"][k], cls="l"), cell(a["ref"], cls="l")])
    out.append(f'<section id="algorithms"><h2>{t("nav")[8]}</h2>'
               + table(t("alg_cols"), rows, left=(0, 1, 3, 4)) + "</section>")
    out.append(f'<section id="methods"><h2>{t("nav")[9]}</h2>{METHODS[k].format(**R)}</section>')
    dl = "".join(f'<li><a href="{f}">{f}</a> &ndash; {d}</li>' for f, d in t("dl_items"))
    if R.get("repo"):
        rel = R["repo"].rstrip("/") + "/releases"
        dl += f'<li><a href="{rel}">{rel}</a> &ndash; {t("dl_release")}</li>'
    out.append(f'<section id="downloads"><h2>{t("downloads")}</h2><ul>{dl}</ul></section>')
    out.append(f"</main><footer>{t('generated').format(**R)}</footer><script>{JS}</script></body></html>")
    doc = "\n".join(out)
    if lang == "zh":
        # line breaks inside Chinese source strings would render as spaces between characters
        doc = _CJK_PUNCT_GAP.sub("", _CJK_GAP.sub("", doc))
    return doc
