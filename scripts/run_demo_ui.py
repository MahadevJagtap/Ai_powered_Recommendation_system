"""Run a dependency-free browser demonstration backed by RecommendationService."""

from __future__ import annotations

import argparse
import html
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from recommender.data import validate_dataset
from recommender.posters import PosterCatalog
from recommender.service import RecommendationService
from recommender.split import chronological_split


def build_service(dataset_root: Path) -> RecommendationService:
    validate_dataset(dataset_root)
    ratings = pd.read_csv(
        dataset_root / "ratings.csv",
        usecols=["userId", "movieId", "rating", "timestamp"],
    )
    movies = pd.read_csv(dataset_root / "movies.csv")
    return RecommendationService().fit(chronological_split(ratings).train, movies)


def load_evaluation_artifacts(results_root: Path) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    baseline_path = results_root / "baseline.json"
    if baseline_path.exists():
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        artifacts["full_dataset_baseline"] = {
            "label": "Full-dataset baseline validation",
            "scope": "Full MovieLens 25M chronological validation split",
            "dataset": payload.get("dataset", {}),
            "split": payload.get("split", {}),
            "metrics": payload.get("metrics", {}),
        }
    bounded_path = results_root / "shared_uniform_2k" / "experiment.json"
    if bounded_path.exists():
        payload = json.loads(bounded_path.read_text(encoding="utf-8"))
        methods = {}
        for name, metrics in payload.get("methods", {}).items():
            methods[name] = {
                key: metrics.get(key)
                for key in (
                    "users_evaluated",
                    "users_with_relevant_items",
                    "candidate_catalog_size",
                    "catalog_coverage",
                    "recall_at_10",
                    "recall_at_50",
                    "ndcg_at_10",
                    "recall_at_10_ci95",
                    "recall_at_50_ci95",
                    "ndcg_at_10_ci95",
                )
            }
        artifacts["bounded_comparison"] = {
            "label": "Bounded uniform-user comparison",
            "scope": "Diagnostic only; 2,000 sampled source users and 1,000 evaluation users per seed",
            "experiment": payload.get("experiment", {}),
            "dataset": payload.get("dataset", {}),
            "split": payload.get("split", {}),
            "methods": methods,
            "runtime_seconds": payload.get("runtime_seconds"),
            "memory_observation": payload.get("memory_observation"),
        }
    return artifacts


def make_handler(
    service: RecommendationService,
    *,
    results_root: Path = Path("results"),
    poster_catalog: PosterCatalog | None = None,
) -> type[BaseHTTPRequestHandler]:
    posters = poster_catalog or PosterCatalog()
    evaluation = load_evaluation_artifacts(results_root)
    hero_movies = (
        (79132, "INCEPTION", "Inception (2010)"),
        (109487, "INTERSTELLAR", "Interstellar (2014)"),
        (2571, "THE MATRIX", "Matrix, The (1999)"),
        (4973, "AMÉLIE", "Amélie (2001)"),
    )
    hero_cards = []
    for movie_id, label, title in hero_movies:
        poster = posters.resolve(movie_id, title)
        image = (
            f'<img src="{html.escape(poster["poster_url"], quote=True)}" '
            f'alt="{html.escape(title, quote=True)} poster" loading="eager" '
            'onerror="this.remove()">'
            if poster.get("poster_url")
            else ""
        )
        hero_cards.append(
            f'<div class="hero-poster">{image}<span>{html.escape(label)}</span></div>'
        )
    hero_cards_markup = "".join(hero_cards)

    class Handler(BaseHTTPRequestHandler):
        def _write(self, status: int, body: str, content_type: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                document = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1"><title>Reelwise | Movie intelligence</title>
<style>
:root{color-scheme:dark;--ink:#f5f7ff;--muted:#a6abc8;--line:#27345d;
--panel:rgba(12,20,53,.88);--accent:#b957ff;--accent2:#624dff;--cyan:#27c8ff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);
font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;
background:#040718;overflow-x:hidden}body:before{content:"";position:fixed;
inset:0;z-index:-2;background:radial-gradient(circle at 78% 2%,#1d2258 0,transparent 34%),
radial-gradient(circle at 7% 30%,#241153 0,transparent 32%),#040718}
.shell{width:min(1180px,calc(100% - 40px));margin:auto;padding:18px 0 64px}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:20px}
.brand{display:flex;align-items:center;gap:12px}.mark{display:grid;place-items:center;
width:30px;height:30px;border-radius:50%;color:#fff;font-weight:900;
background:linear-gradient(135deg,#c356ff,#6048ff);box-shadow:0 0 20px #8c48ff77}
.brand strong{display:block;font-size:16px}.brand span{display:none}
.brand span,.eyebrow,.hint{color:var(--muted);font-size:12px}.status{display:flex;gap:8px;
align-items:center;color:#b7c4df;font-size:12px}.dot{width:8px;height:8px;border-radius:50%;
background:#43e6a7;box-shadow:0 0 12px #43e6a7}
.nav{display:flex;align-items:center;gap:28px;margin-left:auto;margin-right:28px;color:#aeb4d0;font-size:12px}
.nav a{color:inherit;text-decoration:none}.nav a:first-child{color:#fff;border-bottom:2px solid var(--accent);padding:0 6px 9px}
.hero{display:grid;grid-template-columns:44% 56%;min-height:285px;padding:54px 0 25px;align-items:center}
.hero-copy{position:relative;z-index:2}.eyebrow{color:#e2c4ff;letter-spacing:.16em;
text-transform:uppercase;font-weight:700}.hero h1{font-size:clamp(42px,7vw,78px);line-height:.98;
letter-spacing:-.06em;margin:13px 0 14px}.hero h1 em{font-style:normal;color:#c15bff}
.hero p{color:#bac0d7;font-size:14px;max-width:450px;margin:0}.hero-art{height:285px;position:relative}
.hero-art:after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,#040718 0%,transparent 35%,#040718aa 100%)}
.hero-poster{position:absolute;width:145px;height:220px;border-radius:8px;border:1px solid #ffffff44;
box-shadow:0 15px 35px #0008;display:grid;align-items:end;overflow:hidden;font-weight:800;font-size:18px;
background:linear-gradient(145deg,#394d83,#111633)}.hero-poster:nth-child(1){right:370px;top:28px;transform:rotate(-13deg)}
.hero-poster:nth-child(2){right:245px;top:0;transform:rotate(7deg);background:linear-gradient(145deg,#47709b,#142144)}
.hero-poster:nth-child(3){right:112px;top:25px;transform:rotate(14deg);background:linear-gradient(145deg,#305e67,#10182e)}
.hero-poster:nth-child(4){right:0;top:6px;transform:rotate(20deg);background:linear-gradient(145deg,#8b5136,#1b1830)}
.hero-poster img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:block}
.hero-poster:after{content:"";position:absolute;inset:35% 0 0;z-index:1;
background:linear-gradient(transparent,#08102ce8);pointer-events:none}
.hero-poster span{position:absolute;left:12px;right:8px;bottom:13px;z-index:2;color:#f5f7ff;
text-shadow:0 2px 5px #000;font-weight:800}
.control{border:1px solid #334374;background:linear-gradient(120deg,#111d43e8,#101638e8);
border-radius:24px;padding:22px;box-shadow:0 20px 70px #0005}.control-head{display:flex;
justify-content:space-between;gap:20px;margin-bottom:18px}.control h2{margin:0;font-size:18px}
.control-head p{margin:3px 0 0;color:var(--muted);font-size:13px}.fields{display:grid;
grid-template-columns:1fr 1.5fr .7fr auto;gap:13px;align-items:end}label{display:grid;
gap:7px;color:#cbd5ee;font-size:12px;font-weight:700}input,select{width:100%;border:1px solid #304363;
border-radius:11px;background:#0c1830;color:var(--ink);padding:12px;font:inherit;outline:none}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px #b957ff33}
button{border:0;border-radius:9px;padding:12px 20px;color:#fff;background:linear-gradient(100deg,#bf51ef,#5a4cff);
font-weight:800;font:inherit;cursor:pointer;transition:transform .2s,box-shadow .2s}
button:hover{transform:translateY(-2px);box-shadow:0 8px 26px #69e5c440}
button:disabled{opacity:.65;cursor:wait;transform:none}.notice{min-height:24px;margin:16px 2px 0;
color:var(--muted);font-size:13px}.notice.error{color:#ff9eae}.loader{display:none;gap:9px;
align-items:center}.loader.show{display:flex}.spinner{width:15px;height:15px;border:2px solid #ffffff33;
border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.results-head{display:flex;align-items:end;justify-content:space-between;margin:32px 0 17px}
.results-head h2{margin:0;font-size:24px}.results-head p{margin:0;color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}
.card{display:flex;flex-direction:column;height:100%;border:1px solid #253560;background:var(--panel);border-radius:10px;
animation:rise .45s both;transition:transform .2s,border-color .2s}.card:hover{transform:translateY(-5px);
border-color:#69e5c488}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.poster{aspect-ratio:2/3;display:grid;place-items:center;background:linear-gradient(135deg,#243a65,#101b34);
position:relative;overflow:hidden;border-radius:9px 9px 0 0}.poster:after{content:"";position:absolute;
inset:0;background:linear-gradient(transparent,#0d1830)}.poster span{z-index:1;border:1px solid #ffffff35;
border-radius:50%;padding:14px;color:#d8e0f5;font-size:24px}.poster-fallback{display:grid}
.poster img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .35s}
.card:hover .poster img{transform:scale(1.06)}.poster img~.poster-fallback{display:none}
.card-body{padding:13px 12px 14px;background:#0c1830;border-radius:0 0 9px 9px;
min-height:105px;flex:1;overflow:visible}.card h3{margin:0;font-size:13px;line-height:1.3;
font-weight:750;color:#f5f7ff;overflow-wrap:anywhere}.movie-title{display:block}.year{display:block;
margin-top:4px;color:#d39aff;font-size:12px;font-weight:700}.genres{margin-top:7px;
color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.empty{display:grid;place-items:center;min-height:180px;
border:1px dashed #304363;border-radius:18px;color:var(--muted);text-align:center;padding:25px}
.methodology{display:grid;grid-template-columns:1.2fr 1.2fr 1fr;gap:16px;margin-top:26px}.info{padding:20px;
border:1px solid var(--line);border-radius:18px;background:#0d1930aa}.info h3{margin:0 0 8px;font-size:15px}
.info p{color:var(--muted);font-size:13px;margin:0}.footer{margin-top:35px;color:#687898;font-size:12px}
.why{margin-top:34px}.why h2,.evaluation h2{font-size:24px;margin:0 0 6px;letter-spacing:-.02em}.why p,.evaluation p{color:var(--muted);font-size:13px}
.insight{border:1px solid var(--line);border-radius:16px;background:#0d1930aa;padding:20px;margin-top:14px}
.insight strong{display:block;margin-bottom:7px;color:#f4f1ff}.insight ul{margin:10px 0 0;padding-left:20px;color:#c5cbe0;font-size:13px}
.insight li{padding:3px 0}.evaluation{margin-top:48px}.evaluation>p{max-width:760px;line-height:1.65}
.eval-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}
.kpi{border:1px solid #334374;border-radius:14px;padding:16px;background:linear-gradient(145deg,#171e4b,#0d1632)}
.kpi small{display:block;color:#bfc5e4;font-size:11px;letter-spacing:.04em;text-transform:uppercase}.kpi b{display:block;font-size:27px;margin-top:4px}
.eval-table{width:100%;border-collapse:collapse;margin-top:20px;font-size:12px}.eval-table th,.eval-table td{padding:11px 10px;border-bottom:1px solid #263653;text-align:left}.eval-table th{color:#c6cbee;font-size:11px;letter-spacing:.04em;text-transform:uppercase}.eval-table td{color:#d8ddf2}
@media(max-width:900px){.hero{grid-template-columns:1fr}.hero-art{display:none}.cards{grid-template-columns:repeat(3,1fr)}}
@media(max-width:760px){.shell{width:min(100% - 26px,560px);padding-top:18px}.hero{padding:58px 0 30px}
.fields{grid-template-columns:1fr 1fr}.fields label:first-child{grid-column:span 2}.fields button{grid-column:span 2}
.topbar .status{font-size:0}.nav{display:none}.methodology{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}.results-head{margin-top:38px}.eval-grid{grid-template-columns:1fr 1fr}.eval-table{display:block;overflow-x:auto;white-space:nowrap}}
</style></head><body><main class="shell">
<header class="topbar"><div class="brand"><div class="mark">R</div><div><strong>Reelwise</strong>
<span>Personalized movie intelligence</span></div></div><nav class="nav"><a href="#home">Home</a>
<a href="#methods">How it works</a><a href="#methods">About</a></nav><div class="status"><i class="dot"></i>
System Online</div></header>
<section class="hero" id="home"><div class="hero-copy"><div class="eyebrow">AI powered</div>
<h1>Find your next<br><em>favorite story.</em></h1><p>Explore recommendations generated
from the MovieLens catalog using leakage-safe, training-only interaction histories.</p></div>
<div class="hero-art"><!-- HERO_CARDS --></div></section>
<section class="control"><div class="control-head"><div><h2>Build a watchlist</h2>
<p>Choose a retrieval strategy and tune the size of your slate.</p></div><span class="hint">MAX 100</span></div>
<form id="form"><div class="fields"><label for="user">USER ID<input id="user" type="number"
value="1" min="1" required></label><label for="method">RECOMMENDATION ENGINE<select id="method">
<option value="retrieval">NumPy co-occurrence · personalized</option>
<option value="baseline">Popularity baseline · trusted fallback</option></select></label>
<label for="k">MOVIES<input id="k" type="number" value="10" min="1" max="100" required></label>
<button id="submit" type="submit">Recommend <span>→</span></button></div>
<div id="notice" class="notice"><span id="loader" class="loader"><i class="spinner"></i>Curating your slate…</span>
<span id="message">Recommendations exclude movies already seen by this user.</span></div></form></section>
<section><div class="results-head"><h2>Your selection</h2><p id="meta">Ready when you are</p></div>
<div id="results" class="cards"><div class="empty"><div><strong>Your next favorites are waiting.</strong>
<br>Enter a user ID and start the discovery.</div></div></div></section>
<section class="why"><h2>Why These Movies?</h2><p>Transparent signals from the selected recommendation engine.</p>
<div id="insights"><div class="insight"><strong>Ready for a recommendation</strong>
Choose an engine above to see the supporting algorithm signal.</div></div></section>
<section class="methodology" id="methods"><article class="info"><h3>✦ Personalized co-occurrence</h3>
<p>Connects positive training interactions through item-to-item patterns, then filters your seen history.
Unknown or sparse users fall back to popularity.</p></article><article class="info"><h3>▥ Popularity baseline</h3>
<p>A transparent benchmark ranked by positive training interactions. Offline metrics are evaluation signals,
not ratings or explanations of live recommendations.</p></article></section>
<section class="evaluation" id="evaluation"><h2>Model Evaluation</h2>
<p>Read-only results from saved offline experiments. Bounded comparisons are diagnostic and are not equivalent to the full-dataset baseline.</p>
<div id="evaluation-content"><div class="empty">Loading saved evaluation artifacts…</div></div></section>
<div class="footer">Reelwise · MovieLens 25M · Local portfolio demonstration</div></main>
<script>
const form=document.getElementById("form"),userField=document.getElementById("user"),
methodField=document.getElementById("method"),countField=document.getElementById("k"),
results=document.getElementById("results"),message=document.getElementById("message"),
meta=document.getElementById("meta"),loader=document.getElementById("loader"),
submit=document.getElementById("submit");
const escapeHtml=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",
">":"&gt;","\"":"&quot;","'":"&#039;"}[c]));
const yearOf=title=>{const match=String(title??"").match(/\s*\((\d{4})\)\s*$/);return match?match[1]:"";};
const titleOf=title=>String(title??"").replace(/\s*\(\d{4}\)\s*$/,"").trim()||"Untitled movie";
const posterInitial=title=>escapeHtml(String(title??"?").trim().charAt(0).toUpperCase()||"?");
const formatMetric=value=>typeof value==="number"?value.toFixed(4):"Unavailable";
const titleById=items=>new Map(items.map(item=>[String(item.movieId),String(item.title??"")]));
const renderEvaluation=async()=>{const host=document.getElementById("evaluation-content");
try{const response=await fetch("/evaluation");const data=await response.json();if(!response.ok)throw new Error(data.error||"Unavailable");
const full=data.full_dataset_baseline,bounded=data.bounded_comparison;let html="";
if(full){const m=full.metrics||{};html+=`<div class="insight"><strong>${escapeHtml(full.label)}</strong>
<div class="eval-grid"><div class="kpi"><small>Recall@10</small><b>${formatMetric(m.recall_at_10)}</b></div>
<div class="kpi"><small>Recall@50</small><b>${formatMetric(m.recall_at_50)}</b></div>
<div class="kpi"><small>NDCG@10</small><b>${formatMetric(m.ndcg_at_10)}</b></div></div>
<p>${escapeHtml(full.scope||"")}; users evaluated: ${m.users_evaluated??"Unavailable"}.</p></div>`}
if(bounded){const rows=Object.entries(bounded.methods||{}).map(([name,m])=>`<tr><td>${escapeHtml(name.replaceAll("_"," "))}</td>
<td>${formatMetric(m.recall_at_10)}</td><td>${formatMetric(m.recall_at_50)}</td><td>${formatMetric(m.ndcg_at_10)}</td>
<td>${m.users_evaluated??"Unavailable"}</td></tr>`).join("");
html+=`<div class="insight"><strong>${escapeHtml(bounded.label)}</strong><p>${escapeHtml(bounded.scope||"")}</p>
<table class="eval-table"><thead><tr><th>Method</th><th>R@10</th><th>R@50</th><th>NDCG@10</th><th>Users</th></tr></thead>
<tbody>${rows||"<tr><td colspan=\"5\">No method metrics saved.</td></tr>"}</tbody></table></div>`}
host.innerHTML=html||'<div class="empty">No saved evaluation artifacts are available.</div>'}
catch(error){host.innerHTML=`<div class="empty">${escapeHtml(error.message)}</div>`}};
renderEvaluation();
form.onsubmit=async event=>{event.preventDefault();const userId=userField.value.trim(),count=countField.value.trim();
if(!userId||!count||Number(count)<1||Number(count)>100){message.textContent="Use a user ID and a movie count from 1 to 100.";
message.className="error";return}submit.disabled=true;loader.classList.add("show");message.textContent="";
message.className="";meta.textContent="Curating…";results.innerHTML="";
try{const query=new URLSearchParams({user_id:userId,k:count,method:methodField.value});
const response=await fetch("/recommendations?"+query);const data=await response.json();
if(!response.ok)throw new Error(data.error||"Unable to load recommendations.");
const items=data.recommendations||[];meta.textContent=items.length+" movies · "+data.source.replaceAll("_"," ");
if(!items.length){results.innerHTML='<div class="empty"><div>No eligible movies found for this user.</div></div>'}
else{results.innerHTML=items.map((item,index)=>`<article class="card" style="animation-delay:${index*35}ms">
<div class="poster">${item.poster_url?`<img src="${escapeHtml(item.poster_url)}" alt="Poster for ${escapeHtml(item.title)}" loading="lazy" onerror="this.remove();this.nextElementSibling.style.display='grid'">`:""}<span class="poster-fallback">${posterInitial(item.title)}</span></div><div class="card-body">
<h3><span class="movie-title">${escapeHtml(titleOf(item.title))}</span>${yearOf(item.title)?`<span class="year">${yearOf(item.title)}</span>`:`<span class="year">Year unavailable</span>`}</h3>
<div class="genres">${escapeHtml(item.genres||"Genre unavailable")}</div></div></article>`).join("")}
const titles=titleById(items), labelForId=id=>titles.get(String(id))||`MovieLens ID ${id}`;
const insightItems=(data.explanations||[]).filter(item=>item.contributors?.length);
document.getElementById("insights").innerHTML=insightItems.length?
`<div class="insight"><strong>Personalized co-occurrence signal</strong><ul>${insightItems.slice(0,5).map(item=>
`<li><b>${escapeHtml(labelForId(item.movieId))}</b> is connected to ${item.contributors.map(c=>escapeHtml(c.title)+" (score "+c.score+")").join(", ")}.</li>`).join("")}</ul></div>`:
data.method==="baseline"?
`<div class="insight"><strong>Popularity baseline</strong><ul>${(data.explanations||[]).slice(0,5).map(item=>
`<li><b>${escapeHtml(labelForId(item.movieId))}</b> has ${item.positive_interactions} positive training interactions.</li>`).join("")}</ul></div>`:
`<div class="insight"><strong>Algorithm-level explanation</strong>
<p>Recommendations are derived from item-to-item relationships in positive training interactions. Detailed contributors were not available for this result.</p></div>`}
catch(error){meta.textContent="Could not load";message.textContent=error.message;message.className="error";
results.innerHTML='<div class="empty"><div>Try another user or method.</div></div>'}
finally{submit.disabled=false;loader.classList.remove("show")}};
</script></body></html>"""
                document = document.replace("<!-- HERO_CARDS -->", hero_cards_markup)
                self._write(200, document, "text/html; charset=utf-8")
                return
            if parsed.path == "/health":
                self._write(200, json.dumps({"status": "ok"}), "application/json")
                return
            if parsed.path == "/recommendations":
                try:
                    user_id = int(query.get("user_id", [""])[0])
                    k = int(query.get("k", ["10"])[0])
                    method = query.get("method", ["retrieval"])[0]
                    result = service.recommend(user_id, k=k, method=method)
                    recommendations = []
                    for recommendation in result.recommendations:
                        item = dict(recommendation)
                        item.update(posters.resolve(item["movieId"], str(item.get("title", ""))))
                        recommendations.append(item)
                    self._write(
                        200,
                        json.dumps(
                            {
                                "user_id": user_id,
                                "method": method,
                                "source": result.source,
                                "model_version": result.model_version,
                                "recommendations": recommendations,
                                "explanations": result.explanations,
                            }
                        ),
                        "application/json",
                    )
                except (ValueError, RuntimeError) as exc:
                    self._write(400, json.dumps({"error": str(exc)}), "application/json")
                return
            if parsed.path == "/evaluation":
                self._write(200, json.dumps(evaluation), "application/json")
                return
            self._write(404, json.dumps({"error": "not found"}), "application/json")

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data") / "ml-25m")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(
            build_service(args.dataset_root),
            poster_catalog=PosterCatalog(
                links_path=args.dataset_root / "links.csv"
            ),
        ),
    )
    print(f"Recommendation UI listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
