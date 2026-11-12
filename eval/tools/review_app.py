"""
Local review page for benchmark questions.

    python -m eval.tools.review_app --file eval/datasets/checked_pilot.jsonl
    # then open http://127.0.0.1:8765

Shows each question that needs a human look next to the actual PDF page (rendered
image) or transcript window, with Keep / Edit / Reject. Decisions are written back
into the same file under meta.review, so review can stop and resume any time.

What needs review (the plan's rule):
  - every question the cross-check flagged
  - every unanswerable and multi_hop question
  - a random sample (--sample, default 25) of the remaining auto-passed questions
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone

import fitz
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from eval.corpus.download import pdf_path
from eval.corpus.text import evidence_text, sources
from eval.datasets.schema import dump, load, validate

ALWAYS_REVIEW = {"unanswerable", "multi_hop"}


def needs_review(records: list[dict], sample: int, seed: int) -> list[str]:
    must, rest = [], []
    for r in records:
        cc = r.get("meta", {}).get("crosscheck", {})
        if cc.get("status") == "fail":
            continue
        if cc.get("status") != "pass" or r["category"] in ALWAYS_REVIEW:
            must.append(r["id"])
        else:
            rest.append(r["id"])
    rng = random.Random(seed)
    return must + rng.sample(rest, min(sample, len(rest)))


def build_app(path: str, sample: int, seed: int) -> FastAPI:
    app = FastAPI()
    state = {"records": load(path)}
    queue = needs_review(state["records"], sample, seed)
    meta = sources()

    def by_id(qid):
        for r in state["records"]:
            if r["id"] == qid:
                return r
        raise HTTPException(404)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get("/api/items")
    def items():
        out = []
        for qid in queue:
            r = by_id(qid)
            out.append({**r, "source_title": meta[r["doc"]]["title"],
                        "source_type": meta[r["doc"]]["type"],
                        "evidence_text": [evidence_text(ev) for ev in r["gold_evidence"]
                                          if meta[ev["doc"]]["type"] == "video"]})
        return out

    @app.get("/api/page/{doc}/{page}.png")
    def page_png(doc: str, page: int):
        if doc not in meta or meta[doc]["type"] != "pdf":
            raise HTTPException(404)
        with fitz.open(pdf_path(doc)) as pdf:
            if not 1 <= page <= pdf.page_count:
                raise HTTPException(404)
            png = pdf[page - 1].get_pixmap(dpi=110).tobytes("png")
        return Response(png, media_type="image/png")

    @app.post("/api/decision")
    def decision(body: dict = Body(...)):
        r = by_id(body["id"])
        action = body["action"]
        if action not in ("keep", "edit", "reject"):
            raise HTTPException(400, "action must be keep/edit/reject")
        if action == "edit":
            for k in ("question", "gold_answer"):
                if body.get(k):
                    r[k] = body[k].strip()
            if body.get("pages"):
                pages = [int(p) for p in str(body["pages"]).replace(" ", "").split(",") if p]
                r["gold_evidence"] = [{"doc": r["doc"], "pages": pages}]
            errs = validate(r, set(meta))
            if errs:
                raise HTTPException(400, "; ".join(errs))
        r.setdefault("meta", {})["review"] = {
            "decision": action, "by": "human", "note": body.get("note", ""),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        dump(state["records"], path)
        return {"ok": True}

    return app


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>SourceMind question review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f6f5f1;color:#222}
header{padding:10px 18px;background:#222;color:#fff;display:flex;gap:16px;align-items:center}
main{display:grid;grid-template-columns:1.1fr 1fr;gap:16px;padding:16px}
.panel{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0002}
img{max-width:100%;border:1px solid #ddd;margin-bottom:8px}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;background:#eee;margin-right:6px;font-size:12px}
.flag{background:#ffe1a8}.pass{background:#c8f0c8}
textarea,input{width:100%;box-sizing:border-box;font:inherit;padding:6px;margin:4px 0 10px}
button{font:inherit;padding:8px 16px;margin-right:8px;border:0;border-radius:6px;cursor:pointer}
.keep{background:#2e7d32;color:#fff}.edit{background:#1565c0;color:#fff}.reject{background:#c62828;color:#fff}
pre{white-space:pre-wrap;background:#fafafa;padding:8px;border:1px solid #eee;max-height:420px;overflow:auto}
.done{opacity:.6}
</style></head><body>
<header><b>Question review</b><span id="progress"></span>
<button onclick="go(-1)">◀ Prev</button><button onclick="go(1)">Next ▶</button>
<label style="font-size:13px"><input type="checkbox" id="skipdone" checked style="width:auto"> skip reviewed</label></header>
<main><div class="panel" id="left"></div><div class="panel" id="right"></div></main>
<script>
let items=[],i=0;
async function load(){items=await (await fetch('/api/items')).json();
  i=items.findIndex(x=>!(x.meta||{}).review); if(i<0)i=0; render();}
function esc(s){return (s??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function go(d){const skip=document.getElementById('skipdone').checked;let j=i;
  do{j+=d}while(skip&&j>=0&&j<items.length&&(items[j].meta||{}).review);
  if(j>=0&&j<items.length){i=j;render();}}
function render(){const r=items[i]; if(!r){document.getElementById('left').innerHTML='Nothing to review.';return;}
  const done=items.filter(x=>(x.meta||{}).review).length;
  document.getElementById('progress').textContent=`${i+1} / ${items.length}  ·  reviewed ${done}`;
  const cc=(r.meta||{}).crosscheck||{}; const rv=(r.meta||{}).review;
  let left='';
  if(r.source_type==='pdf'){ for(const ev of r.gold_evidence) for(const p of ev.pages)
      left+=`<div><b>${esc(ev.doc)} — page ${p}</b><br><img src="/api/page/${ev.doc}/${p}.png"></div>`;
    if(!r.gold_evidence.length) left+=`<p>Unanswerable: the checker searched the most related pages. Open any page to verify:</p>
      <input id="pp" placeholder="page number"><button onclick="peek()">Show page</button><div id="peek"></div>`;
  } else { left+=r.evidence_text.map(t=>`<pre>${esc(t)}</pre>`).join('') || '<p>Unanswerable (video).</p>'; }
  document.getElementById('left').innerHTML=left;
  const quotes=((r.meta||{}).quotes||[]).map(q=>`<li>${esc(q)}</li>`).join('');
  const hist=(r.history||[]).map(t=>`<div><i>${t.role}:</i> ${esc(t.content)}</div>`).join('');
  document.getElementById('right').innerHTML=`
    <div class="${rv?'done':''}"><span class="tag">${r.category}</span><span class="tag">${esc(r.doc)}</span>
    <span class="tag ${cc.status||''}">check: ${cc.status||'—'} ${(cc.failed_checks||[]).join(', ')}</span>
    ${rv?`<span class="tag">reviewed: ${rv.decision}</span>`:''}</div>
    <p style="color:#666;font-size:13px">${esc(r.source_title)}</p>
    ${hist?`<div style="margin:8px 0;padding:8px;background:#f3f3f3">${hist}</div>`:''}
    <label>Question</label><textarea id="q" rows="3">${esc(r.question)}</textarea>
    <label>Gold answer ${r.answerable?'':'(unanswerable)'}</label><textarea id="a" rows="3">${esc(r.gold_answer||'')}</textarea>
    ${r.source_type==='pdf'&&r.answerable?`<label>Gold pages (comma separated)</label><input id="pg" value="${r.gold_evidence.flatMap(e=>e.pages||[]).join(',')}">`:''}
    ${quotes?`<label>Quotes from generator</label><ul style="font-size:13px">${quotes}</ul>`:''}
    ${cc.issue?`<p style="font-size:13px"><b>Checker:</b> ${esc(cc.issue)}</p>`:''}
    <label>Note (optional)</label><input id="note">
    <button class="keep" onclick="decide('keep')">Keep</button>
    <button class="edit" onclick="decide('edit')">Save edit</button>
    <button class="reject" onclick="decide('reject')">Reject</button>`;}
async function peek(){const r=items[i],p=document.getElementById('pp').value;
  document.getElementById('peek').innerHTML=`<img src="/api/page/${r.doc}/${p}.png">`;}
async function decide(action){const r=items[i];
  const body={id:r.id,action,note:document.getElementById('note').value};
  if(action==='edit'){body.question=document.getElementById('q').value;body.gold_answer=document.getElementById('a').value;
    const pg=document.getElementById('pg'); if(pg) body.pages=pg.value;}
  const res=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!res.ok){alert(await res.text());return;}
  r.meta=r.meta||{}; r.meta.review={decision:action};
  if(action==='edit'){r.question=body.question;r.gold_answer=body.gold_answer;}
  go(1); if((items[i].meta||{}).review) render();}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
  if(e.key==='k')decide('keep'); if(e.key==='r')decide('reject'); if(e.key==='ArrowRight')go(1); if(e.key==='ArrowLeft')go(-1);});
load();
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--sample", type=int, default=25)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    app = build_app(args.file, args.sample, args.seed)
    print(f"review at http://127.0.0.1:{args.port}  (keys: k = keep, r = reject, arrows = navigate)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
