"""Vercel ASGI entrypoint.

Vercel forwards requests under /api/* to this module. The regular development
application remains mounted without changing its local route contract.
"""

import os

os.environ.setdefault("HF_HOME", "/tmp/aegis/huggingface")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/aegis/matplotlib")

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

from service.main import app as core_app  # noqa: E402

app = FastAPI(title="AEGIS Vercel Gateway")


@app.get("/", response_class=HTMLResponse)
def gateway_status():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AEGIS — Agent Runtime</title>
  <style>
    :root{color-scheme:dark;--bg:#07110f;--panel:#0d1c19;--line:#20433a;
      --text:#e8fff8;--muted:#8eb5aa;--accent:#55f2b0;--warn:#ffbf69}
    *{box-sizing:border-box} body{margin:0;font:15px/1.5 Inter,system-ui,sans-serif;
      background:radial-gradient(circle at 80% 0,#12382e 0,transparent 38%),var(--bg);
      color:var(--text)} main{max-width:1100px;margin:auto;padding:48px 22px 80px}
    nav{display:flex;justify-content:space-between;align-items:center;margin-bottom:70px}
    .brand{font-weight:800;letter-spacing:.18em}.live{color:var(--accent);font-size:13px}
    h1{font-size:clamp(42px,8vw,82px);line-height:.95;letter-spacing:-.055em;
      max-width:850px;margin:0 0 24px}.lead{color:var(--muted);max-width:700px;font-size:18px}
    a{color:var(--text);text-decoration:none} nav a{margin-left:22px;color:var(--muted)}
    .grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:36px 0}
    .card{background:color-mix(in srgb,var(--panel) 92%,transparent);border:1px solid var(--line);
      border-radius:16px;padding:20px}.label{color:var(--muted);font-size:12px;
      text-transform:uppercase;letter-spacing:.11em}.value{font-size:28px;font-weight:750;margin-top:7px}
    button{border:0;border-radius:12px;background:var(--accent);color:#052019;
      padding:14px 20px;font-weight:800;cursor:pointer;font-size:15px}
    button:disabled{opacity:.55;cursor:wait}.demo{margin-top:68px}.demo-head{
      display:flex;justify-content:space-between;gap:20px;align-items:end;margin-bottom:18px}
    h2{font-size:30px;margin:0}.result{display:none}.result.show{display:block}
    .wide{grid-column:span 2}.trace{color:var(--muted);word-break:break-word}
    pre{white-space:pre-wrap;color:#b9d8d0;font:13px/1.55 ui-monospace,monospace;margin:8px 0 0}
    .pending{color:var(--warn)} footer{margin-top:50px;color:var(--muted);font-size:13px}
    @media(max-width:760px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}
      nav .links{display:none}.demo-head{align-items:start;flex-direction:column}}
  </style>
</head>
<body><main>
  <nav><div class="brand">AEGIS</div><div class="links">
    <span class="live">● LIVE</span><a href="/api/docs">API Docs</a>
    <a href="https://github.com/inayatarshad/Aegis">GitHub</a></div></nav>
  <section>
    <h1>Evidence before escalation.</h1>
    <p class="lead">An uncertainty-aware LangGraph workflow for synthetic UAV
      telemetry—combining calibrated classification, faithful local attribution,
      policy retrieval, geographic context, and an explicit human-review gate.</p>
    <div class="grid">
      <div class="card"><div class="label">Workflow</div><div class="value">9 nodes</div></div>
      <div class="card"><div class="label">Stress accuracy</div><div class="value">50%</div></div>
      <div class="card"><div class="label">Safety state</div><div class="value">Human gated</div></div>
    </div>
  </section>
  <section class="demo">
    <div class="demo-head"><div><div class="label">Live inference</div>
      <h2>Run a synthetic contact</h2></div>
      <button id="run" onclick="runDemo()">Run analysis</button></div>
    <div id="status" class="card trace">Ready. First cold run may take several seconds.</div>
    <div id="result" class="result grid">
      <div class="card"><div class="label">Threat</div><div id="threat" class="value"></div></div>
      <div class="card"><div class="label">Confidence</div><div id="confidence" class="value"></div></div>
      <div class="card"><div class="label">Review</div><div id="review" class="value"></div></div>
      <div class="card wide"><div class="label">Model explanation</div><pre id="explanation"></pre></div>
      <div class="card"><div class="label">Latency</div><div id="latency" class="value"></div></div>
      <div class="card wide"><div class="label">Doctrine evidence</div><pre id="doctrine"></pre></div>
      <div class="card"><div class="label">Alert dispatched</div><div id="alert" class="value"></div></div>
      <div class="card wide"><div class="label">Agent trace</div><pre id="trace"></pre></div>
    </div>
  </section>
  <footer>Synthetic research prototype · Not an operational ISR or safety system</footer>
</main>
<script>
const sample={scenario_id:"PUBLIC-DEMO-001",timestamp:new Date().toISOString(),
 latitude:33.6844,longitude:73.0479,altitude_m:65,speed_kmh:72,heading_deg:270,
 flight_pattern_entropy:.61,proximity_to_restricted_km:1.8,iff_signal:false,
 estimated_wingspan_m:1.1,loiter_detected:true,rapid_altitude_change:false,
 mission_narrative:"Unidentified contact circling near the protected boundary."};
async function runDemo(){
 const b=document.getElementById("run"),s=document.getElementById("status");
 b.disabled=true;b.textContent="Analyzing…";s.textContent="Executing the agent graph…";
 try{const r=await fetch("/api/analyze",{method:"POST",headers:{"Content-Type":"application/json"},
   body:JSON.stringify({...sample,timestamp:new Date().toISOString()})});
   if(!r.ok)throw new Error("API returned "+r.status);const d=await r.json();
   threat.textContent=d.threat_level;confidence.textContent=(d.confidence*100).toFixed(1)+"%";
   review.textContent=d.review_status;review.className="value "+(d.review_status==="PENDING"?"pending":"");
   explanation.textContent=d.xai_summary;doctrine.textContent=d.doctrine_reference;
   latency.textContent=Math.round(d.processing_latency_ms)+" ms";
   alert.textContent=d.alert_dispatched?"YES":"NO";trace.textContent=d.agent_trace.join(" → ");
   document.getElementById("result").classList.add("show");
   s.textContent="Analysis complete · Pipeline "+d.pipeline_version;
 }catch(e){s.textContent="Analysis failed: "+e.message}
 finally{b.disabled=false;b.textContent="Run again"}
}
</script></body></html>"""


app.mount("/api", core_app)
