/* ═══════════════════════════════════════════════════════════════
   VolatiLiX · Main.js  — Complete XAI Suite
   9 explanation methods: LIME · DiCE · SHAP · BETA · Sensitivity
                          LRP · Ablation · Permutation · Saliency
═══════════════════════════════════════════════════════════════ */
"use strict";

// Chart registry
const CHARTS = {};

Chart.defaults.color       = "#7b8eab";
Chart.defaults.font.family = "IBM Plex Sans";
Chart.defaults.plugins.legend.display = false;

const GRID   = "rgba(37,47,66,0.7)";
const TT_BG  = "#1a2130";
const TT_BD  = "#252f42";
const C_BLUE = "#4d9de0";
const C_HI   = "#e05c5c";
const C_LO   = "#4dbd96";
const C_GOLD = "#c4a24d";
const C_PURP = "#9b7ed6";
const FEAT_COLORS = [C_BLUE, C_LO, C_HI, C_GOLD, C_PURP];

function tt(extra = {}) {
  return { backgroundColor:TT_BG, titleColor:"#c9d4e8", bodyColor:"#7b8eab",
           borderColor:TT_BD, borderWidth:1, padding:10, ...extra };
}
function mkChart(id, config) {
  if (CHARTS[id]) { CHARTS[id].destroy(); }
  const el = document.getElementById(id);
  if (!el) return;
  CHARTS[id] = new Chart(el, config);
  return CHARTS[id];
}
function $id(id) { return document.getElementById(id); }
function show(id) { const e=$id(id); if(e) e.style.display="block"; }
function hide(id) { const e=$id(id); if(e) e.style.display="none"; }
function text(id, v) { const e=$id(id); if(e) e.textContent=v; }

function barCfg(labels, data, colors, titleY="") {
  return {
    type:"bar",
    data:{ labels, datasets:[{ data, backgroundColor:colors.map(c=>c+"99"),
           borderColor:colors, borderWidth:1, borderRadius:3 }] },
    options:{
      responsive:true, maintainAspectRatio:false,
      plugins:{ tooltip:tt() },
      scales:{
        x:{ grid:{display:false}, border:{color:"var(--border)"}, ticks:{font:{size:11}} },
        y:{ grid:{color:GRID}, border:{color:"var(--border)"},
            ticks:{font:{size:10,family:"IBM Plex Mono"},color:"#3e4f6a"},
            title:{display:!!titleY,text:titleY,color:"#3e4f6a",font:{size:10}} },
      },
      animation:{duration:500},
    },
  };
}

// ── POST helper ───────────────────────────────────────────────────────────────
async function post(url, body) {
  const res  = await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  const text = await res.text();
  let data;
  try { data = JSON.parse(text); }
  catch { throw new Error(`Server error (${res.status}). Check Flask terminal.`); }
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ── Init metadata ─────────────────────────────────────────────────────────────
async function init() {
  try {
    const {meta} = await (await fetch("/api/dates")).json();
    if (!meta) return;
    text("s-rf-acc",   (meta.rf_test_acc*100).toFixed(1)+"%");
    text("s-mlp-rmse", "$"+(meta.mlp_rmse||"—"));
    text("s-beta-r2",  (meta.beta_r2||"—")+"");
  } catch(e) { console.warn("Meta:", e); }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN FLOW
// ═══════════════════════════════════════════════════════════════════════════
async function analyzeDate() {
  const date = $id("date-input").value;
  if (!date) return;
  setBtnState(true);
  hide("error-box"); hide("results"); show("loading");

  try {
    const pred = await post("/api/predict", {date});
    hide("loading");
    renderRF(pred);
    show("results");
    // fire all async — each has own spinner
    fetchMLP(pred.date);
    fetchLIME(pred.date);
    fetchDiCE(pred.date);
    fetchSHAP(pred.date);
    fetchBETA(pred.date);
    fetchSensitivity(pred.date);
    fetchAblation(pred.date);
    fetchSaliency(pred.date);
  } catch(e) {
    hide("loading");
    text("error-msg", e.message||"Prediction failed. Check Flask terminal.");
    show("error-box");
  } finally { setBtnState(false); }
}

// ── Secondary fetches ─────────────────────────────────────────────────────────
async function fetchMLP(date) {
  show("mlp-loading"); hide("mlp-result"); show("spark-loading"); hide("spark-wrap");
  try {
    const d = await post("/api/mlp/predict",{date});
    hide("mlp-loading"); show("mlp-result");
    hide("spark-loading"); show("spark-wrap");
    renderMLP(d); text("mlp-date-badge", d.date);
  } catch(e) {
    hide("mlp-loading"); hide("spark-loading");
    const el=$id("mlp-result");
    if(el){el.innerHTML=`<p style="color:var(--hi);font-size:12px;padding:8px 0;">MLP error: ${e.message}</p>`;show("mlp-result");}
  }
}

async function fetchLIME(date) {
  show("lime-loading"); hide("lime-chart-wrap");
  try {
    const d = await post("/api/explain/lime",{date});
    hide("lime-loading"); show("lime-chart-wrap");
    renderLIME(d.explanations);
  } catch(e) {
    hide("lime-loading");
    const w=$id("lime-chart-wrap");
    if(w){w.innerHTML=`<p style="color:var(--hi);font-size:12px;padding:8px 0;">LIME: ${e.message}</p>`;show("lime-chart-wrap");}
  }
}

async function fetchDiCE(date) {
  show("dice-loading"); hide("dice-content"); hide("dice-err");
  try {
    const d = await post("/api/explain/dice",{date});
    hide("dice-loading");
    text("dice-from", d.original_label); text("dice-to", d.target_label);
    renderDiCE(d); show("dice-content");
  } catch(e) {
    hide("dice-loading");
    const el=$id("dice-err"); if(el){el.textContent=`DiCE: ${e.message}`;show("dice-err");}
  }
}

async function fetchSHAP(date) {
  show("shap-loading"); hide("shap-content"); hide("shap-err");
  try {
    const d = await post("/api/explain/shap",{date});
    hide("shap-loading"); renderSHAP(d); show("shap-content");
  } catch(e) {
    hide("shap-loading");
    const el=$id("shap-err"); if(el){el.textContent=`SHAP: ${e.message}`;show("shap-err");}
  }
}

async function fetchBETA(date) {
  show("beta-loading"); hide("beta-content"); hide("beta-err");
  try {
    const d = await post("/api/explain/beta",{date});
    hide("beta-loading"); renderBETA(d); show("beta-content");
  } catch(e) {
    hide("beta-loading");
    const el=$id("beta-err"); if(el){el.textContent=`BETA: ${e.message}`;show("beta-err");}
  }
}

async function fetchSensitivity(date) {
  show("sens-loading"); hide("sens-content"); hide("sens-err");
  try {
    const d = await post("/api/explain/sensitivity",{date});
    hide("sens-loading"); renderSensitivity(d); show("sens-content");
  } catch(e) {
    hide("sens-loading");
    const el=$id("sens-err"); if(el){el.textContent=`Sensitivity: ${e.message}`;show("sens-err");}
  }
}

async function fetchAblation(date) {
  show("ablation-loading"); hide("ablation-content"); hide("ablation-err");
  try {
    const d = await post("/api/explain/ablation",{date});
    hide("ablation-loading"); renderAblation(d); show("ablation-content");
  } catch(e) {
    hide("ablation-loading");
    const el=$id("ablation-err"); if(el){el.textContent=`Ablation: ${e.message}`;show("ablation-err");}
  }
}

async function fetchSaliency(date) {
  show("saliency-loading"); hide("saliency-content"); hide("saliency-err");
  try {
    const d = await post("/api/explain/saliency",{date});
    hide("saliency-loading"); renderSaliency(d); show("saliency-content");
  } catch(e) {
    hide("saliency-loading");
    const el=$id("saliency-err"); if(el){el.textContent=`Saliency: ${e.message}`;show("saliency-err");}
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// RENDERERS
// ═══════════════════════════════════════════════════════════════════════════

// ── 01 RF ─────────────────────────────────────────────────────────────────────
function renderRF(d) {
  const isHigh = d.prediction === 1;
  text("rf-date-badge", d.date);

  const card=$id("verdict-card"), icon=$id("verdict-icon"),
        label=$id("verdict-label"), ring=$id("cr-fill"), actual=$id("actual-tag");
  card.classList.remove("is-hi","is-lo");
  label.classList.remove("is-hi","is-lo");
  ring.classList.remove("is-hi","is-lo");
  if(isHigh){
    card.classList.add("is-hi"); label.classList.add("is-hi"); ring.classList.add("is-hi");
    icon.textContent="↑"; icon.style.color=C_HI; label.textContent="High Volatility";
  } else {
    card.classList.add("is-lo"); label.classList.add("is-lo"); ring.classList.add("is-lo");
    icon.textContent="↓"; icon.style.color=C_LO; label.textContent="Low Volatility";
  }
  text("conf-num", d.confidence.toFixed(1)+"%");
  setTimeout(()=>{ ring.style.strokeDashoffset = 276.5*(1-d.confidence/100); },80);

  const correct = d.actual===d.prediction;
  actual.textContent = `Actual: ${d.actual===1?"High Vol":"Low Vol"}`;
  actual.style.borderColor = correct?"var(--lo-border)":"var(--hi-border)";
  actual.style.color       = correct?C_LO:C_HI;

  mkChart("prob-chart",{
    type:"doughnut",
    data:{ labels:["Low","High"], datasets:[{data:[d.prob_low,d.prob_high],
      backgroundColor:[C_LO+"bb",C_HI+"bb"],borderColor:[C_LO,C_HI],borderWidth:1,hoverOffset:4}] },
    options:{ cutout:"72%",responsive:true,maintainAspectRatio:false,
      plugins:{tooltip:tt({callbacks:{label:c=>` ${c.parsed.toFixed(1)}%`}})},animation:{duration:700} },
  });
  text("prob-lo", d.prob_low.toFixed(1)+"%");
  text("prob-hi", d.prob_high.toFixed(1)+"%");

  const sorted=Object.entries(d.feature_importances).sort((a,b)=>b[1]-a[1]).slice(0,8);
  const maxI=sorted[0][1];
  $id("rf-imp-list").innerHTML=sorted.map(([n,v])=>`
    <div class="imp-item">
      <span class="imp-name" title="${n}">${n}</span>
      <span class="imp-pct">${(v*100).toFixed(1)}%</span>
      <div class="imp-bar-bg"><div class="imp-bar-fill" style="width:${(v/maxI*100).toFixed(1)}%"></div></div>
    </div>`).join("");
}

// ── 02 MLP ────────────────────────────────────────────────────────────────────
function renderMLP(d) {
  text("mlp-pred",    "$"+d.predicted_close);
  text("mlp-actual",  d.actual_close?"$"+d.actual_close:"—");
  text("mlp-err-val", d.error_usd?"$"+d.error_usd:"—");
  text("mlp-rmse-val","$"+d.model_meta.mlp_rmse);
  text("mlp-mae-val", "$"+d.model_meta.mlp_mae);

  const labels=[...d.hist_dates,"Predicted"];
  const histDs=d.hist_closes.slice();
  const predDs=new Array(d.hist_closes.length).fill(null);
  predDs.push(d.predicted_close);

  mkChart("spark-chart",{
    type:"line",
    data:{ labels, datasets:[
      {label:"Actual",data:histDs,borderColor:C_LO,borderWidth:1.5,pointRadius:0,tension:0.3},
      {label:"Predicted",data:predDs,borderColor:C_BLUE,borderWidth:2,borderDash:[4,4],
       pointRadius:predDs.map((_,i)=>i===predDs.length-1?5:0),pointBackgroundColor:C_BLUE,tension:0},
    ]},
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:true,labels:{color:"#7b8eab",boxWidth:12,font:{size:11}}},tooltip:tt()},
      scales:{
        x:{grid:{color:GRID},border:{color:"var(--border)"},ticks:{maxTicksLimit:6,font:{size:10},color:"#3e4f6a"}},
        y:{grid:{color:GRID},border:{color:"var(--border)"},ticks:{font:{size:10,family:"IBM Plex Mono"},color:"#3e4f6a",callback:v=>"$"+v.toFixed(0)}},
      },
      animation:{duration:600},
    },
  });
}

// ── 03 LIME ───────────────────────────────────────────────────────────────────
function renderLIME(explanations) {
  const sorted=[...explanations].sort((a,b)=>Math.abs(b.weight)-Math.abs(a.weight));
  mkChart("lime-chart",{
    type:"bar",
    data:{ labels:sorted.map(e=>e.feature), datasets:[{
      data:sorted.map(e=>e.weight),
      backgroundColor:sorted.map(e=>e.weight>0?C_HI+"aa":C_LO+"aa"),
      borderColor:sorted.map(e=>e.weight>0?C_HI:C_LO),
      borderWidth:1,borderRadius:3,
    }]},
    options:{
      indexAxis:"y",responsive:true,maintainAspectRatio:false,
      plugins:{tooltip:tt({callbacks:{label:c=>{const v=c.parsed.x;return ` ${v.toFixed(5)}  → ${v>0?"High Vol":"Low Vol"}`;}}})},
      scales:{
        x:{grid:{color:GRID},border:{color:"var(--border)"},ticks:{font:{size:10,family:"IBM Plex Mono"},color:"#3e4f6a"},
           title:{display:true,text:"← Low Volatility   |   High Volatility →",color:"#3e4f6a",font:{size:10}}},
        y:{grid:{display:false},border:{display:false},ticks:{font:{size:11},color:"#7b8eab"}},
      },
      animation:{duration:600},
    },
  });
}

// ── 04 DiCE ───────────────────────────────────────────────────────────────────
function renderDiCE(data) {
  const container=$id("dice-content");
  if(!data.counterfactuals||!data.counterfactuals.length){
    container.innerHTML=`<p style="color:var(--txt-2);font-size:12px;padding:8px 0;">No counterfactuals generated. Try another date.</p>`;return;
  }
  container.innerHTML=`<div class="scenario-wrap">${
    data.counterfactuals.map((cf,i)=>{
      const entries=Object.entries(cf.changes||{});
      const rows=entries.length===0
        ?`<tr><td colspan="5" style="color:var(--txt-3);text-align:center;padding:14px;">No significant changes needed.</td></tr>`
        :entries.map(([feat,ch])=>{
          const up=ch.change>0;
          return`<tr>
            <td>${feat}</td>
            <td class="td-mono td-orig">${ch.original.toFixed(4)}</td>
            <td class="td-mono" style="color:var(--txt);">${ch.counterfactual.toFixed(4)}</td>
            <td class="${up?"td-up":"td-dn"}">${up?"↑":"↓"} ${Math.abs(ch.change).toFixed(4)}</td>
            <td><span class="pct-tag ${up?"pct-up":"pct-dn"}">${(ch.pct_change>0?"+":"")+ch.pct_change.toFixed(1)}%</span></td>
          </tr>`;
        }).join("");
      return`<div class="scenario">
        <div class="scen-head">
          <div><div class="scen-title">Scenario ${i+1}</div><div class="scen-sub">${entries.length} feature${entries.length!==1?"s":""} to change</div></div>
          <span class="scen-badge">${entries.length} change${entries.length!==1?"s":""}</span>
        </div>
        <table class="scen-table">
          <thead><tr><th>Feature</th><th>Current</th><th>Required</th><th>Δ</th><th>%</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
    }).join("")
  }</div>`;
}

// ── 05 SHAP ───────────────────────────────────────────────────────────────────
function renderSHAP(d) {
  const feats=Object.keys(d.feature_importance), vals=Object.values(d.feature_importance);
  mkChart("shap-feat-chart", barCfg(feats, vals, FEAT_COLORS, "Mean |SHAP|"));

  const steps=d.temporal_importance.map(t=>t.step), tv=d.temporal_importance.map(t=>t.importance);
  mkChart("shap-time-chart",{
    type:"line",
    data:{labels:steps,datasets:[{data:tv,borderColor:C_BLUE,borderWidth:1.5,backgroundColor:C_BLUE+"18",fill:true,pointRadius:0,tension:0.4}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{tooltip:tt({callbacks:{label:c=>` ${c.label}: ${c.parsed.y.toFixed(6)}`}})},
      scales:{
        x:{grid:{display:false},border:{color:"var(--border)"},ticks:{font:{size:9,family:"IBM Plex Mono"},color:"#3e4f6a",maxTicksLimit:10}},
        y:{grid:{color:GRID},border:{color:"var(--border)"},ticks:{font:{size:10,family:"IBM Plex Mono"},color:"#3e4f6a"},
           title:{display:true,text:"Mean |SHAP|",color:"#3e4f6a",font:{size:10}}},
      },animation:{duration:600}},
  });
}

// ── 06 BETA Surrogate Tree ────────────────────────────────────────────────────
function renderBETA(d) {
  // Stats chips
  $id("beta-stats").innerHTML=`
    <div class="beta-stat-chip"><span class="beta-stat-val">${d.beta_r2}</span><span class="beta-stat-lbl">Surrogate R²</span></div>
    <div class="beta-stat-chip"><span class="beta-stat-val">${d.tree_depth}</span><span class="beta-stat-lbl">Tree Depth</span></div>
    <div class="beta-stat-chip"><span class="beta-stat-val">${d.tree_leaves}</span><span class="beta-stat-lbl">Leaf Nodes</span></div>
    <div class="beta-stat-chip"><span class="beta-stat-val">${d.nodes_visited}</span><span class="beta-stat-lbl">Nodes Visited</span></div>
    <div class="beta-stat-chip"><span class="beta-stat-val">${d.agreement_pct.toFixed(1)}%</span><span class="beta-stat-lbl">MLP Agreement</span></div>
  `;
  const feats=Object.keys(d.feature_importance), gVals=Object.values(d.feature_importance);
  const lVals=Object.values(d.local_importance);
  mkChart("beta-global-chart", barCfg(feats, gVals, FEAT_COLORS, "Importance"));
  mkChart("beta-local-chart",  barCfg(feats, lVals, FEAT_COLORS, "Importance"));
}

// ── 07 Sensitivity + LRP ─────────────────────────────────────────────────────
function renderSensitivity(d) {
  const feats=Object.keys(d.sensitivity);
  const sensVals=Object.values(d.sensitivity);
  const lrpVals =Object.values(d.lrp_scores);

  mkChart("sens-chart", barCfg(feats, sensVals, FEAT_COLORS, "|∂output/∂input|"));
  mkChart("lrp-chart",  barCfg(feats, lrpVals,  FEAT_COLORS, "Relevance Score"));
}

// ── 08 Ablation + Permutation ─────────────────────────────────────────────────
function renderAblation(d) {
  const feats=d.features;
  const ablVals=feats.map(f=>d.ablation_importance[f]||0);
  mkChart("ablation-chart", barCfg(feats, ablVals, FEAT_COLORS, "Prediction Change"));
}

// ── 09 Saliency Map (canvas heatmap) ─────────────────────────────────────────
function renderSaliency(d) {
  // Feature labels
  const labelWrap=$id("saliency-feat-labels");
  labelWrap.innerHTML=d.features.map(f=>`<span>${f}</span>`).join("");

  const canvas=$id("saliency-canvas");
  const nTime=d.time_steps, nFeat=d.features.length;
  const CELL_W=80, CELL_H=8;
  canvas.width  = nFeat * CELL_W;
  canvas.height = nTime * CELL_H;
  canvas.style.height = Math.min(480, nTime*CELL_H)+"px";

  const ctx=canvas.getContext("2d");

  // Colour scale: dark blue → cyan → white
  function heatColor(v) {
    // v in [0,1]
    if(v<0.5){
      const t=v*2;
      return [Math.round(13+t*(0)),Math.round(17+t*(157)),Math.round(38+t*(224))];
    } else {
      const t=(v-0.5)*2;
      return [Math.round(0+t*255),Math.round(157+t*(255-157)),Math.round(224+t*(255-224))];
    }
  }

  for(let row=0;row<nTime;row++){
    for(let col=0;col<nFeat;col++){
      const v=d.saliency[row][col];
      const [r,g,b]=heatColor(v);
      ctx.fillStyle=`rgb(${r},${g},${b})`;
      ctx.fillRect(col*CELL_W, row*CELL_H, CELL_W-1, CELL_H-1);
    }
  }

  // Legend bar
  const leg=$id("saliency-legend-bar");
  if(leg){
    const lctx=leg.getContext("2d");
    const grad=lctx.createLinearGradient(0,0,200,0);
    grad.addColorStop(0,"rgb(13,17,38)");
    grad.addColorStop(0.5,"rgb(0,157,224)");
    grad.addColorStop(1,"rgb(255,255,255)");
    lctx.fillStyle=grad;
    lctx.fillRect(0,0,200,14);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════════════════════════════════════
function setDate(d) { $id("date-input").value=d; }

function setBtnState(loading) {
  const btn=$id("run-btn"); if(!btn)return;
  btn.disabled=loading;
  btn.innerHTML=loading?`Running… <span class="btn-arrow">…</span>`:`Run Analysis <span class="btn-arrow">→</span>`;
}

document.addEventListener("DOMContentLoaded",()=>{
  init();
  const inp=$id("date-input");
  if(inp) inp.addEventListener("keydown",e=>e.key==="Enter"&&analyzeDate());
});