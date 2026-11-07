"""
Build a self-contained HTML journal for the final EMA 7/18 cross-back config.
Calendar views (week / month / year / YTD / 4-year), green/red daily P&L cells,
a stats header that updates per view, and click-a-day -> candlestick chart of
the trade(s). All data + per-trade OHLC windows are embedded in the HTML.
"""
import os
import json
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

import main as bt
import news_filter as nf
import ema_dol_strategy as ed
import realistic_ema_dol_rsi as edr
import frequency_tune as fq

FAST, SLOW = 7, 18
RISK, START = 1000.0, 100000.0
OUT = os.path.join("strategies", "ema718_final", "journal.html")


def load(s):
    p = os.path.join("data_cache", f"{s}_M15.pkl")
    try:
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")
    except Exception:
        if os.path.exists(p):
            os.remove(p)
        return bt.fetch_data(s, mt5.TIMEFRAME_M15, bt.START_DATE, bt.END_DATE, tf_name="M15")


def collect(df, symbol):
    spread, slippage, min_w = fq.SPEC[symbol]
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    o = df["open"].values; idx = df.index
    ema_f = pd.Series(c, index=idx).ewm(span=FAST, adjust=False).mean().values
    ema_s = pd.Series(c, index=idx).ewm(span=SLOW, adjust=False).mean().values
    above = ema_f > ema_s
    N = len(df)
    cu = np.zeros(N, bool); cd = np.zeros(N, bool)
    cu[1:] = above[1:] & ~above[:-1]; cd[1:] = ~above[1:] & above[:-1]
    nb = np.full(N + 1, N, int); nbr = np.full(N + 1, N, int)
    for k in range(N - 1, -1, -1):
        nb[k] = k if cu[k] else nb[k + 1]; nbr[k] = k if cd[k] else nbr[k + 1]
    sh, sl_arr = ed.find_swings_arr(h, l, n=ed.SWING_LOOKBACK)
    pdh_map, pdl_map = ed.build_pdh_pdl(df)
    rsi_arr = edr.rsi(c)
    news_set = nf.build_news_block_set(idx[0].date(), idx[-1].date())
    dig = 3 if "JPY" in symbol else (2 if "." in symbol else 5)

    trades = []; open_until = []
    for i in range(N):
        if not (cu[i] or cd[i]):
            continue
        bias = "long" if cu[i] else "short"
        tdir = "bullish" if bias == "long" else "bearish"
        opp = nbr[i + 1] if bias == "long" else nb[i + 1]
        end_idx = min(i + ed.FVG_LOOKUP_WINDOW, opp - 1, N - 1)
        if end_idx < i + 2:
            continue
        fvg = ed.find_aligned_fvg(h, l, i, end_idx, tdir, min_w)
        if fvg is None:
            continue
        rt_end = min(fvg["form_idx"] + ed.RETEST_WINDOW, opp - 1, N - 1)
        ei = None
        for k in range(fvg["form_idx"] + 1, rt_end + 1):
            if bias == "long" and l[k] <= fvg["top"]:    ei = k; break
            if bias == "short" and h[k] >= fvg["bottom"]: ei = k; break
        if ei is None:
            continue
        open_until = [t for t in open_until if t > idx[ei]]
        if len(open_until) >= 1:
            continue
        if idx[ei].date() in news_set:
            continue
        rv = rsi_arr[ei]
        if np.isnan(rv):
            continue
        if bias == "long" and not (45 <= rv <= 100):
            continue
        if bias == "short" and not (0 <= rv <= 55):
            continue
        d_today = pd.Timestamp(idx[ei]).normalize()
        pdh_t = pdh_map.get(d_today); pdl_t = pdl_map.get(d_today)
        buf = fvg["width"] * ed.SL_BUFFER_FRAC
        pad = edr.sl_pad(symbol, spread)                  # spread+1pip stop buffer
        if bias == "long":
            entry = fvg["top"] + spread + slippage
            swlow = ed.find_recent_swing_low(sl_arr, ei, lookback=80)
            anchor = swlow if swlow is not None and swlow < fvg["bottom"] else fvg["bottom"]
            sl = anchor - buf - pad; risk = entry - sl; direction = "long"
            if risk <= 0: continue
            dol = ed.next_liquidity_above(entry, sh, ei, pdh_t)
            if dol is None or not (2.0 <= (dol - entry) / risk <= 10.0): continue
        else:
            entry = fvg["bottom"] - slippage
            swhigh = ed.find_recent_swing_high(sh, ei, lookback=80)
            anchor = swhigh if swhigh is not None and swhigh > fvg["top"] else fvg["top"]
            sl = anchor + buf + pad; risk = sl - entry; direction = "short"
            if risk <= 0: continue
            dol = ed.next_liquidity_below(entry, sl_arr, ei, pdl_t)
            if dol is None or not (2.0 <= (entry - dol) / risk <= 10.0): continue

        exit_i = None; exitpx = None; pr = None
        for j in range(ei + 1, min(opp, ei + 2000, N - 1) + 1):
            if direction == "long":
                if l[j] <= sl: exit_i, exitpx, pr = j, sl, (sl - entry) / risk; break
            else:
                if h[j] + spread >= sl: exit_i, exitpx, pr = j, sl, (entry - sl) / risk; break
            # GRADIENT EXIT: close when the fast EMA slope turns against us (once in profit)
            grad = ema_f[j] - ema_f[j - 1]
            inprofit = (c[j] > entry) if direction == "long" else (c[j] + spread < entry)
            if inprofit and ((direction == "long" and grad < 0) or (direction == "short" and grad > 0)):
                px = c[j] if direction == "long" else c[j] + spread
                exit_i, exitpx = j, px
                pr = ((px - entry) if direction == "long" else (entry - px)) / risk; break
            if j == opp:                                   # fallback: the actual cross
                px = c[j] if direction == "long" else c[j] + spread
                exit_i, exitpx = j, px
                pr = ((px - entry) if direction == "long" else (entry - px)) / risk; break
        if pr is None:
            exit_i = min(opp, N - 1)
            px = c[exit_i] if direction == "long" else c[exit_i] + spread
            exitpx = px; pr = ((px - entry) if direction == "long" else (entry - px)) / risk
        outcome = "win" if pr > 0.01 else ("loss" if pr < -0.01 else "be")

        w0 = max(0, i - 12); w1 = min(N - 1, exit_i + 6, i + 100)
        bars = [[round(float(o[k]), dig), round(float(h[k]), dig), round(float(l[k]), dig),
                 round(float(c[k]), dig), round(float(ema_f[k]), dig), round(float(ema_s[k]), dig)]
                for k in range(w0, w1 + 1)]
        step = max(1, (w1 - w0) // 6)
        ticks = [[k - w0, idx[k].strftime("%m-%d %H:%M")] for k in range(w0, w1 + 1, step)]
        trades.append(dict(
            date=idx[ei].strftime("%Y-%m-%d"), symbol=symbol, dir=direction,
            rr=round(float(pr), 2), pnl=int(round(float(pr) * RISK)), outcome=outcome,
            rsi=int(round(float(rv))), entry=round(float(entry), dig), sl=round(float(sl), dig),
            exit=round(float(exitpx), dig), et=idx[ei].strftime("%Y-%m-%d %H:%M"),
            xt=idx[exit_i].strftime("%Y-%m-%d %H:%M"),
            hold=round((idx[exit_i] - idx[ei]).total_seconds() / 3600, 1),
            ft=round(float(fvg["top"]), dig), fb=round(float(fvg["bottom"]), dig),
            ci=i - w0, fi=fvg["form_idx"] - w0, xi=exit_i - w0, ei2=ei - w0,
            dig=dig, bars=bars, ticks=ticks,
        ))
        open_until.append(idx[exit_i])
    return trades


def main():
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    allt = []
    for s in fq.UNIVERSE:
        allt += collect(load(s), s)
    allt.sort(key=lambda t: t["et"])
    for i, t in enumerate(allt):
        t["id"] = i
    data = {"config": "EMA 7/18 · gradient exit · DOL>=2R filter · RSI 45/55 · news ON · "
                      "spread+1pip stop · 17 instruments · $1,000 risk / $100k (1%)",
            "risk": RISK, "start": START, "trades": allt}
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}  ({len(allt)} trades, {kb:.0f} KB)")
    mt5.shutdown()


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backtest Journal · EMA 7/18</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#7d8590;
--grn:#3fb950;--grnbg:#132e1c;--red:#f85149;--redbg:#3a1518;--blue:#58a6ff;--amber:#d29922;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;font-size:14px}
.wrap{max-width:1180px;margin:0 auto;padding:18px}
h1{font-size:18px;font-weight:600;margin:0 0 2px}.sub{color:var(--mut);font-size:12px;margin-bottom:14px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.tab{padding:7px 14px;background:var(--panel);border:1px solid var(--line);border-radius:8px;
cursor:pointer;color:var(--mut);font-weight:500}
.tab.on{color:var(--fg);border-color:var(--blue);background:#122033}
.nav{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.nav button{background:var(--panel);border:1px solid var(--line);color:var(--fg);border-radius:8px;
width:34px;height:34px;cursor:pointer;font-size:16px}
.nav .lbl{font-weight:600;font-size:15px;min-width:180px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.03em}
.card .v{font-size:19px;font-weight:600;margin-top:3px}
.pos{color:var(--grn)}.neg{color:var(--red)}
.cal{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
.dow{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;margin-bottom:6px}
.dow div{color:var(--mut);font-size:11px;text-align:center;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(7,1fr);gap:6px}
.cell{aspect-ratio:1.15;border:1px solid var(--line);border-radius:8px;padding:6px;
display:flex;flex-direction:column;justify-content:space-between;background:#0e141b;min-height:56px}
.cell.out{opacity:.25}.cell .dn{font-size:12px;color:var(--mut)}
.cell.win{background:var(--grnbg);border-color:#1f6b38}.cell.loss{background:var(--redbg);border-color:#7a2b2b}
.cell.win .dn,.cell.loss .dn{color:var(--fg)}
.cell .amt{font-size:13px;font-weight:600}.cell.has{cursor:pointer}
.cell.has:hover{border-color:var(--blue)}
.weekrow{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}
.weekrow .cell{aspect-ratio:auto;min-height:120px}
.months{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.mini h3{margin:0 0 6px;font-size:13px;font-weight:600}
.mini .grid{gap:3px}.mini .cell{aspect-ratio:1;min-height:0;border-radius:4px;padding:0;
align-items:center;justify-content:center}
.mini .cell .dn{font-size:9px}.mini .cell .amt{display:none}
.yearcards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.heat{overflow-x:auto;padding-bottom:6px}
.heatgrid{display:grid;grid-auto-flow:column;grid-template-rows:repeat(7,11px);gap:3px}
.hc{width:11px;height:11px;border-radius:2px;background:#161b22;border:1px solid #21262d}
.hc.win{background:var(--grn)}.hc.loss{background:var(--red)}.hc.has{cursor:pointer}
.leg{color:var(--mut);font-size:11px;margin-top:8px}
.modal{position:fixed;inset:0;background:rgba(0,0,0,.7);display:none;align-items:center;
justify-content:center;padding:20px;z-index:10}
.modal.on{display:flex}
.sheet{background:var(--panel);border:1px solid var(--line);border-radius:14px;max-width:920px;
width:100%;max-height:92vh;overflow:auto;padding:18px}
.sheet h2{margin:0 0 2px;font-size:16px}.close{float:right;cursor:pointer;color:var(--mut);font-size:20px}
.tr{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}
.trh{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;margin-bottom:8px}
.trh b{font-weight:600}.chip{padding:2px 8px;border-radius:6px;font-weight:600}
canvas{width:100%;height:300px;background:#0d1117;border:1px solid var(--line);border-radius:8px}
</style></head><body><div class="wrap">
<h1>Backtest Journal</h1><div class="sub" id="cfg"></div>
<div class="tabs" id="tabs"></div>
<div class="nav"><button id="prev">&#8249;</button><div class="lbl" id="plabel"></div>
<button id="next">&#8250;</button></div>
<div class="stats" id="stats"></div>
<div id="body"></div>
</div>
<div class="modal" id="modal"><div class="sheet"><span class="close" id="close">&times;</span>
<h2 id="mtitle"></h2><div id="mbody"></div></div></div>
<script>
const DATA=/*__DATA__*/;
const TR=DATA.trades, byDate={};
TR.forEach(t=>{(byDate[t.date]=byDate[t.date]||[]).push(t)});
const alld=TR.map(t=>t.date).sort();
const MIN=alld[0], MAX=alld[alld.length-1];
document.getElementById('cfg').textContent=DATA.config;
const money=n=>(n<0?'-$':'$')+Math.abs(Math.round(n)).toLocaleString();
const pad=n=>String(n).padStart(2,'0');
const iso=d=>d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate());
const parse=s=>{const[a,b,c]=s.split('-').map(Number);return new Date(a,b-1,c)};
const MON=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
let view='month', anchor=parse(MAX);

const VIEWS=[['week','Week'],['month','Month'],['year','Year'],['ytd','Year-to-date'],['all','4-Year']];
const tabsEl=document.getElementById('tabs');
VIEWS.forEach(([k,l])=>{const b=document.createElement('div');b.className='tab'+(k==view?' on':'');
b.textContent=l;b.onclick=()=>{view=k;[...tabsEl.children].forEach(c=>c.classList.remove('on'));
b.classList.add('on');render()};tabsEl.appendChild(b)});

function range(){
  const y=anchor.getFullYear(),m=anchor.getMonth();
  if(view==='week'){const d=new Date(anchor);const wd=(d.getDay()+6)%7;d.setDate(d.getDate()-wd);
    const e=new Date(d);e.setDate(d.getDate()+6);return[d,e];}
  if(view==='month')return[new Date(y,m,1),new Date(y,m+1,0)];
  if(view==='year')return[new Date(y,0,1),new Date(y,11,31)];
  if(view==='ytd'){const ly=parse(MAX).getFullYear();return[new Date(ly,0,1),parse(MAX)];}
  return[parse(MIN),parse(MAX)];
}
function inRange(ts){const[a,b]=range();const s=iso(a),e=iso(b);return TR.filter(t=>t.date>=s&&t.date<=e);}

function stats(ts){
  ts=ts.slice().sort((a,b)=>a.et<b.et?-1:1);
  let n=ts.length,w=0,l=0,be=0,pnl=0,gw=0,gl=0,rs=0,cw=0,cl=0,bw=0,bl=0,cum=0,pk=0,dd=0,byd={};
  ts.forEach(t=>{pnl+=t.pnl;rs+=t.rr;
    if(t.pnl>0){w++;gw+=t.pnl;cw++;cl=0;bw=Math.max(bw,cw);}
    else if(t.pnl<0){l++;gl-=t.pnl;cl++;cw=0;bl=Math.max(bl,cl);}else be++;
    cum+=t.pnl;pk=Math.max(pk,cum);dd=Math.min(dd,cum-pk);
    byd[t.date]=(byd[t.date]||0)+t.pnl;});
  const dv=Object.values(byd);
  return{n,w,l,be,wr:n?w/n*100:0,pnl,pf:gl?gw/gl:(gw?99:0),aw:w?gw/w:0,al:l?gl/l:0,
    ar:n?rs/n:0,best:dv.length?Math.max(...dv):0,worst:dv.length?Math.min(...dv):0,
    bw,bl,dd,exp:n?pnl/n:0,gdays:dv.filter(x=>x>0).length,bdays:dv.filter(x=>x<0).length};
}
function renderStats(){
  const s=stats(inRange());const cls=v=>v>0?'pos':(v<0?'neg':'');
  const cards=[['Net P&L',money(s.pnl),cls(s.pnl)],['Win rate',s.wr.toFixed(1)+'%',''],
    ['Trades',s.n+' ('+s.w+'W / '+s.l+'L)',''],['Profit factor',s.pf.toFixed(2),cls(s.pf-1)],
    ['Avg win',money(s.aw),'pos'],['Avg loss',money(-s.al),'neg'],['Avg R',(s.ar>=0?'+':'')+s.ar.toFixed(2)+'R',cls(s.ar)],
    ['Best day',money(s.best),'pos'],['Worst day',money(s.worst),'neg'],
    ['Win streak',s.bw,''],['Loss streak',s.bl,''],['Max DD (period)',money(s.dd),'neg'],
    ['Green/Red days',s.gdays+' / '+s.bdays,'']];
  document.getElementById('stats').innerHTML=cards.map(([k,v,c])=>
    '<div class="card"><div class="k">'+k+'</div><div class="v '+c+'">'+v+'</div></div>').join('');
}
function dayPnl(ds){return(byDate[ds]||[]).reduce((a,t)=>a+t.pnl,0);}
function cell(ds,dnum,muted,mini){
  const has=byDate[ds]&&byDate[ds].length;const p=has?dayPnl(ds):0;
  const cl=has?(p>0?'win':(p<0?'loss':'')):'';
  let h='<div class="cell '+cl+(muted?' out':'')+(has?' has':'')+'" '+(has?'data-d="'+ds+'"':'')+
    ' title="'+ds+(has?' · '+money(p)+' · '+byDate[ds].length+' trade(s)':'')+'">';
  h+='<div class="dn">'+dnum+'</div>';
  if(!mini)h+='<div class="amt '+(p>0?'pos':(p<0?'neg':''))+'">'+(has?money(p):'')+'</div>';
  return h+'</div>';
}
function monthGrid(y,m,mini){
  const first=new Date(y,m,1),start=(first.getDay()+6)%7,dim=new Date(y,m+1,0).getDate();
  let h='<div class="grid">';
  for(let i=0;i<start;i++)h+='<div class="cell out"></div>';
  for(let d=1;d<=dim;d++){h+=cell(iso(new Date(y,m,d)),d,false,mini);}
  return h+'</div>';
}
function dowHead(){return '<div class="dow">'+['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
  .map(x=>'<div>'+x+'</div>').join('')+'</div>';}

function render(){
  renderStats();
  const[a,b]=range();const y=anchor.getFullYear();const bd=document.getElementById('body');
  let lbl='';
  if(view==='week')lbl=a.getDate()+' '+MON[a.getMonth()]+' – '+b.getDate()+' '+MON[b.getMonth()]+' '+b.getFullYear();
  else if(view==='month')lbl=MON[anchor.getMonth()]+' '+y;
  else if(view==='year')lbl=y;
  else if(view==='ytd')lbl=parse(MAX).getFullYear()+' YTD';
  else lbl=parse(MIN).getFullYear()+' – '+parse(MAX).getFullYear();
  document.getElementById('plabel').textContent=lbl;
  document.getElementById('prev').style.visibility=(view==='ytd'||view==='all')?'hidden':'visible';
  document.getElementById('next').style.visibility=(view==='ytd'||view==='all')?'hidden':'visible';

  if(view==='week'){
    const d=new Date(a);let h='<div class="cal">'+dowHead()+'<div class="weekrow">';
    for(let i=0;i<7;i++){const ds=iso(d);const has=byDate[ds]&&byDate[ds].length;const p=dayPnl(ds);
      const cl=has?(p>0?'win':(p<0?'loss':'')):'';
      h+='<div class="cell '+cl+(has?' has':'')+'" '+(has?'data-d="'+ds+'"':'')+'>'+
        '<div class="dn">'+MON[d.getMonth()]+' '+d.getDate()+'</div>'+
        '<div><div class="amt '+(p>0?'pos':(p<0?'neg':''))+'">'+(has?money(p):'')+'</div>'+
        (has?'<div style="color:var(--mut);font-size:11px">'+byDate[ds].length+' trade(s)</div>':'')+'</div></div>';
      d.setDate(d.getDate()+1);}
    bd.innerHTML=h+'</div></div>';
  } else if(view==='month'){
    bd.innerHTML='<div class="cal">'+dowHead()+monthGrid(y,anchor.getMonth(),false)+'</div>';
  } else if(view==='year'||view==='ytd'){
    const yr=view==='ytd'?parse(MAX).getFullYear():y;
    const last=view==='ytd'?parse(MAX).getMonth():11;
    let h='<div class="months">';
    for(let m=0;m<=last;m++)h+='<div class="cal mini"><h3>'+MON[m]+' '+yr+'</h3>'+monthGrid(yr,m,true)+'</div>';
    bd.innerHTML=h+'</div>';
  } else {
    // 4-year: year summary cards + heatmap
    const yrs={};TR.forEach(t=>{const yy=t.date.slice(0,4);(yrs[yy]=yrs[yy]||[]).push(t)});
    let h='<div class="yearcards">';
    Object.keys(yrs).sort().forEach(yy=>{const s=stats(yrs[yy]);
      h+='<div class="card"><div class="k">'+yy+'</div>'+
        '<div class="v '+(s.pnl>0?'pos':'neg')+'">'+money(s.pnl)+'</div>'+
        '<div style="color:var(--mut);font-size:11px;margin-top:3px">'+s.n+' trades · '+
        s.wr.toFixed(0)+'% WR · PF '+s.pf.toFixed(2)+'</div></div>';});
    h+='</div><div class="cal"><div class="heat"><div class="heatgrid" id="hg"></div></div>'+
      '<div class="leg">Each square = one day. Green = profit, red = loss. Hover for detail, click for the trade.</div></div>';
    bd.innerHTML=h;
    const hg=document.getElementById('hg');let d=parse(MIN);const wd=(d.getDay()+6)%7;d.setDate(d.getDate()-wd);
    const end=parse(MAX);let html='';
    while(d<=end){const ds=iso(d);const has=byDate[ds]&&byDate[ds].length;const p=has?dayPnl(ds):0;
      const cl=has?(p>0?'win':(p<0?'loss':'')):'';
      html+='<div class="hc '+cl+(has?' has':'')+'" '+(has?'data-d="'+ds+'"':'')+
        ' title="'+ds+(has?' · '+money(p):'')+'"></div>';d.setDate(d.getDate()+1);}
    hg.innerHTML=html;
  }
  bd.querySelectorAll('[data-d]').forEach(el=>el.onclick=()=>openDay(el.getAttribute('data-d')));
}

function openDay(ds){
  const ts=byDate[ds]||[];if(!ts.length)return;
  document.getElementById('mtitle').innerHTML=ds+' &middot; '+ts.length+' trade(s) &middot; '+
    '<span class="'+(dayPnl(ds)>0?'pos':'neg')+'">'+money(dayPnl(ds))+'</span>';
  const mb=document.getElementById('mbody');mb.innerHTML='';
  ts.forEach((t,ix)=>{
    const div=document.createElement('div');div.className='tr';
    const chip=t.pnl>0?'style="background:var(--grnbg);color:var(--grn)"':'style="background:var(--redbg);color:var(--red)"';
    div.innerHTML='<div class="trh"><b>'+t.symbol+'</b><span>'+t.dir.toUpperCase()+'</span>'+
      '<span class="chip" '+chip+'>'+(t.pnl>0?'WIN ':'LOSS ')+(t.rr>=0?'+':'')+t.rr+'R · '+money(t.pnl)+'</span>'+
      '<span>entry '+t.entry+'</span><span>SL '+t.sl+'</span><span>exit '+t.exit+'</span>'+
      '<span>RSI '+t.rsi+'</span><span>held '+t.hold+'h</span></div>'+
      '<canvas id="cv'+ix+'"></canvas>';
    mb.appendChild(div);
  });
  // show the modal FIRST, force a reflow so canvases get a real width, THEN draw
  const modalEl=document.getElementById('modal');
  modalEl.classList.add('on');
  void modalEl.offsetWidth;            // force synchronous layout
  requestAnimationFrame(()=>ts.forEach((t,ix)=>drawChart(document.getElementById('cv'+ix),t)));
  ts.forEach((t,ix)=>drawChart(document.getElementById('cv'+ix),t));
}
document.getElementById('close').onclick=()=>document.getElementById('modal').classList.remove('on');
document.getElementById('modal').onclick=e=>{if(e.target.id==='modal')document.getElementById('modal').classList.remove('on');};

// prev / next navigation — shift the anchor by week / month / year for the current view
function shift(dir){
  const d=new Date(anchor);
  if(view==='week') d.setDate(d.getDate()+dir*7);
  else if(view==='month') d.setMonth(d.getMonth()+dir);
  else if(view==='year') d.setFullYear(d.getFullYear()+dir);
  else return;                 // ytd / all have no navigation
  anchor=d; render();
}
document.getElementById('prev').onclick=()=>shift(-1);
document.getElementById('next').onclick=()=>shift(1);
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')shift(-1); else if(e.key==='ArrowRight')shift(1);
});

function drawChart(cv,t){
  const dpr=window.devicePixelRatio||1;const W=cv.clientWidth||cv.parentElement.clientWidth||860,H=300;
  cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext('2d');x.scale(dpr,dpr);
  const b=t.bars,n=b.length,pl=8,pr=64,pt=12,pb=26,pw=W-pl-pr,ph=H-pt-pb;
  let lo=1e9,hi=-1e9;b.forEach(r=>{lo=Math.min(lo,r[2]);hi=Math.max(hi,r[1])});
  [t.entry,t.sl,t.exit].forEach(v=>{lo=Math.min(lo,v);hi=Math.max(hi,v)});
  const padv=(hi-lo)*0.06||1;lo-=padv;hi+=padv;
  const X=i=>pl+(n<=1?pw/2:i/(n-1)*pw);const Y=p=>pt+(hi-p)/(hi-lo)*ph;
  const bw=Math.max(2,pw/n*0.62);
  x.strokeStyle='#21262d';x.lineWidth=1;
  for(let g=0;g<=4;g++){const yy=pt+g/4*ph;x.beginPath();x.moveTo(pl,yy);x.lineTo(pl+pw,yy);x.stroke();
    x.fillStyle='#7d8590';x.font='10px sans-serif';x.textAlign='left';
    x.fillText((hi-(g/4)*(hi-lo)).toFixed(t.dig),pl+pw+4,yy+3);}
  // FVG zone
  const fx0=X(Math.max(0,t.fi-2)),fx1=X(t.ei2);
  x.fillStyle='rgba(188,140,255,.15)';x.fillRect(fx0,Y(t.ft),fx1-fx0,Y(t.fb)-Y(t.ft));
  // candles
  b.forEach((r,i)=>{const up=r[3]>=r[0];x.strokeStyle=up?'#26a69a':'#ef5350';x.fillStyle=up?'#26a69a':'#ef5350';
    const cx=X(i);x.beginPath();x.moveTo(cx,Y(r[1]));x.lineTo(cx,Y(r[2]));x.stroke();
    const y0=Y(Math.max(r[0],r[3])),hh=Math.max(1,Math.abs(Y(r[0])-Y(r[3])));x.fillRect(cx-bw/2,y0,bw,hh);});
  // EMAs
  const line=(idx,col)=>{x.strokeStyle=col;x.lineWidth=1.3;x.beginPath();
    b.forEach((r,i)=>{const xx=X(i),yy=Y(r[idx]);i?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke();};
  line(4,'#d29922');line(5,'#58a6ff');
  // cross line
  x.strokeStyle='#7d8590';x.setLineDash([3,3]);x.lineWidth=1;x.beginPath();
  x.moveTo(X(t.ci),pt);x.lineTo(X(t.ci),pt+ph);x.stroke();x.setLineDash([]);
  // entry/sl/exit levels (from entry bar to exit bar)
  const seg=(p,col)=>{x.strokeStyle=col;x.lineWidth=1.4;x.beginPath();
    x.moveTo(X(t.ei2),Y(p));x.lineTo(X(t.xi),Y(p));x.stroke();};
  seg(t.sl,'#f85149');seg(t.entry,'#e6edf3');seg(t.exit,t.pnl>0?'#3fb950':'#f85149');
  // markers
  x.fillStyle='#f2cc60';const my=Y(t.entry);x.beginPath();
  if(t.dir==='long'){x.moveTo(X(t.ei2),my-7);x.lineTo(X(t.ei2)-6,my+5);x.lineTo(X(t.ei2)+6,my+5);}
  else{x.moveTo(X(t.ei2),my+7);x.lineTo(X(t.ei2)-6,my-5);x.lineTo(X(t.ei2)+6,my-5);}x.closePath();x.fill();
  x.fillStyle=t.pnl>0?'#3fb950':'#f85149';x.beginPath();x.arc(X(t.xi),Y(t.exit),5,0,7);x.fill();
  // labels
  x.fillStyle='#e6edf3';x.font='10px sans-serif';x.textAlign='left';
  x.fillText('entry '+t.entry,X(t.ei2)+6,my-6);
  x.fillStyle='#7d8590';x.textAlign='center';
  t.ticks.forEach(([i,s])=>{if(i<n)x.fillText(s,X(i),H-8);});
}
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
