"""
첫 화면(시연 무대): 왼쪽 오늘의 상태 · 가운데 스마트폰 · 오른쪽 도착한 메시지

- 프리셋 3개 x 공격 유형 4개의 결과를 Python에서 미리 계산해 넘기므로, 버튼을 누르면 즉시 바뀐다.
- 폰을 누르면 반쯤 뒤집힌 뒤 Python으로 'flip' 신호를 보내고, Python이 넓은 근거 화면을 그린다.
- 직접 입력은 'custom' 신호로 Python에 보내 공격 유형을 분류한다 (LLM, 실패 시 키워드 규칙).
- 사용자가 입력한 문장과 LLM 출력은 textContent로만 넣는다 (HTML로 해석하지 않음).
"""
import streamlit as st

HTML = """<div id="stage"></div>"""

CSS = """
:host{display:block}
*{box-sizing:border-box}
#stage{font-family:inherit;color:#1f2633;word-break:keep-all}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 320px minmax(0,1fr);gap:28px;align-items:start}
.colh{font-size:14px;color:#6b7280;margin:0 0 10px;font-weight:600;letter-spacing:.02em}
.preset{display:block;width:100%;text-align:left;border:1.5px solid #dfe3ea;background:#fff;border-radius:14px;
  padding:12px 14px;margin-bottom:10px;cursor:pointer;font:inherit;color:inherit;transition:border-color .2s,background .2s}
.preset:hover{border-color:#9aa5b8}
.preset.on{border:2.5px solid #1f2633;background:#f4f6fa;padding:11px 13px}
.preset b{display:block;font-size:17px}
.preset span{display:block;font-size:13px;color:#6b7280;margin-top:2px}
.wx{border-radius:18px;padding:18px;margin-top:6px;transition:background .5s}
.wx .row{display:flex;align-items:center;gap:14px}
.wx .emo{font-size:52px;line-height:1}
.wx .name{font-size:15px;color:#4b5563}
.wx .val{font-size:34px;font-weight:800;line-height:1.15;transition:color .5s}
.wx .val small{display:block;font-size:13px;font-weight:600;color:#4b5563}
.wx .emo{flex:none}
.wx .why{font-size:14px;line-height:1.55;margin-top:10px;color:#374151}
.phone-wrap{perspective:1200px}
.phone{width:300px;min-height:560px;margin:0 auto;border:12px solid #1f2633;border-radius:42px;background:#f5f6f8;
  padding:16px 14px 14px;cursor:pointer;display:flex;flex-direction:column;gap:10px;
  transition:transform .28s ease-in;transform-origin:center}
.phone.out{transform:rotateY(90deg)}
.phone.in{animation:flipin .35s ease-out}
@keyframes flipin{from{transform:rotateY(-90deg)}to{transform:rotateY(0)}}
.sbar{display:flex;justify-content:space-between;font-size:12px;color:#6b7280;padding:0 6px}
.noti{background:#fff;border-radius:16px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.noti.new{animation:drop .35s ease-out}
@keyframes drop{from{opacity:0;transform:translateY(-14px)}to{opacity:1;transform:none}}
.noti .from{font-size:12px;color:#6b7280;margin-bottom:4px}
.noti .txt{font-size:14px;line-height:1.45;word-break:break-all}
.meter{background:#fff;border-radius:16px;padding:14px;text-align:center}
.meter .lb{font-size:13px;color:#6b7280}
.meter .big{font-size:58px;font-weight:800;line-height:1.05;margin-top:2px;transition:color .5s}
.meter .rng{font-size:12px;color:#9ca3af}
.pill{display:inline-block;margin-top:8px;font-size:14px;font-weight:700;padding:4px 14px;border-radius:999px;color:#fff;transition:background .5s}
.ai{border-radius:16px;padding:12px;font-size:14px;line-height:1.5;transition:background .5s}
.ai .who{font-size:12px;font-weight:700;margin-bottom:3px}
.note{font-size:11.5px;color:#6b7280;text-align:center;line-height:1.4}
.hint{margin-top:auto;font-size:12px;color:#6b7280;text-align:center}
.empty{color:#9ca3af;font-size:14px;text-align:center;padding:40px 8px}
.card{display:flex;gap:12px;align-items:flex-start;width:100%;text-align:left;border:1.5px solid #dfe3ea;background:#fff;
  border-radius:14px;padding:12px 14px;margin-bottom:10px;cursor:pointer;font:inherit;color:inherit}
.card:hover{border-color:#9aa5b8}
.card.on{border:2.5px solid #1f2633;background:#f4f6fa;padding:11px 13px}
.card .ic{font-size:26px;line-height:1}
.card b{display:block;font-size:16px}
.card span{display:block;font-size:12.5px;color:#6b7280;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:100%}
.card .tx{min-width:0;flex:1}
.custom{border:1.5px dashed #c5ccd8;border-radius:14px;padding:12px 14px}
.custom.on{border:2.5px solid #1f2633;background:#f4f6fa}
.custom label{font-size:16px;font-weight:700;display:block;margin-bottom:8px}
.custom textarea{width:100%;height:62px;border:1px solid #d1d5db;border-radius:10px;padding:8px;font:inherit;font-size:14px;resize:none}
.custom .row{display:flex;justify-content:space-between;align-items:center;margin-top:8px;gap:8px}
.custom button{flex:none;white-space:nowrap;border:0;background:#1f2633;color:#fff;border-radius:10px;padding:8px 16px;font:inherit;font-size:14px;cursor:pointer}
.custom .st{font-size:12.5px;color:#4b5563;line-height:1.4}
.custom .err{color:#d23b3b}
@media (max-width:1100px){.wx .emo{font-size:40px}.wx .val{font-size:30px}}
@media (max-width:900px){
  .grid{grid-template-columns:minmax(0,1fr);gap:18px}
  .phone{width:min(300px,100%);min-height:520px}
}
"""

JS = r"""
const UI = {preset: null, msg: null, lastNonce: -1, pending: false};

function el(tag, cls, text){ const e=document.createElement(tag); if(cls) e.className=cls; if(text!=null) e.textContent=text; return e; }

export default function(component){
  const {data, parentElement, setTriggerValue} = component;
  const D = data;
  if (UI.lastNonce !== D.nonce){          // Python이 새로 정한 선택값으로 동기화
    UI.preset = D.sel.preset; UI.msg = D.sel.msg; UI.lastNonce = D.nonce; UI.pending = false;
  }
  const root = parentElement.querySelector('#stage');
  root.innerHTML = '';
  const grid = el('div','grid'); root.appendChild(grid);

  // ---------- 왼쪽: 오늘의 상태
  const L = el('div'); grid.appendChild(L);
  L.appendChild(el('div','colh','오늘의 상태'));
  const pBtns = D.presets.map((p,i)=>{
    const b = el('button','preset'); b.appendChild(el('b',null,p.label)); b.appendChild(el('span',null,p.sub));
    b.onclick = ()=>{ UI.preset=i; render(false); }; L.appendChild(b); return b;
  });
  const wx = el('div','wx'); L.appendChild(wx);
  const wxRow = el('div','row'); wx.appendChild(wxRow);
  const wxEmo = el('div','emo'); wxRow.appendChild(wxEmo);
  const wxTx = el('div'); wxRow.appendChild(wxTx);
  const wxName = el('div','name'); wxTx.appendChild(wxName);
  const wxVal = el('div','val'); wxTx.appendChild(wxVal);
  const wxWhy = el('div','why'); wx.appendChild(wxWhy);

  // ---------- 가운데: 스마트폰
  const C = el('div','phone-wrap'); grid.appendChild(C);
  const ph = el('div','phone' + (D.flipIn ? ' in' : '')); C.appendChild(ph);
  ph.title = '눌러서 근거 보기';
  ph.onclick = ()=>{
    ph.classList.remove('in'); ph.classList.add('out');
    setTimeout(()=>setTriggerValue('flip', {preset: UI.preset, msg: UI.msg}), 260);
  };

  // ---------- 오른쪽: 도착한 메시지
  const R = el('div'); grid.appendChild(R);
  R.appendChild(el('div','colh','도착한 메시지 (눌러 보세요)'));
  const cBtns = D.cards.map(c=>{
    const b = el('button','card'); b.appendChild(el('div','ic',c.icon));
    const t = el('div','tx'); t.appendChild(el('b',null,c.title)); t.appendChild(el('span',null,c.text)); b.appendChild(t);
    b.onclick = ()=>{ UI.msg=c.key; render(true); }; R.appendChild(b); return b;
  });
  const cu = el('div','custom'); R.appendChild(cu);
  cu.appendChild(el('label',null,'직접 입력'));
  const ta = el('textarea'); ta.maxLength = 300; ta.placeholder = '받은 문자를 붙여 넣어 보세요';
  if (D.custom) ta.value = D.custom.text;
  cu.appendChild(ta);
  const cuRow = el('div','row'); cu.appendChild(cuRow);
  const cuSt = el('div','st'); cuRow.appendChild(cuSt);
  const go = el('button',null,'분석'); cuRow.appendChild(go);
  ta.oninput = ()=>{ cuSt.classList.remove('err'); if(!UI.pending) cuSt.textContent=''; };
  go.onclick = (e)=>{
    e.stopPropagation();
    const t = ta.value.trim();
    if (!t){ cuSt.textContent='문자 내용을 먼저 입력해 주세요'; cuSt.classList.add('err'); return; }
    UI.pending = true; cuSt.classList.remove('err'); cuSt.textContent='AI가 공격 유형을 읽는 중…';
    setTriggerValue('custom', {text: t, preset: UI.preset});
  };

  function lv(m){ return D.levels.find(l=> m < l.cut); }

  function render(newMsg){
    const P = D.presets[UI.preset];
    pBtns.forEach((b,i)=>b.classList.toggle('on', i===UI.preset));
    cBtns.forEach((b,i)=>b.classList.toggle('on', D.cards[i].key===UI.msg));
    cu.classList.toggle('on', UI.msg==='custom');

    const tl = D.levels.find(l=>l.name===P.level) || lv(P.today);   // 잃을 것이 큰 시기 상향 규칙 반영
    wx.style.background = tl.bg;
    wxEmo.textContent = tl.emoji;
    wxName.textContent = '오늘은 ' + tl.weather + ' · ' + tl.name;
    wxVal.textContent = ''; wxVal.style.color = tl.color;
    wxVal.appendChild(document.createTextNode(P.today.toFixed(1) + '배'));
    wxVal.appendChild(el('small',null,'평소 대비 (모든 유형 평균)'));
    wxWhy.textContent = P.explain;

    if (D.custom && !UI.pending){
      const c = D.custom;
      cuSt.textContent = c.key ? ('→ ' + c.name + (c.src==='llm' ? ' (AI 판단)' : ' (키워드 규칙)')) : '→ 등록된 공격 유형과 맞지 않아요';
    }

    // 폰 화면
    ph.innerHTML = '';
    const sb = el('div','sbar'); sb.appendChild(el('span',null,'9:41')); sb.appendChild(el('span',null,'오늘의 피싱 예보')); ph.appendChild(sb);
    let card = null, key = null;
    if (UI.msg === 'custom' && D.custom){ key = D.custom.key; card = {from:'직접 입력한 문자', text: D.custom.text}; }
    else if (UI.msg){ card = D.cards.find(c=>c.key===UI.msg); key = UI.msg; }
    if (!card){
      ph.appendChild(el('div','empty','오른쪽에서 메시지를 골라 주세요'));
    } else {
      const n = el('div','noti' + (newMsg ? ' new' : '')); n.appendChild(el('div','from',card.from)); n.appendChild(el('div','txt',card.text)); ph.appendChild(n);
      const mt = el('div','meter'); ph.appendChild(mt);
      if (!key){
        mt.appendChild(el('div','lb','등록된 공격 유형과 맞지 않아요'));
        mt.appendChild(el('div','rng','이 시제품은 택배·가족 사칭·대환대출·기관 사칭 4가지 유형만 다뤄요'));
      } else {
        const s = P.scen[key], l = lv(s.mult);
        mt.appendChild(el('div','lb', D.names[key] + ' · 평소 대비'));
        const big = el('div','big', s.mult.toFixed(1) + '배'); big.style.color = l.color; mt.appendChild(big);
        mt.appendChild(el('div','rng','90% 범위 ' + s.lo.toFixed(1) + '~' + s.hi.toFixed(1) + '배'));
        const pl = el('span','pill', l.name + ' 단계'); pl.style.background = l.color; mt.appendChild(pl);
        const ai = el('div','ai'); ai.style.background = l.bg; ph.appendChild(ai);
        const who = el('div','who','AI 비서'); who.style.color = l.color; ai.appendChild(who);
        ai.appendChild(el('div',null,l.action));
      }
      ph.appendChild(el('div','note','문자 판정이 아니라, 오늘 이 사람이 이 유형에 얼마나 약한지예요'));
    }
    ph.appendChild(el('div','hint','↻ 폰을 누르면 근거를 보여드려요'));
  }
  render(false);
}
"""

_stage = st.components.v2.component("forecast_stage", html=HTML, css=CSS, js=JS)


def stage(data, on_flip, on_custom):
    return _stage(data=data, key="stage", on_flip_change=on_flip, on_custom_change=on_custom)
