#!/usr/bin/env python3
"""Build a local image-based app for ontology adjudication."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

from build_ontology_review_app import build_records


FINAL_DECISIONS = ("flame", "smoke", "cigarette", "smoking_action", "invalid")


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ontology 最终裁决</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17202a;background:#f4f5f7}body{margin:0}.top{position:sticky;top:0;z-index:5;background:#17202a;color:white;padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}.top strong{font-size:18px}.top input{padding:7px;border-radius:5px;border:0}.wrap{max-width:1320px;margin:18px auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,2fr) minmax(330px,1fr);gap:18px}.panel{background:white;border:1px solid #d5d8dc;border-radius:10px;padding:14px}.meta{font-size:13px;overflow-wrap:anywhere;margin-bottom:8px}.image{position:relative;display:inline-block;max-width:100%;background:#111}.image img{display:block;max-width:100%;max-height:72vh}.box{position:absolute;box-sizing:border-box;border:2px solid #95a5a6;color:white;text-shadow:0 1px 2px #000;pointer-events:none}.box.target{border:4px solid #e74c3c}.box b{background:rgba(0,0,0,.72);font-size:12px}.review{border:1px solid #d5d8dc;border-radius:8px;padding:9px;margin:8px 0;background:#fafafa}.review strong{display:inline-block;min-width:48px}.reason{padding:8px;border-radius:6px;background:#fef5e7;color:#7d4e00}.decisions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.decision{padding:10px;border:2px solid #ccd1d1;border-radius:8px;background:white;text-align:left;cursor:pointer}.decision.active{border-color:#117864;background:#d5f5e3}.decision kbd{float:right}.nav,.actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}button{cursor:pointer}.nav button,.actions button{padding:8px 11px}.notes{width:100%;min-height:88px;box-sizing:border-box;margin-top:10px}.progress{height:8px;background:#566573;border-radius:8px;overflow:hidden;min-width:180px}.progress span{display:block;height:100%;background:#2ecc71}.warning{color:#922b21;font-weight:600}.help{font-size:13px;line-height:1.5}.open-original{display:inline-block;margin-top:8px}@media(max-width:850px){.wrap{grid-template-columns:1fr}.image img{max-height:58vh}}
</style></head><body>
<div class="top"><strong>Ontology 最终裁决（21条）</strong><label>Adjudicator <input id="adjudicator" placeholder="姓名或代号"></label><span id="counter"></span><div class="progress"><span id="bar"></span></div></div>
<main class="wrap"><section class="panel"><div id="meta" class="meta"></div><div class="image" id="imageWrap"><img id="image" alt="adjudication image"></div><br><a id="original" class="open-original" target="_blank">打开原始分辨率图片 ↗</a><p class="warning" id="imageError"></p></section>
<aside class="panel"><div id="reason" class="reason"></div><div id="review1" class="review"></div><div id="review2" class="review"></div><h3>最终类别（不能保留 ambiguous）</h3><div class="decisions" id="decisions"></div><textarea id="notes" class="notes" placeholder="写明最终判断依据；invalid 必须说明原因"></textarea><div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button><button id="pending">下一个未完成</button></div><div class="actions"><button id="exportCsv">导出裁决 CSV</button><button id="exportJson">备份 JSON</button><label><input id="importJson" type="file" accept="application/json" hidden><button id="importButton" type="button">导入 JSON</button></label></div><hr><div class="help"><b>只判断红框</b>；灰框只是同图中的其他标注，不用处理。快捷键：1 flame；2 smoke；3 cigarette；4 smoking_action；5 invalid；←/→ 导航。点击“打开原始分辨率图片”可放大核查。</div></aside></main>
<script>
const RECORDS=__RECORDS__;const DECISIONS=__DECISIONS__;const storageKey="ontology-adjudication-v1:"+RECORDS.map(r=>r.recordId).join("|");let state={adjudicator:"",answers:{}};let index=0;
try{const saved=localStorage.getItem(storageKey);if(saved)state=JSON.parse(saved)}catch(e){}const $=id=>document.getElementById(id);$("adjudicator").value=state.adjudicator||"";
function esc(v){return String(v??"").replace(/[&<>\"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}function answer(){return state.answers[RECORDS[index].recordId]||{decision:"",notes:"",adjudicatedAt:""}}function save(){state.adjudicator=$("adjudicator").value.trim();localStorage.setItem(storageKey,JSON.stringify(state));renderProgress()}
function setDecision(value){const a=answer();a.decision=value;a.adjudicatedAt=new Date().toISOString();state.answers[RECORDS[index].recordId]=a;save();render();setTimeout(nextPending,120)}
function render(){const r=RECORDS[index],a=answer();$("meta").textContent=`${index+1}/${RECORDS.length} | ${r.recordId} | ${r.split}/${r.image} | annotation ${r.annotationIndex}`;$("reason").innerHTML=`<b>进入裁决的原因：</b>${esc(r.reason)}`;$("review1").innerHTML=`<strong>${esc(r.reviewer1||"审核人1")}</strong> ${esc(r.decision1||"未复核")}<br><small>${esc(r.notes1||"无备注")}</small>`;$("review2").innerHTML=`<strong>${esc(r.reviewer2||"审核人2")}</strong> ${esc(r.decision2||"未复核")}<br><small>${esc(r.notes2||"无备注")}</small>`;const img=$("image");img.src=r.imageSrc;$("original").href=r.imageSrc;img.onerror=()=>{$("imageError").textContent="图片加载失败："+r.imageSrc};img.onload=()=>{$("imageError").textContent=""};const wrap=$("imageWrap");wrap.querySelectorAll(".box").forEach(x=>x.remove());r.boxes.forEach(b=>{const s=document.createElement("span");s.className="box "+(b.index===r.annotationIndex?"target":"context");s.style.cssText=`left:${b.left}%;top:${b.top}%;width:${b.width}%;height:${b.height}%`;s.innerHTML=`<b>${b.classId}:${b.index}</b>`;wrap.appendChild(s)});document.querySelectorAll(".decision").forEach(b=>b.classList.toggle("active",b.dataset.value===a.decision));$("notes").value=a.notes||"";renderProgress()}
function renderProgress(){const done=RECORDS.filter(r=>state.answers[r.recordId]?.decision).length;$("counter").textContent=`完成 ${done}/${RECORDS.length}`;$("bar").style.width=(100*done/RECORDS.length)+"%"}function move(delta){index=Math.max(0,Math.min(RECORDS.length-1,index+delta));render()}function nextPending(){for(let n=1;n<=RECORDS.length;n++){const j=(index+n)%RECORDS.length;if(!state.answers[RECORDS[j].recordId]?.decision){index=j;render();return}}move(1)}
function csvEscape(v){const s=String(v??"");return /[\",\n\r]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}function download(name,text,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function validate(){save();if(!state.adjudicator){alert("请先填写裁决人姓名或代号");return false}const missing=RECORDS.filter(r=>!state.answers[r.recordId]?.decision);if(missing.length){alert(`还有 ${missing.length} 条未完成`);return false}const noNotes=RECORDS.filter(r=>{const a=state.answers[r.recordId];return (!a.notes?.trim())});if(noNotes.length){alert(`还有 ${noNotes.length} 条没有填写裁决依据`);return false}return true}
function exportCsv(){if(!validate())return;const rows=[["record_id","adjudicator","final_decision","notes","adjudicated_at"]];RECORDS.forEach(r=>{const a=state.answers[r.recordId];rows.push([r.recordId,state.adjudicator,a.decision,a.notes,a.adjudicatedAt])});download("ontology_adjudication_results.csv",rows.map(row=>row.map(csvEscape).join(",")).join("\n"),"text/csv;charset=utf-8")}
function exportJson(){save();download("ontology_adjudication_backup.json",JSON.stringify({recordIds:RECORDS.map(r=>r.recordId),state},null,2),"application/json")}
DECISIONS.forEach((value,i)=>{const b=document.createElement("button");b.className="decision";b.dataset.value=value;b.innerHTML=`${value}<kbd>${i+1}</kbd>`;b.onclick=()=>setDecision(value);$("decisions").appendChild(b)});$("adjudicator").oninput=save;$("notes").oninput=()=>{const a=answer();a.notes=$("notes").value;state.answers[RECORDS[index].recordId]=a;save()};$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("pending").onclick=nextPending;$("exportCsv").onclick=exportCsv;$("exportJson").onclick=exportJson;$("importButton").onclick=()=>$("importJson").click();$("importJson").onchange=e=>{const f=e.target.files[0];if(!f)return;const reader=new FileReader();reader.onload=()=>{try{state=JSON.parse(reader.result).state;save();render()}catch(err){alert("JSON 无法读取："+err)}};reader.readAsText(f)};document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;if(e.key>="1"&&e.key<="5")setDecision(DECISIONS[Number(e.key)-1]);else if(e.key==="ArrowLeft")move(-1);else if(e.key==="ArrowRight")move(1)});render();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue_csv", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()

    with args.queue_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Adjudication queue is empty")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise SystemExit("record_id is not unique")

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(args.dataset, args.output_html, rows)
    for record, row in zip(records, rows):
        record.update(
            reviewer1=row["reviewer_1"],
            decision1=row["decision_1"],
            notes1=row["notes_1"],
            reviewer2=row["reviewer_2"],
            decision2=row["decision_2"],
            notes2=row["notes_2"],
            reason=row["adjudication_reason"],
        )

    document = (
        TEMPLATE.replace("__RECORDS__", json.dumps(records, ensure_ascii=False))
        .replace("__DECISIONS__", json.dumps(FINAL_DECISIONS, ensure_ascii=False))
    )
    args.output_html.write_text(document, encoding="utf-8")
    print(f"Wrote {len(records)} adjudication records to {args.output_html}")


if __name__ == "__main__":
    main()
