#!/usr/bin/env python3
"""Build a local, keyboard-friendly ontology review application.

The application stores progress in browser localStorage and exports reviewer
results as CSV. It never changes dataset labels or the master review queue.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path


DECISIONS = ("flame", "smoke", "cigarette", "smoking_action", "invalid", "ambiguous")
SPLIT_ORDER = {"train": 0, "valid": 1, "test": 2}


def parse_boxes(label_path: Path) -> list[dict[str, float | int]]:
    boxes: list[dict[str, float | int]] = []
    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        class_id = int(fields[0])
        coordinates = list(map(float, fields[1:]))
        if len(coordinates) == 4:
            x_center, y_center, width, height = coordinates
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            xs, ys = coordinates[0::2], coordinates[1::2]
            left, right, top, bottom = min(xs), max(xs), min(ys), max(ys)
            x_center, y_center = (left + right) / 2, (top + bottom) / 2
            width, height = right - left, bottom - top
        else:
            raise ValueError(f"Unsupported YOLO geometry: {label_path}:{index}")
        boxes.append(
            {
                "index": index,
                "classId": class_id,
                "left": (x_center - width / 2) * 100,
                "top": (y_center - height / 2) * 100,
                "width": width * 100,
                "height": height * 100,
            }
        )
    return boxes


def area_bin(rows: list[dict[str, str]], row: dict[str, str]) -> int:
    areas = sorted(float(item["box_area_fraction"]) for item in rows)
    thresholds = [areas[math.floor((len(areas) - 1) * q)] for q in (0.25, 0.5, 0.75)]
    area = float(row["box_area_fraction"])
    return sum(area > threshold for threshold in thresholds)


def deterministic_sample(rows: list[dict[str, str]], size: int, seed: str) -> list[dict[str, str]]:
    if size <= 0 or size >= len(rows):
        return list(rows)

    # Use at most one annotation per source in the agreement sample so adjacent
    # variants do not make agreement look more precise than it is.
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_key"] or f"{row['split']}/{row['image']}"].append(row)
    candidates = []
    for source, source_rows in by_source.items():
        candidates.append(
            min(
                source_rows,
                key=lambda row: hashlib.sha256(f"{seed}:record:{row['record_id']}".encode()).hexdigest(),
            )
        )
    if size > len(candidates):
        raise ValueError(f"Requested {size} source-independent rows, only {len(candidates)} available")

    strata: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        strata[(row["split"], area_bin(candidates, row))].append(row)

    exact = {key: size * len(items) / len(candidates) for key, items in strata.items()}
    quotas = {key: min(len(strata[key]), math.floor(value)) for key, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(strata, key=lambda key: (exact[key] - quotas[key], len(strata[key])), reverse=True)
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < len(strata[key]):
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the requested sample")

    selected = []
    for key, items in strata.items():
        ranked = sorted(
            items,
            key=lambda row: hashlib.sha256(
                f"{seed}:sample:{row['source_key']}:{row['record_id']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[key]])
    return sorted(selected, key=lambda row: (SPLIT_ORDER.get(row["split"], 99), row["image"], int(row["annotation_index"])))


def write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_records(dataset: Path, output: Path, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    relative_dataset = os.path.relpath(dataset.resolve(), output.parent.resolve())
    box_cache: dict[tuple[str, str], list[dict[str, float | int]]] = {}
    records = []
    for row in rows:
        key = (row["split"], Path(row["image"]).stem)
        if key not in box_cache:
            label_path = dataset / row["split"] / "labels" / f"{key[1]}.txt"
            box_cache[key] = parse_boxes(label_path)
        records.append(
            {
                "recordId": row["record_id"],
                "sourceKey": row["source_key"],
                "split": row["split"],
                "image": row["image"],
                "annotationIndex": int(row["annotation_index"]),
                "area": float(row["box_area_fraction"]),
                "imageSrc": f"{relative_dataset}/{row['split']}/images/{row['image']}",
                "boxes": box_cache[key],
            }
        )
    return records


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#17202a;background:#f4f5f7}body{margin:0}.top{position:sticky;top:0;z-index:5;background:#17202a;color:white;padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}.top strong{font-size:18px}.top input{padding:7px;border-radius:5px;border:0}.wrap{max-width:1200px;margin:18px auto;padding:0 16px;display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:18px}.panel{background:white;border:1px solid #d5d8dc;border-radius:10px;padding:14px}.meta{font-size:13px;overflow-wrap:anywhere}.image{position:relative;display:inline-block;max-width:100%;background:#111}.image img{display:block;max-width:100%;max-height:68vh}.box{position:absolute;box-sizing:border-box;border:2px solid #95a5a6;color:white;text-shadow:0 1px 2px #000}.box.target{border:4px solid #e74c3c}.box b{background:rgba(0,0,0,.7);font-size:12px}.decisions{display:grid;grid-template-columns:1fr 1fr;gap:8px}.decision{padding:10px;border:2px solid #ccd1d1;border-radius:8px;background:white;text-align:left;cursor:pointer}.decision.active{border-color:#117864;background:#d5f5e3}.decision kbd{float:right}.nav{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}button{cursor:pointer}.nav button,.actions button{padding:8px 11px}.notes{width:100%;min-height:80px;box-sizing:border-box;margin-top:10px}.legend{font-size:13px;line-height:1.45}.progress{height:8px;background:#566573;border-radius:8px;overflow:hidden;min-width:180px}.progress span{display:block;height:100%;background:#2ecc71}.warning{color:#922b21;font-weight:600}.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}@media(max-width:800px){.wrap{grid-template-columns:1fr}.image img{max-height:55vh}}
</style></head><body>
<div class="top"><strong>__TITLE__</strong><label>Reviewer <input id="reviewer" placeholder="姓名或代号"></label><span id="counter"></span><div class="progress"><span id="bar"></span></div></div>
<main class="wrap"><section class="panel"><div id="meta" class="meta"></div><div class="image" id="imageWrap"><img id="image" alt="review image"></div><p class="warning" id="imageError"></p></section>
<aside class="panel"><h3>选择红框的语义</h3><div class="decisions" id="decisions"></div><textarea id="notes" class="notes" placeholder="ambiguous / invalid 必须写原因；框偏松可写 box_loose；不可用框写 bad_box"></textarea><div class="nav"><button id="prev">← 上一条</button><button id="next">下一条 →</button><button id="pending">下一个未完成</button></div><div class="actions"><button id="exportCsv">导出 CSV</button><button id="exportJson">备份 JSON</button><label><input id="importJson" type="file" accept="application/json" hidden><button id="importButton" type="button">导入 JSON</button></label><button id="resetState" type="button">清空本页进度</button></div><hr><div class="legend"><b>快捷键</b>：1 flame；2 smoke；3 cigarette；4 smoking_action；5 invalid；6 ambiguous；←/→ 导航。<p>Reviewer 只判断可见像素。文件名和历史类别不是证据。Reviewer 2 不得查看 Reviewer 1 的答案。</p><p><a href="__GUIDELINES__" target="_blank">打开完整判定规范</a></p></div></aside></main>
<script>
const RECORDS=__RECORDS__;
const SLOT=__SLOT__;
const DECISIONS=["flame","smoke","cigarette","smoking_action","invalid","ambiguous"];
const storageKey="ontology-audit-v2:"+SLOT+":"+RECORDS.map(r=>r.recordId).join("|").slice(0,500);
let state={reviewer:"",answers:{}};let index=0;
try{const saved=localStorage.getItem(storageKey);if(saved)state=JSON.parse(saved)}catch(e){}
const $=id=>document.getElementById(id);$("reviewer").value=state.reviewer||"";
function save(){state.reviewer=$("reviewer").value.trim();localStorage.setItem(storageKey,JSON.stringify(state));renderProgress()}
function answer(){return state.answers[RECORDS[index].recordId]||{decision:"",notes:"",reviewedAt:""}}
function setDecision(value){const a=answer();a.decision=value;a.reviewedAt=new Date().toISOString();state.answers[RECORDS[index].recordId]=a;save();render();setTimeout(nextPending,120)}
function render(){const r=RECORDS[index],a=answer();$("meta").textContent=`${index+1}/${RECORDS.length} | ${r.recordId} | ${r.split}/${r.image} | annotation ${r.annotationIndex} | source ${r.sourceKey}`;const img=$("image");img.src=r.imageSrc;img.onerror=()=>{$("imageError").textContent="图片加载失败："+r.imageSrc};img.onload=()=>{$("imageError").textContent=""};const wrap=$("imageWrap");wrap.querySelectorAll(".box").forEach(x=>x.remove());r.boxes.forEach(b=>{const s=document.createElement("span");s.className="box "+(b.index===r.annotationIndex?"target":"context");s.style.cssText=`left:${b.left}%;top:${b.top}%;width:${b.width}%;height:${b.height}%`;s.innerHTML=`<b>${b.classId}:${b.index}</b>`;wrap.appendChild(s)});document.querySelectorAll(".decision").forEach(b=>b.classList.toggle("active",b.dataset.value===a.decision));$("notes").value=a.notes||"";renderProgress()}
function renderProgress(){const done=RECORDS.filter(r=>state.answers[r.recordId]?.decision).length;$("counter").textContent=`完成 ${done}/${RECORDS.length}`;$("bar").style.width=(100*done/RECORDS.length)+"%"}
function move(delta){index=Math.max(0,Math.min(RECORDS.length-1,index+delta));render()}
function nextPending(){for(let n=1;n<=RECORDS.length;n++){const j=(index+n)%RECORDS.length;if(!state.answers[RECORDS[j].recordId]?.decision){index=j;render();return}}move(1)}
function csvEscape(v){const s=String(v??"");return /[",\n\r]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}
function download(name,text,type){const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function exportCsv(){save();if(!state.reviewer){alert("请先填写 Reviewer 姓名或代号");return}const rows=[["record_id","reviewer_slot","reviewer","decision","notes","reviewed_at"]];RECORDS.forEach(r=>{const a=state.answers[r.recordId]||{};rows.push([r.recordId,SLOT,state.reviewer,a.decision||"",a.notes||"",a.reviewedAt||""])});download(`ontology_${SLOT}_results.csv`,rows.map(row=>row.map(csvEscape).join(",")).join("\n"),"text/csv;charset=utf-8")}
function exportJson(){save();download(`ontology_${SLOT}_backup.json`,JSON.stringify({slot:SLOT,recordIds:RECORDS.map(r=>r.recordId),state},null,2),"application/json")}
DECISIONS.forEach((value,i)=>{const b=document.createElement("button");b.className="decision";b.dataset.value=value;b.innerHTML=`${value}<kbd>${i+1}</kbd>`;b.onclick=()=>setDecision(value);$("decisions").appendChild(b)});
$("reviewer").oninput=save;$("notes").oninput=()=>{const a=answer();a.notes=$("notes").value;state.answers[RECORDS[index].recordId]=a;save()};$("prev").onclick=()=>move(-1);$("next").onclick=()=>move(1);$("pending").onclick=nextPending;$("exportCsv").onclick=exportCsv;$("exportJson").onclick=exportJson;$("importButton").onclick=()=>$("importJson").click();$("resetState").onclick=()=>{if(confirm("确定清空本页全部审核进度？请先导出备份。")){state={reviewer:"",answers:{}};index=0;localStorage.removeItem(storageKey);$("reviewer").value="";render()}};$("importJson").onchange=event=>{const file=event.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{try{const incoming=JSON.parse(reader.result);state=incoming.state;save();render()}catch(e){alert("JSON 无法读取："+e)}};reader.readAsText(file)};
document.addEventListener("keydown",e=>{if(e.target.tagName==="TEXTAREA"||e.target.tagName==="INPUT")return;if(e.key>="1"&&e.key<="6")setDecision(DECISIONS[Number(e.key)-1]);else if(e.key==="ArrowLeft")move(-1);else if(e.key==="ArrowRight")move(1)});render();
</script></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("master_csv", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_html", type=Path)
    parser.add_argument("--slot", choices=("reviewer1", "reviewer2"), required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", default="ontology-audit-v1")
    parser.add_argument("--queue-csv", type=Path)
    parser.add_argument("--guidelines", type=Path)
    args = parser.parse_args()

    with args.master_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["record_id"] for row in rows}) != len(rows):
        raise SystemExit("record_id is not unique")
    selected = deterministic_sample(rows, args.sample_size, args.sample_seed)
    if args.queue_csv:
        write_queue(args.queue_csv, selected)

    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    records = build_records(args.dataset, args.output_html, selected)
    guidelines = args.guidelines or Path("../../research/ONTOLOGY_AUDIT_GUIDELINES.md")
    relative_guidelines = os.path.relpath(guidelines.resolve(), args.output_html.parent.resolve())
    title = f"Ontology audit - {args.slot} ({len(records)} records)"
    document = (
        TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__RECORDS__", json.dumps(records, ensure_ascii=False))
        .replace("__SLOT__", json.dumps(args.slot))
        .replace("__GUIDELINES__", html.escape(relative_guidelines))
    )
    args.output_html.write_text(document, encoding="utf-8")
    digest = hashlib.sha256("|".join(record["recordId"] for record in records).encode()).hexdigest()
    print(f"Wrote {len(records)} records to {args.output_html}")
    print(f"Queue digest: {digest}")


if __name__ == "__main__":
    main()
