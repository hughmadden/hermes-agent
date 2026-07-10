#!/usr/bin/env python3
"""Iteration 48 — gate candidate patches against the FROZEN observable tests.
Usage: iter48_gate.py <preds.json> <label>
Applies each candidate patch to the base instance, runs the frozen repro test
(authored earlier from ISSUE TEXT ONLY), and records the gate verdict.
VALIDITY: the gate is the observable signal (repro test). FAIL_TO_PASS is used
ONLY in post-hoc analysis (gate precision), never to make the accept decision.
"""
import json, subprocess, os, tempfile, sys, time
RUNS_IN='/work'; RUNS_HOST='/home/turq/opt/moa-runs/mini-swe/runs'
preds_path=sys.argv[1]; label=sys.argv[2]
validmap=json.load(open(f'{RUNS_IN}/iter48-gatecal-all-validmap.json'))
_p=json.load(open(preds_path)); _p=_p if isinstance(_p,list) else list(_p.values())
PATCH={x['instance_id']:x.get('model_patch','') for x in _p}
os.makedirs(f'{RUNS_IN}/_gate_tmp',exist_ok=True)

def img(iid): return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest"

def classify(rc,out):
    low=out.lower()
    if rc==124: return 'timeout'
    if 'ran 0 tests' in low or 'no module named' in low or 'importerror' in low or 'modulenotfounderror' in low or 'syntaxerror' in low or 'cannot import' in low: return 'collect_error'
    if rc==0 and 'failed' not in low: return 'pass'
    if 'failed (' in low:
        return 'fail' if ('failures=' in low and 'errors=' not in low) else ('collect_error' if 'errors=' in low and 'failures=' not in low else 'fail')
    return 'pass' if rc==0 else 'collect_error'

def gate_candidate(iid, patch, timeout=360):
    """apply candidate patch to base, run frozen test. pass => candidate satisfies repro."""
    tf=f'{RUNS_IN}/_gate_tmp/{iid}.test.py'
    if not os.path.exists(tf): return 'no_gate','no frozen test'
    d=tempfile.mkdtemp(dir=f'{RUNS_IN}/_gate_tmp')
    open(f'{d}/test_repro.py','w').write(open(tf).read())
    open(f'{d}/cand.patch','w').write(patch or '')
    hostd=d.replace(RUNS_IN,RUNS_HOST,1)
    apply="(git apply /injected/cand.patch 2>/dev/null || patch -p1 --fuzz=5 < /injected/cand.patch >/dev/null 2>&1); " if (patch or '').strip() else ""
    inner=(f"source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed; cd /testbed; "
           f"{apply}python /injected/test_repro.py 2>&1")
    cmd=["docker","run","--rm","-v",f"{hostd}:/injected",img(iid),"bash","-lc",inner]
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
        return classify(p.returncode,p.stdout or ''), (p.stdout or '')[-500:]
    except subprocess.TimeoutExpired:
        return 'timeout','TIMEOUT'

report=[]
for iid in sorted(PATCH):
    valid=validmap.get(iid,False)
    patch=PATCH[iid]; empty=not (patch or '').strip()
    if not valid:
        verdict='gate_invalid'; out=''
    else:
        verdict,out=gate_candidate(iid,patch)
    decision = 'accept' if verdict=='pass' else 'escalate'   # gate-pass keeps candidate; else escalate to Fable
    print(f"  {iid:28s} gate_valid={valid} empty_patch={empty} verdict={verdict:13s} -> {decision}",flush=True)
    report.append({'iid':iid,'gate_valid':valid,'empty_patch':empty,'verdict':verdict,'decision':decision})
json.dump(report,open(f'{RUNS_IN}/iter48-gate-{label}.json','w'),indent=1)
acc=[r['iid'] for r in report if r['decision']=='accept']
esc=[r['iid'] for r in report if r['decision']=='escalate']
print(f"\n{label}: ACCEPT(offload)={len(acc)}/25  ESCALATE={len(esc)}/25")
print("ESCALATE_IDS:", '|'.join(esc))
