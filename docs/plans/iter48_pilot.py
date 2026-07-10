#!/usr/bin/env python3
"""Iteration 48 — verification-gated delegation PILOT.
Tests the linchpin: can a strong model (GPT-5.5) author a VALID observable test
gate from the issue text ALONE? A gate is VALID iff the repro test COLLECTS,
FAILS on the buggy base code, and PASSES once the gold fix is applied.

VALIDITY WALL: the test author sees ONLY problem_statement (issue text). The gold
patch is used solely to CALIBRATE whether my gate is a good instrument (offline);
it is never shown to the author and never part of runtime candidate selection.
"""
import json, subprocess, re, os, tempfile, sys, time

WORKER = "http://127.0.0.1:8652/v1/chat/completions"
RUNS_HOST = "/home/turq/opt/moa-runs/mini-swe/runs"   # host path (for sibling docker -v)
RUNS_IN = "/work"                                      # same dir mounted in this container

import requests
ISSUES = {x['instance_id']: x for x in json.load(open(f'{RUNS_IN}/iter48-lite-0-25-issues.json'))}
CALIB  = json.load(open(f'{RUNS_IN}/iter48-CALIBRATION-ONLY-gold.json'))  # instrument-validation ONLY

_arg = sys.argv[1:]
if _arg == ['ALL']:
    PILOT = list(ISSUES.keys())
    OUTNAME = 'iter48-gatecal-all'
else:
    PILOT = _arg or ['astropy__astropy-12907', 'django__django-10914', 'django__django-11039']
    OUTNAME = 'iter48-pilot-gatecal'

def img(iid): return f"swebench/sweb.eval.x86_64.{iid.replace('__','_1776_')}:latest"

def llm(messages, max_tokens=2200):
    r = requests.post(WORKER, json={"model":"moa:gpt55-plan-lane","messages":messages,
                                    "max_tokens":max_tokens,"temperature":0}, timeout=240)
    j = r.json()
    return j['choices'][0]['message']['content'], j.get('usage',{})

def extract_code(text):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return (m[-1] if m else text).strip()

os.makedirs(f'{RUNS_IN}/_gate_tmp', exist_ok=True)

def run_test(iid, test_code, gold_patch=None, timeout=360):
    # unittest-based self-contained script (stdlib — no pytest/runtests needed);
    # run via `python test_repro.py`. Django/astropy deps come from the testbed env.
    d = tempfile.mkdtemp(dir=f'{RUNS_IN}/_gate_tmp')
    open(os.path.join(d,'test_repro.py'),'w').write(test_code)
    apply = ""
    if gold_patch is not None:
        open(os.path.join(d,'gold.patch'),'w').write(gold_patch)
        apply = "(git apply /injected/gold.patch 2>/dev/null || patch -p1 --fuzz=5 < /injected/gold.patch >/dev/null 2>&1); "
    hostd = d.replace(RUNS_IN, RUNS_HOST, 1)
    inner = (f"source /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed; cd /testbed; "
             f"{apply}python /injected/test_repro.py 2>&1")
    cmd = ["docker","run","--rm","-v",f"{hostd}:/injected",img(iid),"bash","-lc",inner]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or '')[-3500:]
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"

def classify(rc, out):
    # unittest semantics: OK=pass; FAILED(failures=)=bug reproduced; FAILED(errors=)/import=setup broken
    if rc == 124: return 'timeout'
    low = out.lower()
    if 'ran 0 tests' in low or 'no module named' in low or 'importerror' in low or 'modulenotfounderror' in low or 'syntaxerror' in low or 'cannot import' in low:
        return 'collect_error'
    ended_ok = ('\nok' in low or low.strip().endswith('ok')) and 'failed' not in low
    if rc == 0 and ended_ok: return 'pass'
    if 'failed (' in low:
        # failures = assertion (bug reproduced / correct-behaviour unmet); errors = setup exception
        has_fail = 'failures=' in low
        has_err  = 'errors=' in low
        if has_fail and not has_err: return 'fail'
        if has_err and not has_fail: return 'collect_error'
        return 'fail' if has_fail else 'collect_error'
    if rc == 0: return 'pass'
    return 'collect_error'

AUTHOR_SYS = ("You write a REGRESSION TEST reproducing a bug from a GitHub issue for the repo "
  "'{repo}' at a fixed commit. STRICT requirements: (1) a SINGLE self-contained script using the stdlib "
  "'unittest' module — NO pytest (it is not installed). Define unittest.TestCase subclass(es) and end the "
  "file with `if __name__ == \"__main__\": unittest.main()` so it runs via `python test.py`. (2) import "
  "only the installed '{pkg}' package + stdlib. (3) reproduce the SPECIFIC bug and assert the CORRECT "
  "behaviour, so it FAILS (assertion failure) on the current buggy code and PASSES once fixed — keep the "
  "assertion NARROW to exactly what the fix changes, do not over-specify unrelated behaviour. (4) no "
  "network, no external files. For Django: call django.conf.settings.configure(...) with minimal settings "
  "then django.setup() BEFORE importing anything using settings; add an in-memory sqlite DATABASES "
  "(':memory:') only if the ORM is required; prefer testing the pure function/class named in the issue. "
  "Output ONLY the script in one ```python code block.")

def author_test(iid):
    iss = ISSUES[iid]; pkg = iss['repo'].split('/')[-1]
    msgs = [{"role":"system","content":AUTHOR_SYS.format(repo=iss['repo'],pkg=pkg)},
            {"role":"user","content":f"Issue for {iss['repo']} @ {iss['base_commit'][:12]}:\n\n{iss['problem_statement']}"}]
    total_usage = {}
    text, u = llm(msgs); total_usage = u
    code = extract_code(text)
    # up to 2 repair rounds on collection/import errors against BASE (buggy) code
    for rnd in range(2):
        rc, out = run_test(iid, code)
        cls = classify(rc, out)
        if cls in ('fail','pass'):   # it collected & ran
            return code, cls, out, total_usage, rnd
        msgs += [{"role":"assistant","content":f"```python\n{code}\n```"},
                 {"role":"user","content":f"Running `python test.py` against the CURRENT code gave:\n\n{out}\n\n"
                  "The test must RUN to a real assertion result (a plain assertion FAILURE is expected and "
                  "good; import/setup/collection ERRORS are not). Fix ONLY those so it runs under stdlib "
                  "unittest via `python test.py`. Output ONLY the corrected script in one ```python block."}]
        text, u = llm(msgs)
        for k,v in (u or {}).items():
            if isinstance(v,(int,float)): total_usage[k]=total_usage.get(k,0)+v
        code = extract_code(text)
    rc, out = run_test(iid, code)
    return code, classify(rc, out), out, total_usage, 2

report = []
for iid in PILOT:
    t0 = time.time()
    print(f"\n=== {iid} ({ISSUES[iid]['repo']}) ===", flush=True)
    code, base_cls, base_out, usage, rounds = author_test(iid)
    print(f"  authored (repair rounds={rounds}), base verdict: {base_cls}", flush=True)
    gold_cls = None
    if base_cls == 'fail':
        rc_g, out_g = run_test(iid, code, gold_patch=CALIB[iid]['gold_patch'])
        gold_cls = classify(rc_g, out_g)
        print(f"  gold-patched verdict: {gold_cls}", flush=True)
    valid = (base_cls == 'fail' and gold_cls == 'pass')
    print(f"  >> VALID GATE: {valid}", flush=True)
    open(f'{RUNS_IN}/_gate_tmp/{iid}.test.py','w').write(code)
    report.append({'iid':iid,'repo':ISSUES[iid]['repo'],'base':base_cls,'gold':gold_cls,
                   'valid':valid,'repair_rounds':rounds,'author_usage':usage,
                   'secs':round(time.time()-t0,1),'base_out_tail':base_out[-600:]})

json.dump(report, open(f'{RUNS_IN}/{OUTNAME}.json','w'), indent=1)
json.dump({r['iid']: r['valid'] for r in report}, open(f'{RUNS_IN}/{OUTNAME}-validmap.json','w'), indent=1)
print("\n==== PILOT SUMMARY ====")
for r in report:
    print(f"  {r['iid']:28s} {r['repo'].split('/')[-1]:8s} base={str(r['base']):12s} gold={str(r['gold'])} valid={r['valid']} ({r['secs']}s, {r['repair_rounds']} repairs)")
nv = sum(1 for r in report if r['valid'])
print(f"\nVALID GATES: {nv}/{len(report)}  — linchpin viability signal")
