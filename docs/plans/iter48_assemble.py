#!/usr/bin/env python3
"""Iteration 48 — assemble verify-escalate final predictions.
For each instance: gate ACCEPT -> use flash candidate patch; ESCALATE -> use
Fable's patch (fresh run on the escalate subset). Writes combined preds.json.
"""
import json, os
RUNS='/work'
gate = json.load(open(f'{RUNS}/iter48-gate-flash.json'))
decision = {r['iid']: r['decision'] for r in gate}
def load_preds(p):
    d=json.load(open(p)); d=d if isinstance(d,list) else list(d.values())
    return {x['instance_id']: x.get('model_patch','') for x in d}
flash = load_preds(f'{RUNS}/iter48-flash/preds.json')
fable = load_preds(f'{RUNS}/iter48-fable-esc/preds.json') if os.path.exists(f'{RUNS}/iter48-fable-esc/preds.json') else {}

out=[]; src={}
for iid in sorted(flash):
    if decision.get(iid)=='escalate' and iid in fable:
        patch, s = fable[iid], 'fable'
    elif decision.get(iid)=='escalate':
        patch, s = flash.get(iid,''), 'flash(esc-nofable)'   # fallback if Fable produced nothing
    else:
        patch, s = flash[iid], 'flash'
    src[iid]=s
    out.append({'model_name_or_path':'openai/moa:vgd-verify-escalate','instance_id':iid,'model_patch':patch})
os.makedirs(f'{RUNS}/iter48-vgd', exist_ok=True)
json.dump(out, open(f'{RUNS}/iter48-vgd/preds.json','w'), indent=1)
json.dump(src, open(f'{RUNS}/iter48-vgd-source.json','w'), indent=1)
from collections import Counter
print('assembled', len(out), 'preds. sources:', dict(Counter(src.values())))
print('escalated->fable:', [i for i,s in src.items() if s=='fable'])
