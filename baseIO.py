from pathlib import Path, PurePath
import json
import pickle

def load_json(fi):
    with open(fi, 'r') as f:
        d = json.load(f)
    return d

def dump_json(fi, outs:dict):
    with open(fi, 'w') as f:
        d = json.dump(outs, f, indent='\t')
    return d

def json_dumps(dp:Path, output:dict):
    dp.write_text(json.dumps(output), encoding='utf8')

def json_loads(dp:Path):
    return json.loads(dp.read_text(encoding="UTF-8"))

def load_pkl(fi):
    with open(fi, 'rb') as f:
        d = pickle.load(f)
    return d