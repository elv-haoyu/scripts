import json
import pickle
from datetime import datetime
from pathlib import Path

from google.cloud.firestore_v1 import _helpers

def load_json(fi):
    with open(fi, 'r') as f:
        d = json.load(f)
    return d

def load_pkl(fi):
    with open(fi, 'rb') as f:
        d = pickle.load(f)
    return d

def dump_json(fi, outs:dict):
    with open(fi, 'w') as f:
        json.dump(outs, f, indent='\t')
    return

def dump_pkl(fi, outs):
    with open(fi, 'wb') as f:
        pickle.dump(outs, f, protocol=pickle.HIGHEST_PROTOCOL)
    return

def json_dumps(dp:Path, output:dict):
    dp.write_text(json.dumps(output), encoding='utf8')

def json_loads(dp:Path):
    return json.loads(dp.read_text(encoding="UTF-8"))


class FirestoreEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _helpers.DatetimeWithNanoseconds):
            return obj.rfc3339()  # Converts to an RFC 3339 formatted string
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

if __name__ == '__main__':
    data = {
        "timestamp": _helpers.DatetimeWithNanoseconds(datetime.now(), nanosecond=123456789)
    }

    json_string = json.dumps(data, cls=FirestoreEncoder)
    print(json_string)
