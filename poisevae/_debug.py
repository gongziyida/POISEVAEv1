import sys, io, time
import re
import pandas as pd
from contextlib import redirect_stdout

class Capturing(list):
    def __init__(self, fname=None, disable_print=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fname = fname
        self.disable_print = disable_print
        
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = io.StringIO()
        return self
    def __exit__(self, *args):
        if not self.disable_print:
            out = self._stringio.getvalue()
            with open(self.fname, 'a') as f:
                f.write(out)
            self.extend(out.splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def catch_print(func):
    fname = 'out %s.txt' % time.strftime('%m%d%H%M%S')
    def wrapper(*args, disable_print=False, **kwargs):
        f = io.StringIO()
        with redirect_stdout(f):
            ret = func(*args, **kwargs)
        out = f.getvalue()
        if not disable_print:
            with open(fname, 'a') as f:
                f.write(out)
        return ret
    return wrapper

def read_debug_log(fname):
    with open(fname, 'r') as f:
        batch = f.read().split('\n\n')
    # Assuming the rest of the batches are printing the same vars.
    if len(batch[0].split('\n')) != len(batch[1].split('\n')):
           batch = batch[1:]
    if len(batch[-1].split('\n')) != len(batch[-2].split('\n')):
           batch = batch[:-1]

    p_G = re.compile(r'dG[012][012].[a-zA-Z]+')
    p_k = re.compile(r'[a-zA-Z]+[012]?p?.[a-zA-Z]+')
    p_v = re.compile(r'([-]?[0-9]\.[0-9]+e[+\-][0-9]+)|([-]?[0-9]+\.[0-9]+)')

    d = {'k': [], 'v': [], 'i': []}
    def add(k, v):
        d['k'].append(k)
        d['v'].append(v)
        # try:
        #     d[k].append(v)
        # except KeyError:
        #     d[k] = [v]

    for i in range(len(batch)):
        l = batch[i].split('\n')
        for li in l:
            if 'inf' in li or 'nan' in li:
                ks = re.findall(r'[a-zA-Z]+ [a-zA-Z]+(?=:)', li)
                vs = [0, 0]
            else:
                ks = p_G.findall(li) if 'G' in li else p_k.findall(li)
                vs_str = p_v.findall(li)
                vs = [float(v) for vs_ in vs_str for v in vs_ if len(v) > 0]

            d['i'].extend([i, i])
            if len(ks) == 1:
                print(li, ks, vs)
                add(ks[0] + '1', vs[0])
                add(ks[0] + '2', vs[1])
            else:
                add(ks[0], vs[0])
                add(ks[1], vs[1])
    df = pd.DataFrame(d)
    print(pd.unique(df.k))
    return df