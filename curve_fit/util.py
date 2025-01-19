import sys
import numpy as np
import pandas as pd

MSG_PREFIX=""

def error_msg(s_msg):
    """Print an error message and quit"""
    #raise Exception(MSG_PREFIX+s_msg)
    raise Exception(MSG_PREFIX+"ERROR> "+str(s_msg))
    #sys.stdout.flush()

def info_msg(s_msg):
    """Print an information message"""
    print(MSG_PREFIX+"INFO> "+s_msg)
    sys.stdout.flush()

def read_string(s_file, encoding="utf-8"):
    """Read s_file into a long string"""
    s_file = str(s_file)
    s=""
    if s_file.endswith('.gz'):
        import gzip
        with gzip.open(s_file, 'rb') as f:
            s=str(f.read())
    else:
        with open(s_file, 'r', encoding="utf-8") as f:
            s=f.read()
    return s

def read_list(s_file, encoding="utf-8"):
    """Read s_file into a list of strings, remove line breaks"""
    if s_file.endswith('.gz'):
        import gzip
        with gzip.open(s_file, mode='rb') as f:
            s=f.read()
    else:
        with open(s_file, mode='rb') as f:
            s=f.read()
    s=s.decode(encoding, 'ignore')
    S=s.splitlines()
    return S

def save_list(s_file, S_lines, s_end=""):
    """Save S_lines to s_file, each line ends with character s_end.
    s_file: str, file name
    S_lines: list[str], lines of text
    s_end: str, append to line, default ''. You may want to set it to '\n' to provide breaks."""

    # sometimes strings arrive in unicode format, so convert to str first
    if type(S_lines) is not list:
        S_lines=[S_lines]
    f=open(s_file, 'w', encoding="utf-8")
    for s in S_lines:
        f.write(s+s_end)
    f.flush()
    f.close()

def sarray2rarray(S):
    """Convert a str array to a np.array"""
    R=np.empty(len(S))
    for i,s in enumerate(S):
        try:
            r=float(s)
        except:
            r=np.nan
        R[i]=r
    return R

def rarray2sarray(R, s_format='%.2g', s_null=''):
    """Convert float array to str array, with format s_format, replace NULL by s_null"""
    return [s_null if np.isnan(r) else s_format % r for r in R]

def split(df, n_chunk=1, chunk_size=0, key=None):
    """split a table/list/2d-array etc into n chunks of roughly equal size, but not gauranteed. This can be used to split an input data into multiple pieces, send to workers, then you need to assemble the output back into one piece by yourself.
    n_chunk: int, number of pieces to break the data into, default 1
    chunk_size: int, default 0, maximum size per piece. 0 means no limit.
    It attempts to break the input into n_chunk, but if too large (defined by chunk_size), it will create more pieces, so each is not bigger than chunk_size,  You may specify one of n_chunk and chunk_size, or both
    If key is provided, it is an array/list, where we need to make sure rows with the same key must stay together.
        e.g., if key is GROUP_ID, then rows with the same GROUP_ID will not be split into two chunks
        However, you need to make sure rows with the same GROUP_ID must be next to each other in df
        Note: we assume on average each key has about the same number of records, if not, results may not be optimal
    """
    n=len(df)
    if key is not None:
        key=np.array(key)
        IDX=np.concatenate([[-1], np.where(key[:-1]!=key[1:])[0], [n-1]])
        IDX=[(a+1,b+1) for a,b in zip(IDX[:-1], IDX[1:])]
        IDX=split(IDX, n_chunk=n_chunk, chunk_size=chunk_size)
        out=[df[X[0][0]:X[-1][1]].copy() if isinstance(df, pd.DataFrame) else df[X[0][0]:X[-1][1]] for X in IDX]
        return out
    sz=int(math.ceil(n*1.0/n_chunk)) if n_chunk>0 else n
    if chunk_size==0:
        chunk_size=sz
    else:
        chunk_size=min(chunk_size, sz)
    chunk_size=max(1, chunk_size)
    out=[]
    for iB in range(0, n, chunk_size):
        iE=min(iB+chunk_size, n)
        out.append(df[iB:iE].copy() if isinstance(df, pd.DataFrame) else df[iB:iE])
    return out

import time
class StopWatch(object):
    """StopWatch is used to measure time lapse between check() calls
    sw=util.StopWatch()
    calc_A()
    sw.check('A calculated')
    """

    def __init__(self, s_prompt=""):
        """Start the timer"""
        self.start=time.time()
        self.prompt="%s> " % s_prompt
        print(self.prompt+"Start timer ...")

    def check(self, s_msg, l_reset=True):
        """Print the time lapse since start, in addtion to message
        s_msg: str, message to print
        l_reset: boolean, default True. reset timer, so next check() will report the time past between now and next time.
        otherwise, timer is not reset."""
        x=time.time()
        dur=x-self.start
        print(MSG_PREFIX+self.prompt+"Passed: %.1f secs, %s" % (dur, s_msg))
        if l_reset: self.start=x
        return dur

class Progress(object):
    """Progress can monitor the progress of a lengthy computation."""

    def __init__(self, n, func=None, position=0):
        """n: int, total number of items to be process
        func: method. A function that takes an item count [0-n] as input and return percentage [0-1.0]
            e.g., lambda x: (x*1.0/n)*(x/n) is good for O(n2) algorithms
            default, None, linear algorithm
            position: progress bar position, passed to tqdm
        """
        import tqdm
        # see lock issue https://github.com/tqdm/tqdm/issues/461
        tqdm.tqdm.get_lock().locks = []

        self.start=time.time()
        self.n=n # total amount of work
        self.func = func
        # a function that convert i,n into percent progress
        if self.func is None:
            self.func=lambda i: i*1.0/n
        if type(self.func) is str:
            if self.func=='O(n)':
                self.func=lambda i: i*1.0/n
            elif self.func=='O(n2)':
                self.func=lambda i: (i*1.0/n)**2
        self.pg=tqdm.tqdm(total=self.n, position=position)
        #print(("Start processing %d records ..." % self.n))

    def check(self, i, s_msg=''):
        """i: int, index of the current item being processed
        s_msg: str, message to print
        return: print progress statistics, estimate the finish time."""
        i_pass=(time.time()-self.start)/60
        pct=max(self.func(i),1e-6)
        i_remain=abs((1-pct)*(i_pass/pct))
        # percentage, time used, additional time required
        #print(("Processed %.1f%%, used %.1fmin, remain %.1fmin. %s" % (pct*100, i_pass, i_remain, s_msg)))
        if i>self.pg.n: # set progress to 0 may lock the underlyig library
            self.pg.update(max(i-self.pg.n, 0))

def display(df, headers='keys', tablefmt='simple', l_print=True):
    """tablefmt: simple, psql, fancy_grid"""
    from tabulate import tabulate
    if type(df) is pd.Series:
        X=[(k,v) for k,v in df.items()]
        df=pd.DataFrame(X, columns=['Column','Value'])
    s=tabulate(df, headers=headers, tablefmt=tablefmt)
    if l_print:
        print(s)
    else:
        return s

def header(df):
    """Obtain column names from a dataframe"""
    return list(df.columns)

def rename2(df, columns):
    """rename column headers, similar to the build-in DataFrame.rename, but always inplace
    and use no additional memory. Default rename seems to create a new copy of table."""
    df.columns=[ columns.get(x, x) for x in header(df) ]

pd.DataFrame.display = display
pd.DataFrame.header = header
pd.DataFrame.rename2 = rename2

def unique2(lst, is_rm_na=False):
    """Unique a string list, but preserve the order. This is useful for taking the top X number of unique entries"""
    out=[]
    c_seen={}
    l_na_seen=False
    for x in lst:
        if x in c_seen: continue
        if pd.isnull(x):
            if is_rm_na or l_na_seen: continue
            l_na_seen=True
        c_seen[x]=True
        out.append(x)
    return out

