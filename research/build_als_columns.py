"""Build ALS columns for all windows behind a target. Usage: build_als_columns.py TARGET|eval"""
import sys; sys.path.insert(0,"research")
from common import *
from alsfeat import als_cols
c=load_cache(); a=sys.argv[1]
wins=[(CUTOFF-k*DAY,CUTOFF) for k in (3,2,1)]+[(CUTOFF,None)] if a=="eval" else [(int(a)-k*DAY,int(a)) for k in (3,2,1)]+[(int(a),CUTOFF)]
for w in wins:
    for f in (12,16,24): als_cols(c,*w,f)
    print("built",w,flush=True)
