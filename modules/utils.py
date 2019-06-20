"""Utility functions for common processing tasks"""

import sys
import math
import resource

def print_progress(n_done, n_all, sl=None):
    """Prints progress in percentage only when it changes"""
    if n_done % math.ceil(float(n_all) / 100) == 0:
        progress = int(float(n_done) / n_all * 100)
        if sl is None:
            progress_msg = '  progress: {0:d}%'.format(progress)
        else:
            progress_msg = '  progress in SL {0:d}: {1:d}%          '.format(sl, progress)
        sys.stdout.write('  {0}\r'.format(progress_msg))
        sys.stdout.flush()
        return True
    return False

def chunks(l, n):
    """Splits a list into evenly sized chunks"""
    return [l[i:i+n] for i in range(0, len(l), n)]

def mem():
    """Get memory used by the process"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
