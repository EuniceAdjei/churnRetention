import sys
import traceback
from datetime import datetime

LOG_PATH = 'verify_run_log.txt'

def write_log(lines):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
        f.flush()

def main():
    start = datetime.utcnow().isoformat() + 'Z'
    write_log([f'=== verify_imports run at {start} ===', f'python executable: {sys.executable}', ''])

    results = {}

    # numpy
    try:
        import numpy as np
        results['numpy'] = (True, getattr(np, '__version__', 'unknown'))
    except Exception:
        results['numpy'] = (False, traceback.format_exc())

    # pandas
    try:
        import pandas as pd
        results['pandas'] = (True, getattr(pd, '__version__', 'unknown'))
    except Exception:
        results['pandas'] = (False, traceback.format_exc())

    # os
    try:
        import os
        results['os'] = (True, 'stdlib')
    except Exception:
        results['os'] = (False, traceback.format_exc())

    # statsmodels and ols
    try:
        import statsmodels
        import statsmodels.formula.api as stats
        from statsmodels.formula.api import ols
        results['statsmodels'] = (True, getattr(statsmodels, '__version__', 'unknown'))
    except Exception:
        results['statsmodels'] = (False, traceback.format_exc())

    # scikit-learn
    try:
        import sklearn
        from sklearn import linear_model, datasets
        from sklearn.metrics import mean_squared_error
        results['scikit-learn'] = (True, getattr(sklearn, '__version__', 'unknown'))
    except Exception:
        results['scikit-learn'] = (False, traceback.format_exc())

    # plotly
    try:
        import plotly
        try:
            import plotly.plotly as py
        except Exception:
            py = None
        try:
            import plotly.graph_objs as go
        except Exception:
            go = None
        try:
            from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
            try:
                init_result = None
                try:
                    init_notebook_mode(connected=True)
                    init_result = 'ok'
                except Exception as ie:
                    init_result = f'init failed: {ie}'
            except Exception as e_inner:
                init_result = f'offline import failed: {e_inner}'
        except Exception:
            init_result = traceback.format_exc()
        results['plotly'] = (True, getattr(plotly, '__version__', 'unknown'), py is not None, go is not None, init_result)
    except Exception:
        results['plotly'] = (False, traceback.format_exc())

    # Write results to log
        start = datetime.utcnow().isoformat() + 'Z'
        write_log([f'=== verify_imports run at {start} ===', f'python executable: {sys.executable}', ''])

        # We'll log each import attempt immediately so partial runs are recorded
        def try_import(name, import_fn):
            try:
                val = import_fn()
                write_log([f"OK: {name} -> {val}"])
                return True
            except Exception:
                tb = traceback.format_exc()
                write_log([f"ERR: {name} ->", tb])
                return False

        try_import('numpy', lambda: __import__('numpy').__version__)
        try_import('pandas', lambda: __import__('pandas').__version__)
        try_import('os', lambda: 'stdlib')

        def import_statsmodels():
            m = __import__('statsmodels')
            __import__('statsmodels.formula.api')
            return getattr(m, '__version__', 'unknown')

        try_import('statsmodels', import_statsmodels)

        def import_sklearn():
            m = __import__('sklearn')
            __import__('sklearn.linear_model')
            __import__('sklearn.datasets')
            __import__('sklearn.metrics')
            return getattr(m, '__version__', 'unknown')

        try_import('scikit-learn', import_sklearn)

        def import_plotly():
            m = __import__('plotly')
            # try submodules
            try:
                __import__('plotly.plotly')
                has_plotly_plotly = True
            except Exception:
                has_plotly_plotly = False
            try:
                __import__('plotly.graph_objs')
                has_graph_objs = True
            except Exception:
                has_graph_objs = False
            # try offline init
            try:
                from plotly.offline import init_notebook_mode
                try:
                    init_notebook_mode(connected=True)
                    init_status = 'ok'
                except Exception as ie:
                    init_status = f'init failed: {ie}'
            except Exception as e:
                init_status = f'offline import failed: {e}'
            return f"{getattr(m, '__version__', 'unknown')} | plotly.plotly={has_plotly_plotly} | graph_objs={has_graph_objs} | init={init_status}"

        try_import('plotly', import_plotly)

        write_log(['', f'=== end run {datetime.utcnow().isoformat()} ===', ''])

C:/Users/barry/AppData/Local/Programs/Python/Python312/python.exe - <<'PY'
import sys, os, traceback
output = []
output.append('python: ' + sys.version.split('\\n')[0])
try:
    import numpy as np
    output.append('numpy: ' + getattr(np, '__version__', 'unknown'))
except Exception:
    output.append('numpy import error: ' + traceback.format_exc())
try:
    import pandas as pd
    output.append('pandas: ' + getattr(pd, '__version__', 'unknown'))
except Exception:
    output.append('pandas import error: ' + traceback.format_exc())
try:
    import statsmodels
    output.append('statsmodels: ' + getattr(statsmodels, '__version__', 'unknown'))
except Exception:
    output.append('statsmodels import error: ' + traceback.format_exc())
try:
    import sklearn
    output.append('scikit-learn: ' + getattr(sklearn, '__version__', 'unknown'))
except Exception:
    output.append('scikit-learn import error: ' + traceback.format_exc())
try:
    import plotly
    from plotly.offline import init_notebook_mode
    try:
        init_notebook_mode(connected=True)
        output.append('plotly init_notebook_mode: ok')
    except Exception as ie:
        output.append('plotly init_notebook_mode error: ' + str(ie))
    output.append('plotly: ' + getattr(plotly, '__version__', 'unknown'))
except Exception:
    output.append('plotly import error: ' + traceback.format_exc())
csv_path = os.path.join(os.getcwd(), 'GhanaTelecomData.csv')
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        output.append(f'CSV found: {csv_path} rows={len(df)} cols={len(df.columns)}')
        output.append('\\nHEAD:\\n' + df.head().to_string())
    except Exception:
        output.append('CSV read error: ' + traceback.format_exc())
else:
    output.append('CSV not found at ' + csv_path)
print('\\n'.join(output))



