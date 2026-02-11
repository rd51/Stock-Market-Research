import py_compile
import glob

errs = False
for f in glob.glob('**/*.py', recursive=True):
    try:
        py_compile.compile(f, doraise=True)
    except Exception as e:
        print(f'ERROR in {f}: {e}')
        errs = True

if not errs:
    print('No syntax errors found in .py files')
