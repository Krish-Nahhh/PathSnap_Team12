import sys, importlib
try:
    import sklearn
    print('sklearn version', sklearn.__version__)
except Exception as e:
    print('sklearn import error', e)
