# How to Reproduce
Run pip install -r requirements.txt. Then run pyright on pyright_perf.py. The performance should vary heavily across 1.1.205-1.1.207. The typings folder holds custom type stubs for tensorflow.

Python 3.8 was used although I'd expect other versions to show similar behavior as pyrightconfig specifies 3.8.
