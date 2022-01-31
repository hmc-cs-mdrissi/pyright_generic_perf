# How to Reproduce
Run pip install -r requirements.txt. Then run pyright on pyright_perf.py. The performance should vary heavily across 1.1.205-1.1.209. The typings folder holds custom type stubs for tensorflow that pyrightconfig.json should enable.
