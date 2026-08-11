@echo off
rem Detached launcher for the syntax size sweep. Start-Process this .cmd so the
rem batch runs as an independent Windows process that survives harness task
rem kills. bash -c avoids MSYS arg path-mangling (the script self-logs).
"C:\Program Files\Git\bin\bash.exe" -c "bash /d/EXPERIMENTS/reopen/scripts/syntax_size_sweep_batch.sh"
