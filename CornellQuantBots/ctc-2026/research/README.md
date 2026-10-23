Research code behind REPORT.md (run on the research machine; absolute paths inside point to it).
Pipeline: fit_params.py (train-only fits) -> verify_settle.py -> precompute.py (point-in-time model outputs for
seasons 1-200) -> sim.py/policies.py/runner.py (event-driven backtester with execution scenarios) ->
compare_v2.py / v1var.py / concen.py (validation) -> final_test.py (seasons 1-100, run once) ->
final_test_results.csv. etf_exec.py and phase_spreads.py analyse read-only testnet recordings (tools/recorder.py).
