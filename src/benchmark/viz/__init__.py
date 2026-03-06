"""Benchmark reporting and plotting utilities.

Force a non-GUI matplotlib backend for batch runs to avoid Tkinter
errors when plotting from non-main threads or headless environments.
"""
try:
	import matplotlib

	matplotlib.use("Agg")
except Exception:
	# Best-effort: if matplotlib is not available or backend cannot be
	# set at import-time, let callers handle ImportError as before.
	pass
