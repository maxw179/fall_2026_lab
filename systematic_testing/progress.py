"""Notebook readouts with a heartbeat and deterministic cleanup.

Widgets avoid frontends that ignore display_id updates. Without widget support,
phase changes and periodic snapshots are written below the terminal tqdm bar.
"""
from html import escape
from threading import Event, RLock, Thread
from time import perf_counter


class LiveReadouts:
    def __init__(self, progress, interval=1.0):
        """Set interval=None for event-only readouts without a background timer."""
        self.progress = progress
        self.interval = interval
        self.started = self.stage_started = perf_counter()
        self.state = dict(run="Preparing", acquisition="Waiting", optimizer="Waiting",
                          result="No completed updates yet")
        self._lock = RLock()
        self._stop = Event()
        self._closed = False
        self._last_snapshot = self.started
        self.widget = None
        self._thread = None

    def __enter__(self):
        from IPython import get_ipython
        shell = get_ipython()
        if shell is not None and getattr(shell, 'kernel', None) is not None:
            try:
                from ipywidgets import HTML
                from IPython.display import display
                self.widget = HTML()
                display(self.widget)
            except ImportError:
                pass
        self._publish(force=True)
        if self.interval is not None:
            self._thread = Thread(target=self._heartbeat, name="correction-readouts", daemon=True)
            self._thread.start()
        return self

    def update(self, **changes):
        with self._lock:
            if self._closed:
                return
            # Stage durations track actual phases, not completed iterations.
            if any(key in changes and changes[key] != self.state.get(key)
                   for key in ('run', 'optimizer')):
                self.stage_started = perf_counter()
            run_changed = changes.get("run", self.state["run"]) != self.state["run"]
            self.state.update(changes)
            self._publish(phase_changed=True, force=run_changed)

    def _publish(self, phase_changed=False, force=False):
        with self._lock:
            now = perf_counter()
            rows = [(label, self.state[key]) for label, key in
                    [('Run', 'run'), ('Acquisition', 'acquisition'),
                     ('Optimizer', 'optimizer'), ('Latest correction', 'result')]]
            rows += [('Elapsed', f'{now - self.started:.1f} s'),
                     ('Current stage', f'{now - self.stage_started:.1f} s')]
            if self.widget is not None:
                self.widget.value = "<div style='line-height:1.6'>" + '<br>'.join(
                    f'<b>{label}:</b> {escape(str(value))}' for label, value in rows) + '</div>'
            elif force or (self.interval is not None and (
                    (phase_changed and now - self._last_snapshot >= 5)
                    or now - self._last_snapshot >= 15)):
                self.progress.write(' | '.join(f'{label}: {value}' for label, value in rows))
                self._last_snapshot = now

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._closed:
                    return
                self.progress.refresh()  # Refresh tqdm elapsed/ETA during long calculations.
                self._publish()

    def __exit__(self, exc_type, exc_value, traceback):
        # Signal and join before publishing final state; reruns cannot leave
        # an old timer rewriting the new run's readouts.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            if exc_type is not None:
                self.state['run'] = 'Interrupted' if issubclass(exc_type, KeyboardInterrupt) else f'Failed: {exc_value}'
            self._publish(force=True)
            self._closed = True
        return False
