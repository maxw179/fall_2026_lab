"""Heartbeat, notebook widget updates, and interruption cleanup."""
import unittest
from types import SimpleNamespace
from threading import Event
from unittest.mock import patch
from systematic_testing.progress import LiveReadouts


class Progress:
    def __init__(self):
        self.refreshed = Event()
        self.messages = []
    def refresh(self):
        self.refreshed.set()
    def write(self, message):
        self.messages.append(message)


class LiveReadoutTests(unittest.TestCase):
    def test_widget_heartbeat_and_cleanup(self):
        class Widget:
            value = ''
        bar = Progress()
        with patch('IPython.get_ipython', return_value=SimpleNamespace(kernel=True)), \
             patch.dict('sys.modules', {'ipywidgets': SimpleNamespace(HTML=Widget)}), \
             patch('IPython.display.display'):
            with LiveReadouts(bar, interval=.01) as panel:
                panel.update(run='Trial 1 | Stack', optimizer='Computing Hessian')
                self.assertTrue(bar.refreshed.wait(timeout=1))
                self.assertIn('Computing Hessian', panel.widget.value)
                self.assertIn('Current stage', panel.widget.value)
                panel.update(run='Completed')
            self.assertFalse(panel._thread.is_alive())
            final = panel.widget.value
            panel.update(run='Late event')
            self.assertEqual(panel.widget.value, final)

    def test_event_only_mode_has_no_timer(self):
        bar = Progress()
        with patch('IPython.get_ipython', return_value=None):
            with LiveReadouts(bar, interval=None) as panel:
                panel.update(optimizer='Iteration 0 complete')
                self.assertIsNone(panel._thread)
                self.assertFalse(bar.refreshed.is_set())
        self.assertTrue(panel._closed)

    def test_interrupt_stops_heartbeat_and_updates_status(self):
        bar = Progress()
        with patch('IPython.get_ipython', return_value=None):
            with self.assertRaises(KeyboardInterrupt):
                with LiveReadouts(bar, interval=.01) as panel:
                    panel.update(run='Working')
                    raise KeyboardInterrupt()
        self.assertFalse(panel._thread.is_alive())
        self.assertIn('Run: Interrupted', bar.messages[-1])


if __name__ == '__main__':
    unittest.main()
