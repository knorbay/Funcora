import os
import sys
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import numpy as np
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Funcora import CachedAnalysis, FuncoraWindow, GraphEngine, sanitize_math_input


class MathEngineTests(unittest.TestCase):
    def test_quadratic_analysis_finds_roots_and_derivative(self):
        x = sp.Symbol("x")
        analysis = CachedAnalysis("x^2-4x+4", x)

        self.assertEqual(analysis.sym_roots, [sp.Integer(2)])
        self.assertEqual(sp.simplify(analysis.derivative_1 - (2 * x - 4)), 0)

    def test_unsafe_expression_is_rejected(self):
        with self.assertRaises(ValueError):
            sanitize_math_input("__import__('os').system('echo nope')")

    def test_discontinuity_break_does_not_mutate_input(self):
        values = GraphEngine.break_discontinuities(
            x_vals=np.array([-0.1, 0.1]),
            y_vals=np.array([-100.0, 100.0]),
            y_min=-10,
            y_max=10,
        )
        self.assertTrue(all(np.isnan(values)))


class WindowInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = FuncoraWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()

    def test_invalid_input_is_debounced_and_reported(self):
        card = self.window.cards[0]
        card.input_field.setText("x+")

        self.assertFalse(card.message_label.isVisible())
        QTest.qWait(280)
        self.assertTrue(card.message_label.isVisible())
        self.assertIn("Could not parse", card.message_label.text())

    def test_grid_toggle_updates_state_and_label(self):
        self.window.toggle_grid()

        self.assertFalse(self.window.grid_visible)
        self.assertEqual(self.window.grid_btn.text(), "Grid: Off")
        self.assertFalse(any(line.get_visible() for line in self.window.canvas.ax.get_xgridlines()))

    def test_function_can_be_duplicated(self):
        source = self.window.cards[0]
        source.fill_area = True

        self.window.duplicate_function_card(source)

        self.assertEqual(len(self.window.cards), 2)
        self.assertEqual(self.window.cards[1].get_expression(), source.get_expression())
        self.assertEqual(self.window.cards[1].color, source.color)
        self.assertTrue(self.window.cards[1].fill_area)

    def test_project_round_trip_restores_workspace(self):
        self.window.set_plot_mode("polar")
        self.window.cards[0].input_field.setText("4cos(3t)")
        self.window.toggle_grid()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "rose.funcora")
            self.assertTrue(self.window.save_project(path))

            restored = FuncoraWindow()
            try:
                self.assertTrue(restored.open_project(path))
                self.assertEqual(restored.plot_mode, "polar")
                self.assertEqual(restored.cards[0].get_expression(), "4cos(3t)")
                self.assertFalse(restored.grid_visible)
            finally:
                restored.close()

    def test_example_switches_mode_and_reuses_initial_card(self):
        self.window.load_example("implicit", "x^2+y^2=25")

        self.assertEqual(self.window.plot_mode, "implicit")
        self.assertEqual(len(self.window.cards), 1)
        self.assertEqual(self.window.cards[0].get_expression(), "x^2+y^2=25")
        self.assertEqual(self.window.cards[0].label.text(), "f(x,y) =")


if __name__ == "__main__":
    unittest.main()
