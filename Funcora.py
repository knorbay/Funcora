import os
import sys
import json
import numpy as np
import sympy as sp
from fractions import Fraction
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import (
    QColor, QFont, QIcon, QPixmap,
    QTextCursor, QTextCharFormat, QAction
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFrame, QScrollArea, QDoubleSpinBox,
    QSplitter, QToolButton, QColorDialog, QFileDialog, QTextEdit, QSplashScreen,
    QComboBox, QMenu, QMessageBox
)
from sympy.calculus.util import continuous_domain
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter, ScalarFormatter

DARK_STYLESHEET = """
QMainWindow { background-color: #0F1017; }
QWidget { color: #E2E8F0; font-family: 'Segoe UI', 'Inter', sans-serif; font-size: 13px; }
QFrame#sidebar, QFrame#analysisBar { background-color: #161722; border-color: #232536; }
QFrame#functionCard { background-color: #1E202E; border: 1px solid #2B2D42; border-radius: 8px; }
QFrame#functionCard:hover { border-color: #3B3E5B; }
QPushButton#primaryBtn { background-color: #2563EB; color: #FFFFFF; border: none; border-radius: 6px; padding: 6px 14px; font-weight: 600; }
QPushButton#primaryBtn:hover { background-color: #1D4ED8; }
QPushButton#dangerBtn { background-color: transparent; color: #EF4444; border: 1px solid #EF4444; border-radius: 6px; padding: 4px 10px; }
QPushButton#dangerBtn:hover { background-color: rgba(239, 68, 68, 0.15); }
QPushButton#colorPickerBtn { border-radius: 8px; border: 1px solid #475569; }
QPushButton#toggleBtn { background-color: #2563EB; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 12px; }
QPushButton#toggleBtn[visible_state="false"] { background-color: #2B2D42; color: #64748B; }
QPushButton#fillBtn { background-color: #10B981; color: white; border: none; border-radius: 4px; font-weight: bold; font-size: 11px; }
QPushButton#fillBtn[fill_state="false"] { background-color: #2B2D42; color: #64748B; }
QPushButton#deleteBtn { background-color: rgba(239, 68, 68, 0.1); color: #EF4444; border: none; border-radius: 4px; font-weight: 600; font-size: 11px; }
QPushButton#deleteBtn:hover { background-color: #EF4444; color: white; }
QPushButton#duplicateBtn { background-color: #252738; color: #A5B4FC; border: none; border-radius: 4px; font-weight: 700; }
QPushButton#duplicateBtn:hover { background-color: #31344A; color: white; }
QLineEdit#funcInput { background-color: #12131C; border: 1px solid #2B2D42; border-radius: 6px; color: #F8FAFC; padding: 5px 8px; font-family: 'Fira Code', 'Consolas', monospace; }
QLineEdit#funcInput:focus { border-color: #2563EB; }
QDoubleSpinBox { background-color: #181924; border: 1px solid #2B2D42; border-radius: 4px; color: #F8FAFC; padding: 3px 6px; }
QToolButton { background-color: #1E202E; border: 1px solid #2B2D42; border-radius: 6px; padding: 5px 10px; }
QToolButton:hover { background-color: #2B2D42; }
QComboBox { background-color: #1E202E; border: 1px solid #2B2D42; border-radius: 6px; padding: 5px 10px; min-width: 120px; }
QComboBox:hover, QComboBox:focus { border-color: #3B3E5B; }
QComboBox QAbstractItemView { background-color: #1E202E; border: 1px solid #2B2D42; selection-background-color: #2563EB; }
QMenu { background-color: #1E202E; border: 1px solid #2B2D42; padding: 5px; }
QMenu::item { padding: 7px 24px 7px 10px; border-radius: 4px; }
QMenu::item:selected { background-color: #2B2D42; }
QLabel#emptyState { color: #64748B; padding: 36px 18px; font-size: 12px; }
QTextEdit#analysisText { background-color: #12131C; border: 1px solid #232536; border-radius: 6px; color: #CBD5E1; font-family: 'Fira Code', 'Consolas', monospace; font-size: 12px; }
"""

def get_func_name(index: int) -> str:
    names = ['f', 'g', 'h', 'p', 'q', 'r', 'u', 'v', 'w']
    if 1 <= index <= len(names):
        return names[index - 1]
    return f"F{index}"

MAX_ROOT_SEARCH = 300
MAX_ROOT_LABELS = 40

import re
_SAFE_EXPR_RE = re.compile(r'^[0-9A-Za-z\s\.\,\+\-\*\/\^\(\)\=]*$')

def sanitize_math_input(raw_str: str) -> str:
    """Reject anything outside a plain-math character whitelist BEFORE it
    ever reaches sympy's parse_expr/eval machinery. This blocks code
    injection payloads (e.g. quotes, underscores, brackets needed for
    things like __import__(...) or list/attribute tricks) without
    restricting any legitimate mathematical expression the app supports."""
    if not _SAFE_EXPR_RE.match(raw_str):
        raise ValueError("Invalid characters in expression")
    return raw_str

def format_pi(x, pos):
    if abs(x) < 1e-9: return "0"
    frac = Fraction(x / np.pi).limit_denominator(4)
    num, den = frac.numerator, frac.denominator
    if den == 1:
        if num == 1: return "π"
        if num == -1: return "-π"
        return f"{num}π"
    if den == 2:
        if num == 1: return "π/2"
        if num == -1: return "-π/2"
        return f"{num}π/2"
    return f"{x:.2g}"

def normalize_expr(expr):
    if expr is None: return None
    for method in [sp.trigsimp, sp.cancel, sp.factor, sp.powsimp, sp.simplify]:
        try:
            expr = method(expr)
        except Exception:
            pass
    return expr

def pretty(expr):
    if expr is None: return "Undefined"
    expr = normalize_expr(expr)
    text = sp.sstr(expr)
    replacements = {
        "pi": "π", "oo": "∞",
        "**2": "²", "**3": "³", "**4": "⁴", "**5": "⁵",
        "sqrt(": "√(", "log(": "ln(", "exp(": "e^(",
        "**": "^", "*": ""
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def domain_pretty(domain):
    if isinstance(domain, sp.Complement):
        base, removed = domain.args
        if base == sp.S.Reals and isinstance(removed, sp.Union) and all(isinstance(a, sp.ImageSet) for a in removed.args):
            return "ℝ \\ {π/2 + kπ}"
    if domain is None: return "Undefined"
    if domain == sp.S.Reals: return "(-∞, ∞)"
    if isinstance(domain, sp.Interval):
        left = "(" if domain.left_open else "["
        right = ")" if domain.right_open else "]"
        return f"{left}{pretty(domain.start)}, {pretty(domain.end)}{right}"
    if isinstance(domain, sp.Union):
        return " U ".join(domain_pretty(d) for d in domain.args)
    if isinstance(domain, sp.Complement):
        base, removed = domain.args
        if base == sp.S.Reals:
            return f"ℝ \\ {pretty(removed)}"
    return pretty(domain)

def fmt_num(n):
    if abs(n - round(n)) < 1e-9: return str(int(round(n)))
    return f"{n:.2f}"

def build_domain_intervals(domain):
    intervals, excluded = [], []
    def _bound(val, fallback):
        try: return float(val)
        except Exception: return fallback
    def _walk(d):
        if d is None or d == sp.EmptySet: return
        if d == sp.S.Reals:
            intervals.append((-np.inf, np.inf, True, True))
        elif isinstance(d, sp.Interval):
            intervals.append((_bound(d.start, -np.inf), _bound(d.end, np.inf), bool(d.left_open), bool(d.right_open)))
        elif isinstance(d, sp.Union):
            for sub in d.args: _walk(sub)
        elif isinstance(d, sp.Complement):
            base, removed = d.args
            _walk(base)
            if isinstance(removed, sp.FiniteSet):
                for p in removed.args:
                    try: excluded.append(float(p.evalf()))
                    except Exception: continue
        elif isinstance(d, sp.FiniteSet):
            for p in d.args:
                try:
                    v = float(p.evalf())
                    intervals.append((v, v, False, False))
                except Exception: continue
        else:
            intervals.append((-np.inf, np.inf, True, True))
    _walk(domain)
    return intervals, excluded

class CachedAnalysis:
    def __init__(self, raw_str: str, x_symbol: sp.Symbol): 
        sanitize_math_input(raw_str)
        self.raw_str = raw_str
        clean_str = raw_str.replace("^", "**").replace("cosec", "csc").replace("cosech", "csch")
        transformations = standard_transformations + (implicit_multiplication_application,)
        self.expr = parse_expr(clean_str, transformations=transformations, local_dict=GraphEngine.ALIASES)
        if self.expr.free_symbols - {x_symbol}: raise ValueError("Unknown symbols")
        self.f_lambdified = sp.lambdify(x_symbol, self.expr, modules=["numpy", GraphEngine.ALIASES])
        try: self.domain = continuous_domain(self.expr, x_symbol, sp.S.Reals)
        except Exception: self.domain = sp.S.Reals
        self.domain_intervals, self.domain_excluded = build_domain_intervals(self.domain)
        try: self.derivative_1 = normalize_expr(sp.diff(self.expr, x_symbol))
        except Exception: self.derivative_1 = None
        try: self.derivative_2 = normalize_expr(sp.diff(self.derivative_1, x_symbol)) if self.derivative_1 is not None else None
        except Exception: self.derivative_2 = None
        try: self.integral = normalize_expr(sp.integrate(self.expr, x_symbol))
        except Exception: self.integral = None
        self.v_asymptotes_exact = []
        try:
            num, den = sp.fraction(sp.cancel(self.expr))
            if den != 1:
                zeros = sp.solveset(den, x_symbol, domain=sp.S.Reals)
                if isinstance(zeros, sp.FiniteSet):
                    self.v_asymptotes_exact.extend(float(z.evalf()) for z in zeros)

            def get_exact_bounds(d):
                bounds = []
                if d == sp.S.Reals or d == sp.EmptySet or d is None:
                    return bounds
                if isinstance(d, sp.Interval):
                    bounds.append((d.start, d.end, d.left_open, d.right_open))
                elif isinstance(d, sp.Union):
                    for sub in d.args:
                        bounds.extend(get_exact_bounds(sub))
                elif isinstance(d, sp.Complement):
                    bounds.extend(get_exact_bounds(d.args[0]))
                return bounds

            for ex_low, ex_high, l_open, r_open in get_exact_bounds(self.domain):
                if ex_low != -sp.oo and l_open:
                    try:
                        if sp.limit(self.expr, x_symbol, ex_low, dir="+") in (sp.oo, -sp.oo):
                            self.v_asymptotes_exact.append(float(ex_low.evalf()))
                    except Exception: pass
                if ex_high != sp.oo and r_open:
                    try:
                        if sp.limit(self.expr, x_symbol, ex_high, dir="-") in (sp.oo, -sp.oo):
                            self.v_asymptotes_exact.append(float(ex_high.evalf()))
                    except Exception: pass

            self.v_asymptotes_exact = sorted(set(round(v, 12) for v in self.v_asymptotes_exact))
        except Exception: pass
        self.h_asymptotes_exact = []
        try:
            lim_p = sp.limit(self.expr, x_symbol, sp.oo)
            if lim_p.is_real and lim_p.is_number: self.h_asymptotes_exact.append(float(lim_p))
            lim_m = sp.limit(self.expr, x_symbol, -sp.oo)
            if lim_m.is_real and lim_m.is_number and float(lim_m) not in self.h_asymptotes_exact: self.h_asymptotes_exact.append(float(lim_m))
        except Exception: pass
        self.sym_roots = []
        try:
            roots = sp.solveset(self.expr, x_symbol, domain=sp.S.Reals)
            if isinstance(roots, sp.FiniteSet): self.sym_roots = [sp.simplify(r) for r in roots]
        except Exception: pass
        self.critical_points = []
        try:
            if self.derivative_1 is not None and self.derivative_2 is not None:
                crit_roots = sp.solveset(self.derivative_1, x_symbol, domain=sp.S.Reals)
                if isinstance(crit_roots, sp.FiniteSet):
                    for r in crit_roots:
                        try:
                            x_num, y_num, sec = float(r.evalf()), float(self.expr.subs(x_symbol, r).evalf()), float(self.derivative_2.subs(x_symbol, r).evalf())
                            if sec < 0: self.critical_points.append((x_num, y_num, "max"))
                            elif sec > 0: self.critical_points.append((x_num, y_num, "min"))
                        except Exception: continue
        except Exception: pass
        self.root_info = []
        try:
            for r in self.sym_roots:
                mult, k = 1, 1
                while k <= 6:
                    try:
                        if sp.simplify(sp.diff(self.expr, x_symbol, k).subs(x_symbol, r)) == 0:
                            mult = k + 1
                            k += 1
                        else: break
                    except Exception: break
                try: self.root_info.append((float(r.evalf()), mult))
                except Exception: continue
        except Exception: pass
        self.holes = []
        try:
            check_points = set(self.domain_excluded)
            for low, high, left_open, right_open in self.domain_intervals:
                if np.isfinite(low) and left_open: check_points.add(low)
                if np.isfinite(high) and right_open: check_points.add(high)
            for p in check_points:
                if any(abs(p - a) < 1e-9 for a in self.v_asymptotes_exact): continue
                try:
                    lim_left = sp.limit(self.expr, x_symbol, sp.Float(p), dir="-")
                    lim_right = sp.limit(self.expr, x_symbol, sp.Float(p), dir="+")
                    if lim_left == lim_right and lim_left.is_real and lim_left.is_finite: self.holes.append((p, float(lim_left.evalf())))
                    elif lim_left in (sp.oo, -sp.oo) or lim_right in (sp.oo, -sp.oo): self.v_asymptotes_exact.append(p)
                except Exception: continue
            self.v_asymptotes_exact = sorted(set(round(v, 12) for v in self.v_asymptotes_exact))
        except Exception: pass

class GraphEngine:
    ALIASES = {
        "sech": lambda v: 1 / sp.cosh(v), "cosech": lambda v: 1 / sp.sinh(v),
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
        "arcsinh": sp.asinh, "arccosh": sp.acosh, "arctanh": sp.atanh,
        "ln": sp.log, "log": lambda v: sp.log(v, 10),
        "abs": sp.Abs, "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E, "i": sp.I,
    }

    @staticmethod
    def evaluate(cached: CachedAnalysis, x_symbol: sp.Symbol, x_vals: np.ndarray):
        with np.errstate(all="ignore"):
            y_vals = np.asarray(cached.f_lambdified(x_vals), dtype=float)
            if y_vals.ndim == 0: y_vals = np.full_like(x_vals, float(y_vals))
        mask = np.zeros_like(x_vals, dtype=bool)
        for low, high, left_open, right_open in cached.domain_intervals:
            mask |= ((x_vals > low if left_open else x_vals >= low) & (x_vals < high if right_open else x_vals <= high))
        for p in cached.domain_excluded: mask &= np.abs(x_vals - p) > 1e-9
        y_vals[~mask] = np.nan
        y_vals[~np.isfinite(y_vals)] = np.nan
        return y_vals

    @staticmethod
    def adaptive_sample(f_lambdified, xmin, xmax, pois=None, base_points=600, max_extra_rounds=5, max_points=10000):
        xs = list(np.linspace(xmin, xmax, base_points))
        if pois:
            for p in pois:
                if xmin <= p <= xmax:
                    for eps in [1e-3, 1e-5, 1e-7]:
                        if p + eps <= xmax: xs.append(p + eps)
                        if p - eps >= xmin: xs.append(p - eps)
        xs = np.unique(np.sort(xs))
        with np.errstate(all="ignore"):
            ys = np.asarray(f_lambdified(xs), dtype=float)
            if ys.ndim == 0: ys = np.full_like(xs, float(ys))
        for _ in range(max_extra_rounds):
            dy = np.abs(np.diff(ys))
            dx = np.diff(xs)
            valid = (dx > (xmax - xmin) * 1e-6) & (~np.isnan(dy))
            needs_insert = valid & (dy > np.maximum(np.abs(ys[:-1]), np.abs(ys[1:])) * 0.05 + 1e-6)
            nan_transitions = (np.isnan(ys[:-1]) != np.isnan(ys[1:])) & (dx > (xmax - xmin) * 1e-6)
            mask = needs_insert | nan_transitions
            insert_positions = np.where(mask)[0]
            if not len(insert_positions): break
            insert_x = (xs[insert_positions] + xs[insert_positions + 1]) / 2.0
            with np.errstate(all="ignore"):
                insert_y = np.asarray(f_lambdified(insert_x), dtype=float)
                if insert_y.ndim == 0: insert_y = np.full_like(insert_x, float(insert_y))
            xs = np.insert(xs, insert_positions + 1, insert_x)
            ys = np.insert(ys, insert_positions + 1, insert_y)
            if len(xs) > max_points: break
        return xs, ys

    @staticmethod
    def adaptive_evaluate(cached: CachedAnalysis, x_symbol: sp.Symbol, xmin: float, xmax: float):
        pois = set(cached.v_asymptotes_exact)
        for low, high, _, _ in cached.domain_intervals:
            if np.isfinite(low): pois.add(low)
            if np.isfinite(high): pois.add(high)
        x_vals, y_vals = GraphEngine.adaptive_sample(cached.f_lambdified, xmin, xmax, pois=list(pois))
        mask = np.zeros_like(x_vals, dtype=bool)
        for low, high, left_open, right_open in cached.domain_intervals:
            mask |= ((x_vals > low if left_open else x_vals >= low) & (x_vals < high if right_open else x_vals <= high))
        for p in cached.domain_excluded: mask &= np.abs(x_vals - p) > 1e-9
        y_vals = y_vals.copy()
        y_vals[~mask] = np.nan
        y_vals[~np.isfinite(y_vals)] = np.nan
        return x_vals, y_vals

    @staticmethod
    def break_discontinuities(x_vals: np.ndarray, y_vals: np.ndarray, y_min: float, y_max: float):
        y_plot = y_vals.copy()
        view_range = abs(y_max - y_min)
        if view_range <= 0 or len(y_plot) < 2: return y_plot
        threshold = view_range * 1.5
        dy = np.abs(np.diff(y_plot))
        cross_zero = y_plot[:-1] * y_plot[1:] < 0
        breaks = (dy > threshold) & cross_zero
        break_indices = np.where(breaks)[0]
        y_plot[break_indices] = np.nan
        y_plot[break_indices + 1] = np.nan
        return y_plot

    @staticmethod
    def find_vertical_asymptotes_numeric(x_vals: np.ndarray, y_vals: np.ndarray, y_min: float, y_max: float):
        asymptotes = []
        view_range = abs(y_max - y_min)
        if view_range <= 0 or len(y_vals) < 2: return asymptotes
        threshold = view_range * 1.5
        dy = np.abs(np.diff(y_vals))
        cross_zero = y_vals[:-1] * y_vals[1:] < 0
        breaks = (dy > threshold) & cross_zero
        break_indices = np.where(breaks)[0]
        for i in break_indices:
            if not np.isnan(y_vals[i]) and not np.isnan(y_vals[i+1]):
                asymptotes.append(float((x_vals[i] + x_vals[i+1]) / 2.0))
        return asymptotes

    @staticmethod
    def find_roots_in_range(cached: CachedAnalysis, x_symbol: sp.Symbol, x_min: float, x_max: float):
        if cached.sym_roots:
            num_r = []
            for r in cached.sym_roots:
                try:
                    val = float(r.evalf())
                    if x_min <= val <= x_max: num_r.append(val)
                except Exception: continue
            return num_r, cached.sym_roots
        found = []
        try:
            xs = np.linspace(x_min, x_max, 2500)
            with np.errstate(all="ignore"): ys = np.asarray(cached.f_lambdified(xs), dtype=float)
            crossings = np.where((ys[:-1] * ys[1:] <= 0) & np.isfinite(ys[:-1]) & np.isfinite(ys[1:]))[0]
            for i in crossings:
                if len(found) >= MAX_ROOT_SEARCH: break
                try:
                    r = float(sp.nsolve(cached.expr, x_symbol, (xs[i], xs[i+1])))
                    if x_min <= r <= x_max: found.append(r)
                except Exception: continue
        except Exception: pass
        found.sort()
        unique = []
        for r in found:
            if not unique or abs(r - unique[-1]) > 1e-6: unique.append(r)
        return unique, []

    @staticmethod
    def find_extrema_in_range(cached: CachedAnalysis, x_symbol: sp.Symbol, x_min: float, x_max: float):
        return ([(x, y) for x, y, kind in cached.critical_points if kind == "max" and x_min <= x <= x_max],
                [(x, y) for x, y, kind in cached.critical_points if kind == "min" and x_min <= x <= x_max])

class PlotCanvas(FigureCanvas):
    def __init__(self, main_window):
        self.main_window = main_window
        self.fig = Figure(facecolor='#0F1017')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setup_axes()
        self._is_panning = False
        self._pan_start = None
        self._zoom_timer = QTimer(self)
        self._zoom_timer.setSingleShot(True)
        self._zoom_timer.setInterval(120)
        self._zoom_timer.timeout.connect(lambda: self.main_window.draw_graph())
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def setup_axes(self):
        self.ax.set_facecolor('#0F1017')
        if self.main_window.grid_visible:
            self.ax.grid(True, linestyle='--', color='#232536', alpha=0.6)
        else:
            self.ax.grid(False)
        self.ax.axhline(0, color='#64748B', linewidth=1.2)
        self.ax.axvline(0, color='#64748B', linewidth=1.2)
        self.ax.tick_params(colors='#94A3B8', labelsize=9)
        for spine in self.ax.spines.values(): spine.set_color('#232536')
        self.fig.subplots_adjust(left=0.095, right=0.975, top=0.94, bottom=0.12)

    def on_scroll(self, event):
        if event.inaxes != self.ax: return
        factor = 0.85 if event.button == 'up' else 1.15
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        self.ax.set_xlim(xdata - (xdata - cur_xlim[0]) * factor, xdata + (cur_xlim[1] - xdata) * factor)
        self.ax.set_ylim(ydata - (ydata - cur_ylim[0]) * factor, ydata + (cur_ylim[1] - ydata) * factor)
        self.draw_idle()
        self.main_window.update_spin_boxes_silent(*self.ax.get_xlim(), *self.ax.get_ylim())
        self._zoom_timer.start()

    def on_press(self, event):
        if event.button == 1:
            if event.dblclick: self.main_window.reset_view()
            else:
                self._is_panning = True
                self._pan_start = (event.x, event.y)

    def on_release(self, event):
        if event.button == 1 and self._is_panning:
            self._is_panning = False
            self._pan_start = None
            self.main_window.update_spin_boxes_silent(*self.ax.get_xlim(), *self.ax.get_ylim())
            self.main_window.draw_graph()

    def on_motion(self, event):
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.main_window.update_coord_status(event.xdata, event.ydata)
            if not self._is_panning:
                hovered_card = None
                for card, line in list(self.main_window.plotted_lines.items()):
                    try:
                        if line.contains(event)[0]:
                            hovered_card = card
                            break
                    except Exception: pass
                if hovered_card is not None: self.main_window.highlight_function(hovered_card)
                else: self.main_window.clear_highlight()
        elif not self._is_panning: self.main_window.clear_highlight()
        if not self._is_panning or self._pan_start is None or event.inaxes != self.ax: return
        dx, dy = event.x - self._pan_start[0], event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        dx_data = -dx * ((xlim[1] - xlim[0]) / self.width())
        dy_data = -dy * ((ylim[1] - ylim[0]) / self.height())
        self.ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
        self.ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
        self.draw_idle()

class FunctionCard(QFrame):
    card_deleted = Signal(object)
    duplicate_requested = Signal(object)
    data_changed = Signal()
    hover_enter = Signal(object)
    hover_leave = Signal(object)

    def __init__(self, index: int, color: str = "#2563EB", expression: str = "x**2"):
        super().__init__()
        self.setObjectName("functionCard")
        self.color = color
        self.is_visible = True
        self.fill_area = False
        self.cached_analysis = None
        self.last_parsed_str = ""
        card_layout = QVBoxLayout(self)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(4)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.color_btn = QPushButton()
        self.color_btn.setObjectName("colorPickerBtn")
        self.color_btn.setFixedSize(16, 16)
        self.update_color_style()
        self.color_btn.clicked.connect(self.open_color_dialog)
        self.label = QLabel(f"{get_func_name(index)}(x) =")
        self.label.setFont(QFont("Fira Code", 10, QFont.Bold))
        self.label.setStyleSheet("color: #94A3B8;")
        self.input_field = QLineEdit(expression)
        self.input_field.setObjectName("funcInput")
        self.input_field.setCursorPosition(0)
        self.input_field.textChanged.connect(lambda: self.data_changed.emit())
        self.fill_btn = QPushButton("Area")
        self.fill_btn.setObjectName("fillBtn")
        self.fill_btn.setFixedSize(40, 26)
        self.fill_btn.setProperty("fill_state", "false")
        self.fill_btn.clicked.connect(self.toggle_fill)
        self.toggle_btn = QPushButton("ON")
        self.toggle_btn.setObjectName("toggleBtn")
        self.toggle_btn.setFixedSize(32, 26)
        self.toggle_btn.clicked.connect(self.toggle_visibility)
        self.duplicate_btn = QPushButton("⧉")
        self.duplicate_btn.setObjectName("duplicateBtn")
        self.duplicate_btn.setFixedSize(28, 26)
        self.duplicate_btn.setToolTip("Duplicate function")
        self.duplicate_btn.clicked.connect(lambda: self.duplicate_requested.emit(self))
        self.delete_btn = QPushButton("Del")
        self.delete_btn.setObjectName("deleteBtn")
        self.delete_btn.setFixedSize(32, 26)
        self.delete_btn.clicked.connect(lambda: self.card_deleted.emit(self))
        layout.addWidget(self.color_btn)
        layout.addWidget(self.label)
        layout.addWidget(self.input_field, 1)
        layout.addWidget(self.fill_btn)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.duplicate_btn)
        layout.addWidget(self.delete_btn)
        card_layout.addLayout(layout)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet("color: #FCA5A5; font-size: 11px; padding: 2px 22px 0 22px;")
        self.message_label.hide()
        card_layout.addWidget(self.message_label)

    def set_index(self, index: int, argument: str = "x"):
        self.label.setText(f"{get_func_name(index)}({argument}) =")

    def open_color_dialog(self):
        new_color = QColorDialog.getColor(QColor(self.color), self, "Select Color")
        if new_color.isValid():
            self.color = new_color.name()
            self.update_color_style()
            self.data_changed.emit()

    def update_color_style(self):
        self.color_btn.setStyleSheet(f"background-color: {self.color}; border: 1px solid #475569;")

    def toggle_visibility(self):
        self.is_visible = not self.is_visible
        self.toggle_btn.setText("ON" if self.is_visible else "OFF")
        self.toggle_btn.setProperty("visible_state", "true" if self.is_visible else "false")
        self.toggle_btn.setStyle(self.toggle_btn.style())
        self.input_field.setEnabled(self.is_visible)
        self.data_changed.emit()

    def toggle_fill(self):
        self.fill_area = not self.fill_area
        self.fill_btn.setProperty("fill_state", "true" if self.fill_area else "false")
        self.fill_btn.setStyle(self.fill_btn.style())
        self.data_changed.emit()

    def get_expression(self) -> str:
        return self.input_field.text().strip()

    def get_analysis(self, x_symbol: sp.Symbol):
        expr_str = self.get_expression()
        if not expr_str:
            self.cached_analysis = None
            self.last_parsed_str = ""
            return None
        if self.cached_analysis is None or self.last_parsed_str != expr_str:
            self.cached_analysis = CachedAnalysis(expr_str, x_symbol)
            self.last_parsed_str = expr_str
        return self.cached_analysis

    def set_error(self, message: str = ""):
        if message:
            friendly = (str(message).strip().splitlines()[0] or "Invalid mathematical expression")[:160]
            self.message_label.setText(friendly)
            self.message_label.show()
            self.input_field.setToolTip(friendly)
            self.input_field.setStyleSheet("border-color: #EF4444;")
        else:
            self.message_label.clear()
            self.message_label.hide()
            self.input_field.setToolTip("")
            self.input_field.setStyleSheet("")

    def enterEvent(self, event):
        self.hover_enter.emit(self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hover_leave.emit(self)
        super().leaveEvent(event)

class FuncoraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(load_app_icon())
        self.setWindowTitle("Funcora")
        self.resize(1480, 820)
        self.setStyleSheet(DARK_STYLESHEET)
        self.cards = []
        self.default_colors = ["#38bdf8", "#f87171", "#4ade80", "#c084fc", "#fb923c", "#f472b6"]
        self.is_trig_mode = False
        self.grid_visible = True
        self.x_symbol = sp.Symbol("x")
        self.t_symbol = sp.Symbol("t")
        self.plot_mode = "cartesian"
        self.plotted_lines = {}
        self.function_text_ranges = {}
        self._active_card = None
        self.current_project_path = None
        self._draw_timer = QTimer(self)
        self._draw_timer.setSingleShot(True)
        self._draw_timer.setInterval(220)
        self._draw_timer.timeout.connect(self.draw_graph)
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        toolbar = QFrame()
        toolbar.setFixedHeight(46)
        toolbar.setStyleSheet("background-color: #161722; border-bottom: 1px solid #232536;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        brand_label = QLabel("Funcora")
        brand_label.setFont(QFont("Inter", 13, QFont.Bold))
        tb_layout.addWidget(brand_label)
        project_btn = QToolButton()
        project_btn.setText("Project")
        project_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        project_menu = QMenu(project_btn)
        for label, shortcut, callback in [
            ("New Project", "Ctrl+Shift+N", self.new_project),
            ("Open Project…", "Ctrl+O", self.open_project),
            ("Save Project…", "Ctrl+S", self.save_project),
        ]:
            action = project_menu.addAction(label)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
        project_menu.addSeparator()
        export_action = project_menu.addAction("Export Graph…", self.save_graph)
        export_action.setShortcut("Ctrl+E")
        project_btn.setMenu(project_menu)
        tb_layout.addWidget(project_btn)
        tb_layout.addStretch()
        for name, callback in [("Zoom +", self.zoom_in), ("Zoom -", self.zoom_out), ("Trig [-2π, 2π]", self.preset_trig),
                               ("Auto Scale", self.auto_scale), ("Reset View", self.reset_view), ("Save Graph", self.save_graph)]:
            btn = QToolButton()
            btn.setText(name)
            btn.clicked.connect(callback)
            tb_layout.addWidget(btn)
        self.grid_btn = QToolButton()
        self.grid_btn.setText("Grid: On")
        self.grid_btn.clicked.connect(self.toggle_grid)
        tb_layout.addWidget(self.grid_btn)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Cartesian", "Parametric", "Polar", "Implicit"])
        self.mode_combo.setToolTip("Plot mode")
        self.mode_combo.currentTextChanged.connect(lambda text: self.set_plot_mode(text.lower()))
        tb_layout.addWidget(self.mode_combo)
        main_layout.addWidget(toolbar)
        splitter = QSplitter(Qt.Horizontal)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(380)
        sidebar.setMaximumWidth(520)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        func_header = QHBoxLayout()
        func_title = QLabel("Functions")
        func_title.setFont(QFont("Inter", 11, QFont.Bold))
        examples_btn = QToolButton()
        examples_btn.setText("Examples")
        examples_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        examples_menu = QMenu(examples_btn)
        examples = [
            ("Quadratic", "cartesian", "x^2-4x+4"),
            ("Sine wave", "cartesian", "sin(x)"),
            ("Rational curve", "cartesian", "(x+1)/(x-2)"),
            ("Parametric circle", "parametric", "5cos(t), 5sin(t)"),
            ("Polar rose", "polar", "4cos(3t)"),
            ("Implicit circle", "implicit", "x^2+y^2=25"),
        ]
        for label, mode, expression in examples:
            action = examples_menu.addAction(label)
            action.triggered.connect(lambda checked=False, m=mode, e=expression: self.load_example(m, e))
        examples_btn.setMenu(examples_menu)
        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("primaryBtn")
        add_btn.clicked.connect(lambda: self.add_function_card())
        func_header.addWidget(func_title)
        func_header.addStretch()
        func_header.addWidget(examples_btn)
        func_header.addWidget(add_btn)
        sidebar_layout.addLayout(func_header)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.empty_label = QLabel("No functions yet.\nAdd one or start from an example.")
        self.empty_label.setObjectName("emptyState")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.cards_layout.addWidget(self.empty_label)
        self.cards_layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.cards_container)
        scroll_area.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { border: none; background-color: #161722; }")
        sidebar_layout.addWidget(scroll_area)
        clear_btn = QPushButton("Clear All")
        clear_btn.setObjectName("dangerBtn")
        clear_btn.clicked.connect(lambda: self.clear_all_cards(confirm=True))
        sidebar_layout.addWidget(clear_btn)
        graph_widget = QWidget()
        graph_layout = QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PlotCanvas(self)
        graph_layout.addWidget(self.canvas)
        bottom_bar = QFrame()
        bottom_bar.setFixedHeight(46)
        bottom_bar.setStyleSheet("background-color: #161722; border-top: 1px solid #232536;")
        bb_layout = QHBoxLayout(bottom_bar)
        bb_layout.setContentsMargins(12, 0, 12, 0)
        bb_layout.setSpacing(8)
        self.spins = {}
        for label, val in [("X Min", -10.0), ("X Max", 10.0), ("Y Min", -10.0), ("Y Max", 10.0)]:
            bb_layout.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setValue(val)
            spin.setSingleStep(1.0)
            spin.valueChanged.connect(self.on_spin_changed)
            bb_layout.addWidget(spin)
            self.spins[label] = spin
        bb_layout.addStretch()
        self.coord_label = QLabel("x: 0.00 | y: 0.00")
        self.coord_label.setStyleSheet("color: #94A3B8; font-family: 'Fira Code', monospace; font-size: 11px;")
        self.coord_label.setFixedWidth(160)
        bb_layout.addWidget(self.coord_label)
        graph_layout.addWidget(bottom_bar)
        analysis_frame = QFrame()
        analysis_frame.setObjectName("analysisBar")
        analysis_frame.setMinimumWidth(250)
        analysis_layout = QVBoxLayout(analysis_frame)
        analysis_layout.setContentsMargins(12, 12, 12, 12)
        analysis_header = QHBoxLayout()
        analysis_title = QLabel("Analysis Panel")
        analysis_title.setFont(QFont("Inter", 11, QFont.Bold))
        copy_btn = QToolButton()
        copy_btn.setText("Copy")
        copy_btn.setToolTip("Copy analysis to clipboard")
        copy_btn.clicked.connect(self.copy_analysis)
        analysis_header.addWidget(analysis_title)
        analysis_header.addStretch()
        analysis_header.addWidget(copy_btn)
        analysis_layout.addLayout(analysis_header)
        self.analysis_text = QTextEdit()
        self.analysis_text.setObjectName("analysisText")
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setPlaceholderText("Function metrics will appear here...")
        analysis_layout.addWidget(self.analysis_text)
        splitter.addWidget(sidebar)
        splitter.addWidget(graph_widget)
        splitter.addWidget(analysis_frame)
        splitter.setSizes([390, 690, 280])
        main_layout.addWidget(splitter)
        self.install_shortcuts()
        self.add_function_card("x**2-4x+4")

    def install_shortcuts(self):
        shortcuts = [
            ("New function", "Ctrl+N", lambda: self.add_function_card()),
            ("Reset view", "Ctrl+0", self.reset_view),
            ("Toggle grid", "Ctrl+Shift+G", self.toggle_grid),
        ]
        for name, shortcut, callback in shortcuts:
            action = QAction(name, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            self.addAction(action)

    def schedule_draw(self):
        self._draw_timer.start()

    def toggle_grid(self):
        self.grid_visible = not self.grid_visible
        self.grid_btn.setText("Grid: On" if self.grid_visible else "Grid: Off")
        self.draw_graph()

    def copy_analysis(self):
        text = self.analysis_text.toPlainText().strip()
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Analysis copied", 1800)

    def project_data(self):
        return {
            "version": 1,
            "mode": self.plot_mode,
            "grid_visible": self.grid_visible,
            "bounds": {name: spin.value() for name, spin in self.spins.items()},
            "functions": [
                {
                    "expression": card.get_expression(),
                    "color": card.color,
                    "visible": card.is_visible,
                    "fill_area": card.fill_area,
                }
                for card in self.cards
            ],
        }

    def save_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            path = QFileDialog.getSaveFileName(
                self, "Save Funcora Project", self.current_project_path or "Untitled.funcora",
                "Funcora Projects (*.funcora)"
            )[0]
        if not path:
            return False
        if not path.lower().endswith(".funcora"):
            path += ".funcora"
        try:
            with open(path, "w", encoding="utf-8") as project_file:
                json.dump(self.project_data(), project_file, indent=2)
        except (OSError, TypeError) as exc:
            QMessageBox.warning(self, "Could not save project", str(exc))
            return False
        self.current_project_path = path
        self.setWindowTitle(f"Funcora — {os.path.basename(path)}")
        self.statusBar().showMessage("Project saved", 1800)
        return True

    def open_project(self, path=None):
        if isinstance(path, bool):
            path = None
        if not path:
            path = QFileDialog.getOpenFileName(
                self, "Open Funcora Project", "", "Funcora Projects (*.funcora)"
            )[0]
        if not path:
            return False
        try:
            with open(path, "r", encoding="utf-8") as project_file:
                data = json.load(project_file)
            self.load_project_data(data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "Could not open project", str(exc))
            return False
        self.current_project_path = path
        self.setWindowTitle(f"Funcora — {os.path.basename(path)}")
        self.statusBar().showMessage("Project opened", 1800)
        return True

    def load_project_data(self, data):
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("This is not a supported Funcora project file.")
        mode = data.get("mode", "cartesian")
        if mode not in {"cartesian", "parametric", "polar", "implicit"}:
            raise ValueError("The project contains an unknown plot mode.")
        function_data = data.get("functions", [])
        if not isinstance(function_data, list):
            raise ValueError("The project function list is invalid.")

        self.clear_all_cards(redraw=False)
        self.set_plot_mode(mode, redraw=False)
        self.grid_visible = bool(data.get("grid_visible", True))
        self.grid_btn.setText("Grid: On" if self.grid_visible else "Grid: Off")
        bounds = data.get("bounds", {})
        for name, fallback in [("X Min", -10.0), ("X Max", 10.0), ("Y Min", -10.0), ("Y Max", 10.0)]:
            try:
                value = float(bounds.get(name, fallback))
            except (TypeError, ValueError):
                value = fallback
            self.spins[name].blockSignals(True)
            self.spins[name].setValue(value)
            self.spins[name].blockSignals(False)

        for item in function_data:
            if not isinstance(item, dict):
                continue
            card = self.add_function_card(
                str(item.get("expression", "x")),
                color=str(item.get("color", "#2563EB")),
                redraw=False,
            )
            card.is_visible = bool(item.get("visible", True))
            card.toggle_btn.setText("ON" if card.is_visible else "OFF")
            card.toggle_btn.setProperty("visible_state", "true" if card.is_visible else "false")
            card.toggle_btn.setStyle(card.toggle_btn.style())
            card.input_field.setEnabled(card.is_visible)
            card.fill_area = bool(item.get("fill_area", False))
            card.fill_btn.setProperty("fill_state", "true" if card.fill_area else "false")
            card.fill_btn.setStyle(card.fill_btn.style())
        self.update_empty_state()
        self.draw_graph()

    def new_project(self):
        if self.cards:
            result = QMessageBox.question(
                self, "New project", "Start a new project? Unsaved changes will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if result != QMessageBox.StandardButton.Yes:
                return
        self.current_project_path = None
        self.setWindowTitle("Funcora")
        self.clear_all_cards(redraw=False)
        self.set_plot_mode("cartesian", redraw=False)
        self.grid_visible = True
        self.grid_btn.setText("Grid: On")
        self.update_spin_boxes_silent(-10.0, 10.0, -10.0, 10.0)
        self.add_function_card("x")
        self.statusBar().showMessage("New project created", 1800)

    def update_coord_status(self, x, y):
        self.coord_label.setText(f"x: {x:.2f} | y: {y:.2f}")

    def update_spin_boxes_silent(self, xmin, xmax, ymin, ymax):
        for name, val in zip(["X Min", "X Max", "Y Min", "Y Max"], [xmin, xmax, ymin, ymax]):
            self.spins[name].blockSignals(True)
            self.spins[name].setValue(val)
            self.spins[name].blockSignals(False)

    def add_function_card(self, expr: str = "x", color=None, redraw=True):
        idx = len(self.cards) + 1
        color = color or self.default_colors[(idx - 1) % len(self.default_colors)]
        card = FunctionCard(index=idx, color=color, expression=expr)
        card.card_deleted.connect(self.remove_function_card)
        card.duplicate_requested.connect(self.duplicate_function_card)
        card.data_changed.connect(self.schedule_draw)
        card.hover_enter.connect(self.highlight_function)
        card.hover_leave.connect(lambda c: self.clear_highlight())
        self.cards.append(card)
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self.reindex_cards()
        self.apply_mode_placeholder(card)
        self.update_empty_state()
        if redraw:
            self.draw_graph()
        return card

    def duplicate_function_card(self, source: FunctionCard):
        card = self.add_function_card(source.get_expression(), color=source.color, redraw=False)
        card.fill_area = source.fill_area
        card.fill_btn.setProperty("fill_state", "true" if card.fill_area else "false")
        card.fill_btn.setStyle(card.fill_btn.style())
        self.draw_graph()
        card.input_field.setFocus()
        card.input_field.selectAll()
        self.statusBar().showMessage("Function duplicated", 1800)

    def remove_function_card(self, card: FunctionCard):
        if card in self.cards:
            self.cards.remove(card)
            self.cards_layout.removeWidget(card)
            self.plotted_lines.pop(card, None)
            self.function_text_ranges.pop(card, None)
            if self._active_card is card: self._active_card = None
            card.deleteLater()
            self.reindex_cards()
            self.update_empty_state()
            self.draw_graph()

    def clear_all_cards(self, redraw=True, confirm=False):
        if confirm and self.cards:
            result = QMessageBox.question(
                self, "Clear functions", "Remove all functions from this project?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if result != QMessageBox.StandardButton.Yes:
                return False
        for card in list(self.cards):
            self.cards.remove(card)
            self.cards_layout.removeWidget(card)
            card.deleteLater()
        self.plotted_lines = {}
        self.function_text_ranges = {}
        self._active_card = None
        self.reindex_cards()
        self.update_empty_state()
        if redraw:
            self.draw_graph()
        return True

    def update_empty_state(self):
        self.empty_label.setVisible(not self.cards)

    def reindex_cards(self):
        argument = {"parametric": "t", "polar": "θ", "implicit": "x,y"}.get(self.plot_mode, "x")
        for i, card in enumerate(self.cards, start=1):
            card.set_index(i, argument)

    def apply_mode_placeholder(self, card):
        if self.plot_mode == "parametric": card.input_field.setPlaceholderText("cos(t), sin(t)")
        elif self.plot_mode == "polar": card.input_field.setPlaceholderText("1 + cos(t)")
        elif self.plot_mode == "implicit": card.input_field.setPlaceholderText("x**2 + y**2 = 25")
        else: card.input_field.setPlaceholderText("e.g. x^2 - 4x + 4")
        card.fill_btn.setVisible(self.plot_mode == "cartesian")

    def set_plot_mode(self, mode, redraw=True):
        if mode not in {"cartesian", "parametric", "polar", "implicit"}:
            return
        self.plot_mode = mode
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(mode.capitalize())
        self.mode_combo.blockSignals(False)
        for card in self.cards: self.apply_mode_placeholder(card)
        self.reindex_cards()
        self.clear_highlight()
        if redraw:
            self.draw_graph()

    def load_example(self, mode, expression):
        self.set_plot_mode(mode, redraw=False)
        if len(self.cards) == 1 and self.cards[0].get_expression() in {"x", "x**2-4x+4"}:
            self.cards[0].input_field.setText(expression)
            self.cards[0].input_field.setCursorPosition(0)
        else:
            self.add_function_card(expression, redraw=False)
        self.draw_graph()
        self.statusBar().showMessage("Example loaded", 1800)

    def highlight_function(self, card):
        if self._active_card is card: return
        self._active_card = card
        for c, obj in self.plotted_lines.items():
            if hasattr(obj, 'set_linewidth'):
                obj.set_linewidth(3.2 if c is card else 1.2)
                obj.set_alpha(1.0 if c is card else 0.25)
        self.canvas.draw_idle()
        card.setStyleSheet(f"QFrame#functionCard {{ border: 2px solid {card.color}; }}")
        self.highlight_analysis_text(card)

    def clear_highlight(self):
        if self._active_card is None: return
        for c, obj in self.plotted_lines.items():
            if hasattr(obj, 'set_linewidth'):
                obj.set_linewidth(2.0)
                obj.set_alpha(1.0)
            c.setStyleSheet("")
        self._active_card = None
        self.canvas.draw_idle()
        self.analysis_text.setExtraSelections([])

    def highlight_analysis_text(self, card):
        if not (rng := self.function_text_ranges.get(card)): return
        cursor = self.analysis_text.textCursor()
        cursor.setPosition(rng[0])
        cursor.setPosition(rng[1], QTextCursor.KeepAnchor)
        fmt = QTextCharFormat()
        hc = QColor(card.color)
        hc.setAlpha(60)
        fmt.setBackground(hc)
        selection = QTextEdit.ExtraSelection()
        selection.cursor, selection.format = cursor, fmt
        self.analysis_text.setExtraSelections([selection])
        self.analysis_text.setTextCursor(cursor)
        self.analysis_text.ensureCursorVisible()

    def draw_graph(self):
        self._draw_timer.stop()
        if self.plot_mode == "parametric": self._draw_parametric()
        elif self.plot_mode == "polar": self._draw_polar()
        elif self.plot_mode == "implicit": self._draw_implicit()
        else: self._draw_cartesian()

    def _draw_cartesian(self):
        self.canvas.ax.clear()
        self.canvas.setup_axes()
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        self.plotted_lines, self.function_text_ranges = {}, {}
        if xmin >= xmax or ymin >= ymax: return
        analysis_output = ""
        any_drawn = False
        valid_funcs = []
        for idx, card in enumerate(self.cards, start=1):
            card.set_error()
            if not card.is_visible: continue
            try:
                cached = card.get_analysis(self.x_symbol)
            except Exception as exc:
                card.set_error(f"Could not parse: {exc}")
                continue
            if cached:
                valid_funcs.append((idx, card, cached))

        intersections = []
        if len(valid_funcs) > 1:
            for i in range(len(valid_funcs)):
                for j in range(i + 1, len(valid_funcs)):
                    c1, c2 = valid_funcs[i][1], valid_funcs[j][1]
                    f1_cached = valid_funcs[i][2]
                    f2_cached = valid_funcs[j][2]
                    diff_expr = f1_cached.expr - f2_cached.expr

                    found_roots = []

                    try:
                        sol = sp.solveset(diff_expr, self.x_symbol, domain=sp.Interval(xmin, xmax))
                        if isinstance(sol, sp.FiniteSet):
                            found_roots = [float(r.evalf()) for r in sol]
                    except Exception:
                        pass

                    if not found_roots:
                        cx = np.linspace(xmin, xmax, 1200) 
                        with np.errstate(all="ignore"):
                            y1 = f1_cached.f_lambdified(cx)
                            y2 = f2_cached.f_lambdified(cx)
                            diff = y1 - y2

                            valid_mask = np.isfinite(diff[:-1]) & np.isfinite(diff[1:])
                            crossings = np.where((diff[:-1] * diff[1:] <= 0) & valid_mask)[0]

                            for cr in crossings:
                                try:
                                    rx = float(sp.nsolve(valid_funcs[i][2].expr - valid_funcs[j][2].expr, self.x_symbol, cx[cr]))
                                    if xmin <= rx <= xmax:
                                        f1_cached = valid_funcs[i][2]
                                        f2_cached = valid_funcs[j][2]
                                        
                                        y1 = float(f1_cached.f_lambdified(rx))
                                        y2 = float(f2_cached.f_lambdified(rx))
                                        
                                        if not (np.isfinite(y1) and np.isfinite(y2)) or abs(y1) > 1e4 or abs(y2) > 1e4:
                                            continue
                                        
                                        if abs(y1 - y2) > 1e-3:
                                            continue

                                        ry = (y1 + y2) / 2.0
                                        rx_clean = 0.0 if abs(rx) < 1e-7 else rx
                                        ry_clean = 0.0 if abs(ry) < 1e-7 else ry

                                        if all(abs(rx_clean - ex) > 1e-5 for ex, _, _, _ in intersections):
                                            intersections.append((rx_clean, ry_clean, c1, c2))
                                except Exception: 
                                    pass
                    for rx in found_roots:
                        if xmin <= rx <= xmax:
                            try:
                                ry = float(f1_cached.f_lambdified(rx))
                                if np.isfinite(ry):
                                    rx_clean = 0.0 if abs(rx) < 1e-7 else rx
                                    ry_clean = 0.0 if abs(ry) < 1e-7 else ry

                                    if all(abs(rx_clean - ex) > 1e-4 for ex, _, _, _ in intersections):
                                        intersections.append((rx_clean, ry_clean, c1, c2))
                            except Exception:
                                pass

        for idx, card, cached in valid_funcs:
            try:
                x_vals, y_vals = GraphEngine.adaptive_evaluate(cached, self.x_symbol, xmin, xmax)
                func_name = f"{get_func_name(idx)}(x)"
                numeric_v_asymptotes = GraphEngine.find_vertical_asymptotes_numeric(x_vals, y_vals, ymin, ymax)
                combined_v_asymptotes = list(cached.v_asymptotes_exact)
                merge_tol = max((xmax - xmin) * 0.01, 1e-6)
                for na in numeric_v_asymptotes:
                    if not any(abs(na - a) < merge_tol for a in combined_v_asymptotes):
                        combined_v_asymptotes.append(na)
                combined_v_asymptotes.sort()
                y_plot = GraphEngine.break_discontinuities(x_vals, y_vals, ymin, ymax)
                block_start = len(analysis_output)
                line, = self.canvas.ax.plot(x_vals, y_plot, color=card.color, linewidth=2, label=f"{func_name} = {cached.raw_str}")
                self.plotted_lines[card] = line
                if card.fill_area:
                    self.canvas.ax.fill_between(x_vals, y_plot, 0, color=card.color, alpha=0.15, zorder=1)
                any_drawn = True
                for asymp in [a for a in combined_v_asymptotes if xmin <= a <= xmax]:
                    self.canvas.ax.axvline(x=asymp, color=card.color, linestyle=(0, (4, 6)), alpha=0.35, linewidth=1.5)
                for hasymp in cached.h_asymptotes_exact:
                    if ymin <= hasymp <= ymax:
                        self.canvas.ax.axhline(y=hasymp, color=card.color, linestyle=(0, (4, 6)), alpha=0.35, linewidth=1.5)
                for hx, hy in cached.holes:
                    if xmin <= hx <= xmax and ymin <= hy <= ymax:
                        self.canvas.ax.scatter(hx, hy, s=50, facecolors="#0F1017", edgecolors=card.color, linewidths=1.8, zorder=6)
                num_roots, sym_roots = GraphEngine.find_roots_in_range(cached, self.x_symbol, xmin, xmax)
                roots_to_plot = [(x, m) for x, m in cached.root_info if xmin <= x <= xmax] if cached.root_info else [(x, 1) for x in num_roots]
                for r_num, mult in roots_to_plot:
                    tangent = mult % 2 == 0
                    self.canvas.ax.scatter(r_num, 0, marker="D" if tangent else "o", color=card.color, s=34 if len(roots_to_plot) <= MAX_ROOT_LABELS else 16, zorder=5, edgecolor="#0F1017", linewidth=1.2)
                    if len(roots_to_plot) <= MAX_ROOT_LABELS:
                        self.canvas.ax.annotate((format_pi(r_num, 0) if self.is_trig_mode else fmt_num(r_num)) + (" (t)" if tangent else ""), xy=(r_num, 0), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=8, color="#FFFFFF", bbox=dict(facecolor="#1E202E", edgecolor="none", alpha=0.8, boxstyle="round,pad=0.2"))
                maximums, minimums = GraphEngine.find_extrema_in_range(cached, self.x_symbol, xmin, xmax)
                analysis_output += f"{func_name} = {cached.raw_str}\n\nRoots: "
                if cached.root_info: analysis_output += ", ".join([f"{pretty(r)}{' (t)' if m % 2 == 0 else ''}" for r in sym_roots for rx, m in cached.root_info if abs(rx - float(r.evalf())) < 1e-6])
                elif num_roots: analysis_output += f"{len(num_roots)} roots" if len(num_roots) > MAX_ROOT_LABELS else ", ".join([fmt_num(r) for r in num_roots])
                else: analysis_output += "None"
                analysis_output += f"\nDomain: {domain_pretty(cached.domain)}\nHoles: {', '.join([f'({fmt_num(hx)}, {fmt_num(hy)})' for hx, hy in cached.holes if xmin <= hx <= xmax]) or 'None'}\n"
                analysis_output += f"V. Asymptotes: {f'{len(combined_v_asymptotes)} in view' if len(combined_v_asymptotes) > 10 else ', '.join([f'x = {fmt_num(a)}' for a in combined_v_asymptotes]) or 'None'}\n"
                analysis_output += f"H. Asymptotes: {f'{len(cached.h_asymptotes_exact)} in view' if len(cached.h_asymptotes_exact) > 10 else ', '.join([f'y = {fmt_num(a)}' for a in cached.h_asymptotes_exact]) or 'None'}\n"
                analysis_output += f"Local Max: {', '.join([f'({fmt_num(x)}, {fmt_num(y)})' for x, y in maximums]) or 'None'}\nLocal Min: {', '.join([f'({fmt_num(x)}, {fmt_num(y)})' for x, y in minimums]) or 'None'}\n\n"
                analysis_output += f"Derivative: {pretty(cached.derivative_1)}\nIntegral: {pretty(cached.integral)} + C\n\n\n"
                self.function_text_ranges[card] = (block_start, len(analysis_output))
            except Exception as exc:
                card.set_error(f"Could not render: {exc}")
                continue

        if intersections:
            analysis_output += "Intersections:\n"
            for ix, iy, c1, c2 in intersections:
                self.canvas.ax.scatter(ix, iy, color='#FBBF24', s=45, zorder=7, edgecolor="#0F1017", linewidth=1.2)
                analysis_output += f"({fmt_num(ix)}, {fmt_num(iy)})\n"

        self.canvas.ax.set_xlim(xmin, xmax)
        self.canvas.ax.set_ylim(ymin, ymax)
        if self.is_trig_mode:
            w = xmax - xmin
            if w <= 3 * np.pi: step = np.pi / 4
            elif w <= 8 * np.pi: step = np.pi / 2
            elif w <= 16 * np.pi: step = np.pi
            elif w <= 32 * np.pi: step = 2 * np.pi
            elif w <= 80 * np.pi: step = 5 * np.pi
            elif w <= 160 * np.pi: step = 10 * np.pi
            else: step = max(25.0, np.round(w / (10 * np.pi))) * np.pi
            self.canvas.ax.xaxis.set_major_locator(MultipleLocator(step))
            self.canvas.ax.xaxis.set_major_formatter(FuncFormatter(format_pi))
        else:
            self.canvas.ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            self.canvas.ax.xaxis.set_major_formatter(ScalarFormatter())
        self.canvas.ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        if any_drawn: self.canvas.ax.legend(facecolor="#1E202E", edgecolor="#2B2D42", labelcolor="#E2E8F0").get_frame().set_alpha(0.85)
        self.canvas.draw()
        self.analysis_text.setPlainText(analysis_output.strip())

    def _draw_parametric(self):
        self.canvas.ax.clear()
        self.canvas.setup_axes()
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        self.plotted_lines, self.function_text_ranges = {}, {}
        if xmin >= xmax or ymin >= ymax: return
        t_vals = np.linspace(-20.0, 20.0, 4000)
        any_drawn = False
        transformations = standard_transformations + (implicit_multiplication_application,)
        for idx, card in enumerate(self.cards, start=1):
            card.set_error()
            if not card.is_visible: continue
            expr_str = card.get_expression()
            if not expr_str: continue
            if "," not in expr_str:
                card.set_error("Enter a pair separated by a comma, e.g. cos(t), sin(t)")
                continue
            try:
                sanitize_math_input(expr_str)
                x_str, y_str = expr_str.split(",", 1)
                fx = sp.lambdify(self.t_symbol, parse_expr(x_str.strip().replace("^", "**"), transformations=transformations, local_dict=GraphEngine.ALIASES), modules=["numpy", GraphEngine.ALIASES])
                fy = sp.lambdify(self.t_symbol, parse_expr(y_str.strip().replace("^", "**"), transformations=transformations, local_dict=GraphEngine.ALIASES), modules=["numpy", GraphEngine.ALIASES])
                with np.errstate(all="ignore"):
                    xv, yv = np.asarray(fx(t_vals), dtype=float), np.asarray(fy(t_vals), dtype=float)
                    if xv.ndim == 0: xv = np.full_like(t_vals, float(xv))
                    if yv.ndim == 0: yv = np.full_like(t_vals, float(yv))
                line, = self.canvas.ax.plot(xv, yv, color=card.color, linewidth=2, label=f"{get_func_name(idx)}(t) = ({x_str.strip()}, {y_str.strip()})")
                self.plotted_lines[card] = line
                any_drawn = True
            except Exception as exc:
                card.set_error(f"Use x(t), y(t): {exc}")
                continue
        self.canvas.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        if self.is_trig_mode:
            w = xmax - xmin
            if w <= 3 * np.pi: step = np.pi / 4
            elif w <= 8 * np.pi: step = np.pi / 2
            elif w <= 16 * np.pi: step = np.pi
            elif w <= 32 * np.pi: step = 2 * np.pi
            elif w <= 80 * np.pi: step = 5 * np.pi
            elif w <= 160 * np.pi: step = 10 * np.pi
            else: step = max(25.0, np.round(w / (10 * np.pi))) * np.pi
            self.canvas.ax.xaxis.set_major_locator(MultipleLocator(step))
            self.canvas.ax.xaxis.set_major_formatter(FuncFormatter(format_pi))
        else:
            self.canvas.ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            self.canvas.ax.xaxis.set_major_formatter(ScalarFormatter())
        self.canvas.ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        if any_drawn: self.canvas.ax.legend(facecolor="#1E202E", edgecolor="#2B2D42", labelcolor="#E2E8F0").get_frame().set_alpha(0.85)
        self.canvas.draw()
        self.analysis_text.setPlainText("Parametric mode: x(t), y(t) with t from -20 to 20.")

    def _draw_polar(self):
        self.canvas.ax.clear()
        self.canvas.setup_axes()
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        self.plotted_lines, self.function_text_ranges = {}, {}
        if xmin >= xmax or ymin >= ymax: return
        t_vals = np.linspace(0, 8 * np.pi, 4000)
        any_drawn = False
        for idx, card in enumerate(self.cards, start=1):
            card.set_error()
            if not card.is_visible or not (expr_str := card.get_expression()): continue
            try:
                sanitize_math_input(expr_str)
                fr = sp.lambdify(self.t_symbol, parse_expr(expr_str.replace("^", "**"), transformations=standard_transformations + (implicit_multiplication_application,), local_dict=GraphEngine.ALIASES), modules=["numpy", GraphEngine.ALIASES])
                with np.errstate(all="ignore"):
                    rv = np.asarray(fr(t_vals), dtype=float)
                    if rv.ndim == 0: rv = np.full_like(t_vals, float(rv))
                line, = self.canvas.ax.plot(rv * np.cos(t_vals), rv * np.sin(t_vals), color=card.color, linewidth=2, label=f"{get_func_name(idx)}(θ) = {expr_str}")
                self.plotted_lines[card] = line
                any_drawn = True
            except Exception as exc:
                card.set_error(f"Could not parse r(θ): {exc}")
                continue
        self.canvas.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        if self.is_trig_mode:
            w = xmax - xmin
            if w <= 3 * np.pi: step = np.pi / 4
            elif w <= 8 * np.pi: step = np.pi / 2
            elif w <= 16 * np.pi: step = np.pi
            elif w <= 32 * np.pi: step = 2 * np.pi
            elif w <= 80 * np.pi: step = 5 * np.pi
            elif w <= 160 * np.pi: step = 10 * np.pi
            else: step = max(25.0, np.round(w / (10 * np.pi))) * np.pi
            self.canvas.ax.xaxis.set_major_locator(MultipleLocator(step))
            self.canvas.ax.xaxis.set_major_formatter(FuncFormatter(format_pi))
        else:
            self.canvas.ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            self.canvas.ax.xaxis.set_major_formatter(ScalarFormatter())
        self.canvas.ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        if any_drawn: self.canvas.ax.legend(facecolor="#1E202E", edgecolor="#2B2D42", labelcolor="#E2E8F0").get_frame().set_alpha(0.85)
        self.canvas.draw()
        self.analysis_text.setPlainText("Polar mode: r(θ) with θ from 0 to 8π.")

    def _draw_implicit(self):
        self.canvas.ax.clear()
        self.canvas.setup_axes()
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        self.plotted_lines, self.function_text_ranges = {}, {}
        if xmin >= xmax or ymin >= ymax: return
        y_grid, x_grid = np.mgrid[ymin:ymax:800j, xmin:xmax:800j]
        x_sym, y_sym = sp.Symbol("x"), sp.Symbol("y")
        any_drawn = False
        for idx, card in enumerate(self.cards, start=1):
            card.set_error()
            if not card.is_visible or not (expr_str := card.get_expression()): continue
            eq_str = f"({expr_str.split('=', 1)[0]})-({expr_str.split('=', 1)[1]})" if "=" in expr_str else expr_str
            try:
                sanitize_math_input(expr_str)
                f = sp.lambdify((x_sym, y_sym), parse_expr(eq_str.replace("^", "**"), transformations=standard_transformations + (implicit_multiplication_application,), local_dict=GraphEngine.ALIASES), modules=["numpy", GraphEngine.ALIASES])
                with np.errstate(all="ignore"): z_vals = f(x_grid, y_grid)
                cs = self.canvas.ax.contour(x_grid, y_grid, z_vals, levels=[0], colors=[card.color], linewidths=2, antialiased=True)
                try:
                    plotted_obj = cs.collections[0]
                except (AttributeError, IndexError):
                    plotted_obj = cs
                self.plotted_lines[card] = plotted_obj
                any_drawn = True
            except Exception as exc:
                card.set_error(f"Could not parse equation: {exc}")
                continue
        self.canvas.ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
        if self.is_trig_mode:
            w = xmax - xmin
            if w <= 3 * np.pi: step = np.pi / 4
            elif w <= 8 * np.pi: step = np.pi / 2
            elif w <= 16 * np.pi: step = np.pi
            elif w <= 32 * np.pi: step = 2 * np.pi
            elif w <= 80 * np.pi: step = 5 * np.pi
            elif w <= 160 * np.pi: step = 10 * np.pi
            else: step = max(25.0, np.round(w / (10 * np.pi))) * np.pi
            self.canvas.ax.xaxis.set_major_locator(MultipleLocator(step))
            self.canvas.ax.xaxis.set_major_formatter(FuncFormatter(format_pi))
        else:
            self.canvas.ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            self.canvas.ax.xaxis.set_major_formatter(ScalarFormatter())
        self.canvas.ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        self.canvas.draw()
        self.analysis_text.setPlainText("Implicit mode: e.g. x**2 + y**2 = 25")

    def set_axis_bounds(self, xmin, xmax, ymin, ymax):
        self.update_spin_boxes_silent(xmin, xmax, ymin, ymax)
        self.draw_graph()

    def on_spin_changed(self): self.schedule_draw()

    def zoom_in(self):
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        xr, yr = (xmax - xmin) * 0.35, (ymax - ymin) * 0.35
        self.set_axis_bounds(xc - xr, xc + xr, yc - yr, yc + yr)

    def zoom_out(self):
        xmin, xmax, ymin, ymax = self.spins["X Min"].value(), self.spins["X Max"].value(), self.spins["Y Min"].value(), self.spins["Y Max"].value()
        xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
        xr, yr = (xmax - xmin) * 0.7, (ymax - ymin) * 0.7
        self.set_axis_bounds(xc - xr, xc + xr, yc - yr, yc + yr)

    def reset_view(self):
        self.is_trig_mode = False
        self.set_axis_bounds(-10.0, 10.0, -10.0, 10.0)

    def preset_trig(self):
        self.is_trig_mode = True
        self.set_axis_bounds(-2.5 * np.pi, 2.5 * np.pi, -5.0, 5.0)

    def auto_scale(self):
        if self.plot_mode != "cartesian":
            self.reset_view()
            return
        xs, ys = [], []
        for card in self.cards:
            if not card.is_visible: continue
            try:
                cached = card.get_analysis(self.x_symbol)
            except Exception:
                continue
            if cached:
                try:
                    xs.extend([float(r.evalf()) for r in cached.sym_roots])
                    for x, y, _ in cached.critical_points:
                        xs.append(x)
                        ys.append(y)
                    probe = np.linspace(-15, 15, 400)
                    with np.errstate(all="ignore"): finite = (yv := GraphEngine.evaluate(cached, self.x_symbol, probe))[np.isfinite(yv)]
                    if finite.size:
                        ys.extend([float(np.percentile(finite, 5)), float(np.percentile(finite, 95))])
                        xs.extend([-8, 8])
                except Exception: continue
        if not xs or not ys:
            self.reset_view()
            return
        xmin_v, xmax_v = min(xs), max(xs)
        ymin_v, ymax_v = min(ys), max(ys)
        if xmax_v - xmin_v < 1e-6: xmin_v, xmax_v = xmin_v - 5, xmax_v + 5
        if ymax_v - ymin_v < 1e-6: ymin_v, ymax_v = ymin_v - 5, ymax_v + 5
        pad_x, pad_y = (xmax_v - xmin_v) * 0.25, (ymax_v - ymin_v) * 0.25
        self.is_trig_mode = False
        self.set_axis_bounds(xmin_v - pad_x, xmax_v + pad_x, ymin_v - pad_y, ymax_v + pad_y)
 
    def save_graph(self):
        if path := QFileDialog.getSaveFileName(self, "Save Graph", "", "PNG Images (*.png);;SVG Images (*.svg)")[0]:
            self.canvas.fig.savefig(path, facecolor=self.canvas.fig.get_facecolor(), edgecolor='none', dpi=300)
def resource_path(relative):
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), relative)
    return os.path.join(os.path.dirname(__file__), relative)
def load_app_icon() -> QIcon:
    for path in [
        resource_path("assets/icon.ico"),
        resource_path("assets/icon.png"),
        resource_path("assets/icon.icns")
    ]:
        if os.path.exists(path):
            icon = QIcon(path)
            if not icon.isNull():
                return icon
    return QIcon()

def create_splash_pixmap(size: int = 220) -> QPixmap:
    for path in [
        resource_path("assets/icon.ico"),
        resource_path("assets/icon.png"),
        resource_path("assets/icon.icns")
    ]:
        if os.path.exists(path):
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                return pixmap.scaled(
                    size,
                    size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
    return QPixmap()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(load_app_icon())
    splash = QSplashScreen(create_splash_pixmap(), Qt.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.FramelessWindowHint)
    splash.show()
    app.processEvents()
    window = FuncoraWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
