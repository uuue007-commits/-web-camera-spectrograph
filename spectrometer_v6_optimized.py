#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrometer v6 (Final Optimized & Fixed)
"""
import csv, json, os, platform, queue, threading, time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

try:
    from scipy.signal import find_peaks as _scipy_find_peaks, savgol_filter as _savgol
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

WIDTH, HEIGHT, FPS = 1920, 1080, 30
CONFIG_PATH = Path.home() / ".spectrometer_v6_calibration.json"

LAMP_PRESETS = {
    "CFL / ЛДС": [404.7, 435.8, 546.1, 577.0, 579.1],
    "Ртуть (Hg)": [404.7, 435.8, 546.1, 577.0, 579.1],
    "Водород (H)": [410.2, 434.0, 486.1, 656.3],
    "Гелий (He)": [447.1, 501.6, 587.6, 667.8],
    "Неон (Ne)": [540.1, 585.2, 614.3, 640.2, 692.9],
    "Натрий (Na)": [589.0, 589.6],
    "Свободный режим": [],
}

# ================= ОПТИМИЗИРОВАННЫЕ УТИЛИТЫ =================
def wavelengths_to_rgb_batch(wl_array):
    wl = np.asarray(wl_array, dtype=float)
    r, g, b = np.zeros_like(wl), np.zeros_like(wl), np.zeros_like(wl)
    m1 = (wl >= 380) & (wl <= 439); r[m1] = -(wl[m1] - 440) / 60; b[m1] = 1.0
    m2 = (wl >= 440) & (wl <= 489); g[m2] = (wl[m2] - 440) / 50; b[m2] = 1.0
    m3 = (wl >= 490) & (wl <= 509); g[m3] = 1.0; b[m3] = -(wl[m3] - 510) / 20
    m4 = (wl >= 510) & (wl <= 579); r[m4] = (wl[m4] - 510) / 70; g[m4] = 1.0
    m5 = (wl >= 580) & (wl <= 644); r[m5] = 1.0; g[m5] = -(wl[m5] - 645) / 65
    m6 = (wl >= 645) & (wl <= 780); r[m6] = 1.0
    
    factor = np.zeros_like(wl)
    f1 = (wl >= 380) & (wl <= 419); factor[f1] = 0.3 + 0.7 * (wl[f1] - 380) / 40
    f2 = (wl >= 420) & (wl <= 700); factor[f2] = 1.0
    f3 = (wl >= 701) & (wl <= 780); factor[f3] = 0.3 + 0.7 * (780 - wl[f3]) / 80
    
    gamma = 0.8
    def adj_vec(v):
        res = np.zeros_like(v)
        mask = v > 0
        res[mask] = np.clip(255 * ((v[mask] * factor[mask]) ** gamma), 0, 255)
        return res
    rr, gg, bb = adj_vec(r), adj_vec(g), adj_vec(b)
    zero_mask = (rr + gg + bb) == 0
    rr[zero_mask] = 155; gg[zero_mask] = 155; bb[zero_mask] = 155
    return np.stack([rr, gg, bb], axis=-1) / 255.0

def find_peaks(y, thresh_rel=0.15, min_dist=20):
    y = np.asarray(y, dtype=float)
    if len(y) < 3: return np.array([], dtype=int)
    thr = thresh_rel * (float(y.max()) if y.max() > 0 else 1.0)
    if HAS_SCIPY:
        peaks, _ = _scipy_find_peaks(y, height=thr, distance=min_dist)
        return peaks
    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > thr:
            if not peaks or i - peaks[-1] >= min_dist: peaks.append(i)
            elif y[i] > y[peaks[-1]]: peaks[-1] = i
    return np.array(peaks, dtype=int)

def smooth_data(data, window, polyorder=3):
    data = np.asarray(data, dtype=float)
    if window < 3: return data
    if HAS_SCIPY and window % 2 == 1 and polyorder < window:
        try: return _savgol(data, window, polyorder)
        except Exception: pass
    return np.convolve(data, np.ones(window)/window, mode="same")

def darken(hex_color, factor=0.8):
    if len(hex_color) == 4: hex_color = "#" + "".join(c * 2 for c in hex_color[1:])
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

def load_theremino_corrections(filepath):
    wl, co = [], []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("-"): continue
                parts = line.replace(",", ".").split()
                if len(parts) < 2: continue
                try: wl.append(float(parts[0])); co.append(float(parts[1]))
                except ValueError: continue
    except Exception: return None, None
    if len(wl) < 2: return None, None
    order = np.argsort(wl)
    wl, co = np.asarray(wl, dtype=float)[order], np.asarray(co, dtype=float)[order]
    uniq_wl, idx = np.unique(wl, return_index=True)
    return uniq_wl, co[idx]

# ================= ПОТОКОБЕЗОПАСНАЯ КАМЕРА =================
class Camera:
    def __init__(self, index=0):
        self.index, self.stop_event = index, threading.Event()
        self.lock, self.commands = threading.Lock(), queue.Queue()
        self.frame, self.sequence, self.fps, self.error = None, 0, 0.0, ""
        self.width, self.height = 0, 0
        self.queries, self.responses = queue.Queue(), {}
        self.responses_lock, self.request_counter = threading.Lock(), 0
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

         
    def run(self):
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
        cap = None
        for be in backends:
            try:
                c = cv2.VideoCapture(self.index, be)
                if c.isOpened(): cap = c; break
                c.release()
            except Exception: continue
        if cap is None:
            with self.lock: self.error = f"Не удалось открыть камеру {self.index}"
            return
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception: pass
            with self.lock:
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count, started = 0, time.monotonic()
            while not self.stop_event.is_set():
                while True:
                    try: prop, value = self.commands.get_nowait()
                    except queue.Empty: break
                    try: cap.set(prop, value)
                    except Exception as exc:
                        with self.lock: self.error = f"set {prop}: {exc}"
                while True:
                    try: request_id, props = self.queries.get_nowait()
                    except queue.Empty: break
                    values = [float(cap.get(p)) if cap.get(p) is not None else None for p in props]
                    with self.responses_lock: self.responses[request_id] = values
                
                ok, frame = cap.read()
                if not ok or frame is None:
                    with self.lock: self.error = "Камера не возвращает кадры."
                    self.stop_event.wait(0.1); continue
                
                count += 1
                now = time.monotonic()
                with self.lock:
                    self.frame = frame.copy() # ИСПРАВЛЕНИЕ: Копируем для безопасности
                    self.sequence += 1
                    if now - started >= 1.0:
                        self.fps = count / (now - started); self.error = ""
                        count, started = 0, now
        except Exception as exc:
            with self.lock: self.error = str(exc)
        finally: cap.release()

    def snapshot(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            return (frame, self.sequence, self.fps, self.error, self.width, self.height)

    def set_property(self, prop, value): self.commands.put((prop, value))
    def get_properties(self, props, timeout=1.0):
        with self.lock:
            self.request_counter += 1; request_id = self.request_counter
            self.queries.put((request_id, list(props)))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.responses_lock:
                if request_id in self.responses: return self.responses.pop(request_id)
            time.sleep(0.005)
        return [None] * len(props)
    def close(self): self.stop_event.set(); self.thread.join(timeout=1.5)

# ================= ГЛАВНОЕ ПРИЛОЖЕНИЕ =================
class SpectrometerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrometer v6 (Fixed)")
        self.root.geometry("1500x1000"); self.root.configure(bg="#1e1e1e")
        
        self.camera, self.last_sequence, self.frame_w, self.frame_h = None, -1, 0, 0
        self.running, self.camera_index, self.graph_paused = False, 0, False
        self.intensity = self.peak_intensity = self.raw_spectrum = self.wavelengths = self.dark_spectrum = None
        self.calib_px, self.calib_nm, self.calib_file = [], [], "caldata_v6.txt"
        self._default_calib_active = True
        self.irradiance_wl = self.irradiance_coef = None; self.irradiance_name = ""
        
        # ИСПРАВЛЕНИЕ: DoubleVar для экспозиции и гейна
        self.var_exposure = tk.DoubleVar(value=-6.0)
        self.var_gain = tk.DoubleVar(value=10.0)
        self.var_smooth, self.var_thresh, self.var_mindist = tk.IntVar(value=5), tk.IntVar(value=15), tk.IntVar(value=30)
        self.var_holdpeaks, self.var_dark, self.var_irradiance, self.var_logscale = tk.BooleanVar(value=False), tk.BooleanVar(value=False), tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.flip_cam, self.flip_spec = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.var_channel = tk.StringVar(value="Яркость")
        self.var_roi_center, self.var_roi_height, self.var_cam_index = tk.DoubleVar(value=50.0), tk.IntVar(value=20), tk.IntVar(value=0)
        self.var_status, self.var_calib, self.var_fps, self.var_corr_label = tk.StringVar(value="Готово"), tk.StringVar(value="НЕ КАЛИБРОВАНО"), tk.StringVar(value="FPS: 0"), tk.StringVar(value="Коррекция: не загружена")
        
        self.last_preview_time, self.last_plot_time, self._cached_colors_float = 0.0, 0.0, None
        self.fill_coll, self.line_main, self.peak_line, self.peak_texts = None, None, None, []

        # === Выделение области спектра ===
        self.selection_range = None       # (wl_min, wl_max) выделенная область, нм
        self.zoom_range = None            # текущий зум, None = полный вид
        self.sel_rect = None              # прямоугольник выделения (matplotlib)
        self.sel_start_x = None
        self.sel_dragging = False
        self.var_select_mode = tk.BooleanVar(value=False)
        self.var_only_sel_peaks = tk.BooleanVar(value=False)
        self.var_sel_label = tk.StringVar(value="Область: не выбрана")
        
        self._build_ui()
        self._setup_default_calibration(); self._load_calibration_file(); self._load_calibration_json()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(150, self.toggle_camera)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1e1e1e", height=45); header.pack(side="top", fill="x")
        tk.Label(header, text="Spectrometer v6", font=("Segoe UI", 16, "bold"), fg="#ffffff", bg="#1e1e1e").pack(side="left", padx=20, pady=6)
        tk.Label(header, textvariable=self.var_calib, font=("Segoe UI", 11, "bold"), fg="#ffb400", bg="#1e1e1e").pack(side="right", padx=20)
        tk.Label(header, textvariable=self.var_fps, font=("Consolas", 10), fg="#9bd96b", bg="#1e1e1e").pack(side="right", padx=10)

        graph_frame = tk.Frame(self.root, bg="#ffffff"); graph_frame.pack(side="top", fill="both", expand=True, padx=8, pady=4)
        self.fig = Figure(figsize=(12, 4), dpi=100, facecolor="white"); self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Длина волны, нм"); self.ax.set_ylabel("Интенсивность")
        self.ax.set_ylim(0, 255); self.ax.set_xlim(380, 780); self.ax.grid(True, alpha=0.3)
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12)
        self.line_main, = self.ax.plot([], [], color="black", linewidth=1.0, zorder=5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # Обработчики выделения области на графике
        self.canvas.mpl_connect("button_press_event", self._on_graph_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_graph_motion)
        self.canvas.mpl_connect("button_release_event", self._on_graph_release)

        mid_frame = tk.Frame(self.root, bg="#1e1e1e"); mid_frame.pack(side="top", fill="x", padx=8, pady=4)
        self.video_label = tk.Label(mid_frame, bg="black", borderwidth=1, relief="solid"); self.video_label.pack(side="left", padx=4)
        strip_frame = tk.Frame(mid_frame, bg="#1e1e1e"); strip_frame.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(strip_frame, text="Полоса спектра:", fg="#aaa", bg="#1e1e1e").pack(anchor="w")
        self.strip_label = tk.Label(strip_frame, bg="black", borderwidth=1, relief="solid"); self.strip_label.pack(fill="x", expand=True)

        ctrl = tk.Frame(self.root, bg="#2a2a2a"); ctrl.pack(side="bottom", fill="x", padx=8, pady=(4, 8))
        btn_row = tk.Frame(ctrl, bg="#2a2a2a"); btn_row.pack(fill="x", pady=4)
        def mk_btn(p, t, c, col="#3a7fd5"):
            b = tk.Button(p, text=t, command=c, bg=col, fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=10, pady=5, cursor="hand2", activebackground=darken(col))
            b.pack(side="left", padx=3); return b
        
        self.btn_cam = mk_btn(btn_row, "Стоп", self.toggle_camera, "#d54a3a")
        self.btn_pause = mk_btn(btn_row, "Пауза", self.toggle_pause, "#d4a23a")
        mk_btn(btn_row, "Настройки камеры", self.camera_settings, "#4a6a8a")
        mk_btn(btn_row, "Калибровка", self.open_calibration, "#3aa55a")
        mk_btn(btn_row, "Коррекция…", self.load_irradiance_file, "#8a6a2c")
        mk_btn(btn_row, "Тёмный спектр", self.save_dark, "#7a5a3a")
        mk_btn(btn_row, "Снимок PNG", self.save_png, "#5a3ad5")
        mk_btn(btn_row, "CSV", self.save_csv, "#b57a1a")
        mk_btn(btn_row, "Выход", self.on_close, "#555555")

        chk_row = tk.Frame(ctrl, bg="#2a2a2a"); chk_row.pack(fill="x", pady=2)
        def mk_chk(p, t, v, cmd=None):
            cb = tk.Checkbutton(p, text=t, variable=v, bg="#2a2a2a", fg="white", selectcolor="#444", activebackground="#2a2a2a", activeforeground="white", command=cmd)
            cb.pack(side="left", padx=8); return cb
        mk_chk(chk_row, "Flip камеры", self.flip_cam, self._on_chk_change)
        mk_chk(chk_row, "Flip спектра", self.flip_spec, self._on_chk_change)
        mk_chk(chk_row, "Hold Peaks", self.var_holdpeaks, self._on_chk_change)
        mk_chk(chk_row, "Dark subtraction", self.var_dark)
        mk_chk(chk_row, "Коррекция вкл", self.var_irradiance)
        mk_chk(chk_row, "Log шкала", self.var_logscale)
        tk.Label(chk_row, text="Канал:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=(20, 4))
        ttk.Combobox(chk_row, textvariable=self.var_channel, values=["Яркость", "Среднее RGB", "R", "G", "B"], state="readonly", width=13).pack(side="left")

        slider_row = tk.Frame(ctrl, bg="#2a2a2a"); slider_row.pack(fill="x", pady=2)
        def mk_slider(p, l, v, lo, hi, cmd=None, res=1):
            f = tk.Frame(p, bg="#2a2a2a"); f.pack(side="left", padx=6)
            tk.Label(f, text=l, fg="#ccc", bg="#2a2a2a", font=("Segoe UI", 8)).pack(anchor="w")
            s = tk.Scale(f, from_=lo, to=hi, orient="horizontal", variable=v, bg="#2a2a2a", fg="white", troughcolor="#444", highlightthickness=0, length=120, showvalue=True, command=cmd, resolution=res)
            s.pack(); return s
        mk_slider(slider_row, "Экспозиция", self.var_exposure, -13, -1, lambda v: self._set_cam_prop(cv2.CAP_PROP_EXPOSURE, float(v)), 0.1)
        mk_slider(slider_row, "Усиление", self.var_gain, 0, 100, lambda v: self._set_cam_prop(cv2.CAP_PROP_GAIN, float(v)), 1)
        mk_slider(slider_row, "Сглаживание", self.var_smooth, 1, 31)
        mk_slider(slider_row, "Порог пиков %", self.var_thresh, 1, 80)
        mk_slider(slider_row, "Мин. дистанция", self.var_mindist, 5, 200)

        roi_row = tk.Frame(ctrl, bg="#2a2a2a"); roi_row.pack(fill="x", pady=2)
        tk.Label(roi_row, text="Центр ROI, %:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=(6, 4))
        tk.Scale(roi_row, from_=0, to=100, orient="horizontal", variable=self.var_roi_center, bg="#2a2a2a", fg="white", troughcolor="#444", length=200, showvalue=True).pack(side="left", padx=4)
        tk.Label(roi_row, text="Толщина, px:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=(12, 4))
        tk.Spinbox(roi_row, from_=1, to=1080, increment=2, textvariable=self.var_roi_height, width=6, bg="#1e1e1e", fg="white").pack(side="left", padx=4)
        tk.Label(roi_row, text="Индекс камеры:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=(12, 4))
        tk.Spinbox(roi_row, from_=0, to=20, textvariable=self.var_cam_index, width=4, bg="#1e1e1e", fg="white").pack(side="left", padx=4)
        sel_row = tk.Frame(ctrl, bg="#2a2a2a"); sel_row.pack(fill="x", pady=2)
        tk.Label(sel_row, text="Выделение области:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=(6, 4))
        def mk_sel_btn(t, c, col="#3a7fd5"):
            b = tk.Button(sel_row, text=t, command=c, bg=col, fg="white",
                          font=("Segoe UI", 9), relief="flat", padx=10, pady=3,
                          cursor="hand2", activebackground=darken(col))
            b.pack(side="left", padx=3); return b
        mk_sel_btn("Режим выделения", self._toggle_select_mode, "#4a6a8a")
        mk_sel_btn("Приблизить",       self._zoom_to_selection,  "#3a7fd5")
        mk_sel_btn("Сбросить вид",     self._reset_zoom,         "#555555")
        mk_sel_btn("Снять выделение",  self._clear_selection,    "#7a5a3a")
        tk.Checkbutton(sel_row, text="Пики только в выделении",
                       variable=self.var_only_sel_peaks,
                       bg="#2a2a2a", fg="white", selectcolor="#444",
                       activebackground="#2a2a2a",
                       activeforeground="white").pack(side="left", padx=10)
        tk.Label(sel_row, textvariable=self.var_sel_label, fg="#ffb400",
                 bg="#2a2a2a", font=("Consolas", 9)).pack(side="left", padx=8)

        status_row = tk.Frame(ctrl, bg="#2a2a2a"); status_row.pack(fill="x")
        tk.Label(status_row, textvariable=self.var_status, fg="#9bd96b", bg="#2a2a2a", font=("Consolas", 10)).pack(side="left", padx=10, pady=4)
        tk.Label(status_row, textvariable=self.var_corr_label, fg="#ffb400", bg="#2a2a2a", font=("Consolas", 10)).pack(side="right", padx=10, pady=4)
        
    def _on_chk_change(self):
        if self.var_holdpeaks.get() and self.intensity is not None: self.peak_intensity = self.intensity.copy()
        if self.intensity is not None and self.wavelengths is not None: self._draw_graph()
    def _set_cam_prop(self, prop, val):
        if self.camera is not None: self.camera.set_property(prop, val)
    def toggle_pause(self):
        if self.graph_paused:
            self.graph_paused = False; self.btn_pause.config(text="Пауза", bg="#d4a23a"); self.var_status.set("График: воспроизведение")
        else:
            if self.intensity is None: messagebox.showwarning("Пауза", "Сначала запустите камеру."); return
            self.graph_paused = True; self.btn_pause.config(text="Продолжить", bg="#3aa55a"); self.var_status.set("График: ПАУЗА")
    def toggle_camera(self):
        if self.running:
            self.running = False
            if self.camera: self.camera.close(); self.camera = None
            self.btn_cam.config(text="Пуск", bg="#3aa55a"); self.var_status.set("Камера остановлена")
        else:
            self.camera_index = int(self.var_cam_index.get()); self.camera = Camera(self.camera_index)
            self.last_sequence, self.frame_w, self.frame_h = -1, 0, 0
            self.camera.set_property(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.camera.set_property(cv2.CAP_PROP_EXPOSURE, self.var_exposure.get())
            self.camera.set_property(cv2.CAP_PROP_GAIN, self.var_gain.get())
            self.camera.set_property(cv2.CAP_PROP_AUTO_WB, 0)
            self.running, self.graph_paused = True, False
            self.btn_pause.config(text="Пауза", bg="#d4a23a"); self.btn_cam.config(text="Стоп", bg="#d54a3a")
            self.last_preview_time, self.last_plot_time = 0.0, 0.0
            self.var_status.set(f"Открытие камеры {self.camera_index}..."); self._update_loop()

    def _update_loop(self):
        if not self.running or not self.camera: return
        try:
            frame, seq, fps, error, w, h = self.camera.snapshot()
            if frame is not None and seq != self.last_sequence:
                self.last_sequence = seq
                if self.frame_w != w or self.frame_h != h:
                    self.frame_w, self.frame_h = w, h
                    self.intensity = self.peak_intensity = self.raw_spectrum = np.zeros(self.frame_w, dtype=float)
                    self._maybe_rescale_default_calibration(); self._rebuild_wavelengths(); self._rebuild_color_cache()
                disp_frame = cv2.flip(frame, 1) if self.flip_cam.get() else frame
                roi_h = max(1, min(h, int(self.var_roi_height.get())))
                center = int(h * float(self.var_roi_center.get()) / 100.0)
                top = max(0, min(h - roi_h, center - roi_h // 2)); bottom = min(h, top + roi_h)
                strip_raw = frame[top:bottom, 0:self.frame_w]
                now = time.monotonic()
                if now - self.last_preview_time >= 1 / 20:
                    self._draw_preview(disp_frame, top, bottom)
                    self._show_strip(strip_raw if self.flip_spec.get() else cv2.flip(strip_raw, 1))
                    self.last_preview_time = now
                if not self.graph_paused:
                    self._process_spectrum(strip_raw)
                    if now - self.last_plot_time >= 1 / 10: self._draw_graph(); self.last_plot_time = now
                if fps > 0: self.var_fps.set(f"FPS: {fps:.0f}")
                if error: self.var_status.set(error)
        except Exception as exc: self.var_status.set(f"Ошибка: {exc}")
        if self.running: self.root.after(20, self._update_loop)

    def _compute_signal(self, strip):
        bgr = strip.astype(np.float32).mean(axis=0); sel = self.var_channel.get()
        if sel == "R": return bgr[:, 2]
        if sel == "G": return bgr[:, 1]
        if sel == "B": return bgr[:, 0]
        if sel == "Среднее RGB": return bgr.mean(axis=1)
        return 0.114 * bgr[:, 0] + 0.587 * bgr[:, 1] + 0.299 * bgr[:, 2]
    def _process_spectrum(self, strip):
        signal = self._compute_signal(strip)
        if len(signal) != self.frame_w: self.frame_w = len(signal); self._rebuild_wavelengths(); self._rebuild_color_cache()
        self.raw_spectrum = signal.copy()
        if self.var_holdpeaks.get():
            if self.peak_intensity is not None and len(self.peak_intensity) == len(signal):
                self.peak_intensity = np.maximum(self.peak_intensity, signal); signal = self.peak_intensity.copy()
            else: self.peak_intensity = signal.copy()
        if self.var_dark.get() and self.dark_spectrum is not None and len(self.dark_spectrum) == len(signal):
            signal = np.maximum(signal - self.dark_spectrum, 0.0)
        signal = self._apply_irradiance(signal)
        w = int(self.var_smooth.get())
        if w >= 3: signal = smooth_data(signal, w if w % 2 == 1 else w + 1)
        self.intensity = signal
    def _apply_irradiance(self, y):
        if not self.var_irradiance.get() or self.irradiance_wl is None or self.wavelengths is None or len(y) != len(self.wavelengths): return y
        return y * np.interp(self.wavelengths, self.irradiance_wl, self.irradiance_coef, left=float(self.irradiance_coef[0]), right=float(self.irradiance_coef[-1]))

    def _show_ppm(self, label, rgb_array):
        h, w = rgb_array.shape[:2]
        img = tk.PhotoImage(data=f"P6 {w} {h} 255 ".encode("ascii") + np.ascontiguousarray(rgb_array).tobytes(), format="PPM")
        label.configure(image=img); label.image = img
    def _draw_preview(self, frame, top, bottom):
        h, w = frame.shape[:2]; disp_w = 400; scale = disp_w / w; disp_h = int(h * scale)
        preview = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        y1, y2 = max(0, min(disp_h-1, int(top*scale))), max(0, min(disp_h-1, int(bottom*scale)))
        cv2.line(preview, (0, y1), (disp_w-1, y1), (0, 255, 0), 1); cv2.line(preview, (0, y2), (disp_w-1, y2), (255, 255, 0), 1)
        self._show_ppm(self.video_label, cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    def _show_strip(self, strip):
        if strip is None or strip.size == 0: return
        w = self.strip_label.winfo_width(); w = 400 if w < 50 else w
        self._show_ppm(self.strip_label, cv2.resize(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB), (w, 50), interpolation=cv2.INTER_AREA))

    def save_dark(self):
        if self.intensity is None: messagebox.showwarning("Тёмный спектр", "Сначала запустите камеру."); return
        self.dark_spectrum = self.intensity.copy(); self.var_dark.set(True); self.var_status.set("Тёмный спектр сохранён")
    def load_irradiance_file(self):
        path = filedialog.askopenfilename(title="Файл коррекции", filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")])
        if not path: return
        wl, co = load_theremino_corrections(path)
        if wl is None: messagebox.showerror("Коррекция", "Не удалось распознать файл."); return
        self.irradiance_wl, self.irradiance_coef, self.irradiance_name = wl, co, os.path.basename(path)
        self.var_irradiance.set(True); self.var_corr_label.set(f"Коррекция: {self.irradiance_name} ({len(wl)} тчк)")
        self.var_status.set(f"Загружена коррекция: {self.irradiance_name}")

    def _setup_default_calibration(self):
        self.calib_px, self.calib_nm, self._default_calib_active = [0, 959, 1919], [380, 545, 750], True
    def _maybe_rescale_default_calibration(self):
        if self._default_calib_active and self.frame_w >= 2: self.calib_px = [0, self.frame_w // 2, self.frame_w - 1]
    def _load_calibration_file(self):
        if not os.path.exists(self.calib_file): return
        try:
            with open(self.calib_file, "r") as f:
                lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    px = [int(float(x)) for x in lines[0].split(",")]; nm = [float(x) for x in lines[1].split(",")]
                    if len(px) == len(nm) and len(px) >= 2:
                        self.calib_px, self.calib_nm, self._default_calib_active = px, nm, False
                        self.var_calib.set(f"Калибровка: {len(px)} точек (файл)")
        except Exception: pass
    def _load_calibration_json(self):
        if not CONFIG_PATH.exists(): return
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8")); pts = data.get("points")
            if pts and len(pts) >= 2:
                self.calib_px = [int(p) for p, _ in pts]; self.calib_nm = [float(w) for _, w in pts]
                self._default_calib_active = False; self.var_calib.set(f"Калибровка: {len(pts)} точек (JSON)")
        except Exception: pass
    def _rebuild_wavelengths(self):
        if self.frame_w == 0 or len(self.calib_px) < 2: return
        px, nm = np.array(self.calib_px, dtype=float), np.array(self.calib_nm, dtype=float)
        mask = (px >= 0) & (px <= self.frame_w - 1)
        if mask.sum() < 2: mask = np.ones_like(px, dtype=bool)
        px, nm = px[mask][np.argsort(px[mask])], nm[mask][np.argsort(px[mask])]
        self.wavelengths = np.polyval(np.polyfit(px, nm, min(len(px)-1, 3)), np.arange(self.frame_w, dtype=float))
    def _rebuild_color_cache(self):
        if self.wavelengths is not None and self.frame_w > 0: self._cached_colors_float = wavelengths_to_rgb_batch(self.wavelengths)
    def _save_calibration(self):
        try:
            with open(self.calib_file, "w") as f:
                f.write(",".join(str(int(p)) for p in self.calib_px) + "\n")
                f.write(",".join(str(float(n)) for n in self.calib_nm) + "\n")
            data = {"width": self.frame_w, "points": [[float(p), float(w)] for p, w in zip(self.calib_px, self.calib_nm)]}
            tmp = CONFIG_PATH.with_suffix(".tmp"); tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(CONFIG_PATH)
        except Exception: pass

    # ================= ПОЛНЫЙ МЕТОД КАЛИБРОВКИ =================
    def open_calibration(self):
        if self.raw_spectrum is None: messagebox.showwarning("Калибровка", "Сначала запустите камеру."); return
        signal = self.raw_spectrum.copy(); width = len(signal)
        points = [(float(p), float(w)) for p, w in zip(self.calib_px, self.calib_nm) if 0 <= p <= width - 1]
        win = tk.Toplevel(self.root); win.title("Калибровка по известным линиям"); win.geometry("1000x900")
        win.configure(bg="#2a2a2a"); win.transient(self.root); win.grab_set()
        tk.Label(win, text="1. Выберите лампу.\n2. Кликайте по пикам (слева направо).\n3. Стрелки ← → сдвигают выбор.", fg="#ccc", bg="#2a2a2a", justify="left").pack(pady=6, padx=10, anchor="w")
        lamp_row = tk.Frame(win, bg="#2a2a2a"); lamp_row.pack(fill="x", padx=10, pady=2)
        tk.Label(lamp_row, text="Лампа:", fg="#ccc", bg="#2a2a2a").pack(side="left", padx=5)
        lamp_var = tk.StringVar(value="CFL / ЛДС")
        ttk.Combobox(lamp_row, textvariable=lamp_var, values=list(LAMP_PRESETS.keys()), state="readonly", width=20).pack(side="left", padx=5)
        
        fig = Figure(figsize=(9, 3), dpi=100, facecolor="white"); ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.20, left=0.08, right=0.98)
        ax.plot(np.arange(width), signal, linewidth=1); ax.set_xlabel("Исходный пиксель"); ax.set_ylabel("Интенсивность"); ax.grid(True, alpha=0.3)
        marker = ax.axvline(0, color="red", alpha=0.6)
        if self.flip_spec.get(): ax.invert_xaxis()
        canvas = FigureCanvasTkAgg(fig, master=win); canvas.get_tk_widget().pack(fill="both", expand=True)
        
        ZOOM_HALF = 25
        mag_fig = Figure(figsize=(9, 1.8), dpi=100, facecolor="white"); mag_ax = mag_fig.add_subplot(111)
        mag_fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.30)
        mag_line, = mag_ax.plot([], [], linewidth=1.4, color="#1685d1")
        mag_point, = mag_ax.plot([], [], marker="o", markersize=5, color="red", zorder=5)
        mag_vline = mag_ax.axvline(0, color="red", alpha=0.45, linewidth=1.0)
        mag_ax.set_xlabel("Пиксель (лупа)"); mag_ax.set_ylabel("Инт."); mag_ax.grid(True, alpha=0.3)
        mag_info = mag_ax.text(0.99, 0.95, "", transform=mag_ax.transAxes, ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        mag_canvas = FigureCanvasTkAgg(mag_fig, master=win); mag_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        fields = tk.Frame(win, bg="#2a2a2a"); fields.pack(fill="x", padx=10, pady=4)
        pixel_var = tk.StringVar(value="0"); wave_var = tk.StringVar()
        tk.Label(fields, text="Пиксель:", fg="#ccc", bg="#2a2a2a").pack(side="left")
        tk.Entry(fields, textvariable=pixel_var, width=12, bg="#1e1e1e", fg="white").pack(side="left", padx=5)
        tk.Label(fields, text="Длина волны, нм:", fg="#ccc", bg="#2a2a2a").pack(side="left")
        wave_entry = tk.Entry(fields, textvariable=wave_var, width=12, bg="#1e1e1e", fg="white"); wave_entry.pack(side="left", padx=5)
        
        table = ttk.Treeview(win, columns=("pixel", "wave"), show="headings", height=5)
        table.heading("pixel", text="Пиксель"); table.heading("wave", text="Длина волны, нм"); table.pack(fill="x", padx=8, pady=5)
        
        def value_at(x):
            if x <= 0.0: return float(signal[0])
            if x >= width - 1: return float(signal[-1])
            xi = int(np.floor(x)); t = x - xi
            return float(signal[xi] * (1.0 - t) + signal[xi + 1] * t)
        def update_magnifier(pixel):
            lo, hi = max(0, int(np.floor(pixel - ZOOM_HALF))), min(width - 1, int(np.ceil(pixel + ZOOM_HALF)))
            if hi <= lo: return
            xs, ys = np.arange(lo, hi + 1), signal[lo:hi + 1]
            mag_line.set_data(xs, ys); mag_ax.set_xlim(lo - 0.5, hi + 0.5)
            ymin, ymax = float(np.min(ys)), float(np.max(ys)); pad = max(2.0, (ymax - ymin) * 0.15)
            mag_ax.set_ylim(ymin - pad, ymax + pad); mag_vline.set_xdata([pixel, pixel])
            mag_point.set_data([pixel], [value_at(pixel)]); mag_info.set_text(f"x = {pixel:.3f} px\nI = {value_at(pixel):.2f}")
            mag_canvas.draw_idle()
        def set_pixel(pixel): pixel_var.set(f"{max(0.0, min(width - 1.0, float(pixel))):.3f}")
        def on_pixel_change(*_):
            try: pixel = float(pixel_var.get().replace(",", "."))
            except ValueError: return
            if not np.isfinite(pixel): return
            pixel = max(0.0, min(width - 1.0, pixel)); marker.set_xdata([pixel, pixel]); canvas.draw_idle(); update_magnifier(pixel)
        pixel_var.trace_add("write", on_pixel_change)
        def refresh_points():
            for item in table.get_children(): table.delete(item)
            for i, (px, wv) in enumerate(points): table.insert("", "end", iid=str(i), values=(f"{px:.3f}", f"{wv:.3f}"))
        def click(event):
            if event.inaxes is not ax or event.xdata is None: return
            px = float(event.xdata); set_pixel(px)
            lines = LAMP_PRESETS.get(lamp_var.get(), [])
            if lines and len(points) < len(lines):
                wv = lines[len(points)]; lo, hi = max(1, int(px) - 15), min(width - 2, int(px) + 15)
                if hi > lo:
                    seg = signal[lo:hi + 1]; loc = int(np.argmax(seg)) + lo; points.append((float(loc), float(wv)))
                else: points.append((px, float(wv)))
                points.sort(); refresh_points()
        def mag_click(event):
            if event.inaxes is not mag_ax or event.xdata is None: return
            set_pixel(event.xdata); wave_entry.focus_set()
        canvas.mpl_connect("button_press_event", click); mag_canvas.mpl_connect("button_press_event", mag_click)
        def current_pixel():
            try: value = float(pixel_var.get().replace(",", "."))
            except ValueError: return None
            return max(0.0, min(width - 1.0, value)) if np.isfinite(value) else None
        def on_key(event):
            if isinstance(event.widget, (ttk.Entry, tk.Entry)): return
            step = 0
            if event.keysym == "Left": step = -1.0
            elif event.keysym == "Right": step = 1.0
            else: return
            if event.state & 0x0004: step *= 10.0
            elif event.state & 0x0001: step *= 0.1
            px = current_pixel()
            if px is not None: set_pixel(px + step)
        win.bind("<Left>", on_key); win.bind("<Right>", on_key); win.bind("<Shift-Left>", on_key); win.bind("<Shift-Right>", on_key); win.bind("<Control-Left>", on_key); win.bind("<Control-Right>", on_key)
        def snap_to_peak():
            px = current_pixel()
            if px is None: return
            lo, hi = max(1, int(np.floor(px - ZOOM_HALF))), min(width - 2, int(np.ceil(px + ZOOM_HALF)))
            if hi <= lo: return
            peaks = [i for i in range(lo, hi + 1) if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]]
            if peaks: set_pixel(float(max(peaks, key=lambda i: (signal[i], -abs(i - px)))))
        tk.Button(fields, text="К пику", command=snap_to_peak, bg="#555555", fg="white", relief="flat", padx=8, pady=3).pack(side="left", padx=5)
        def add_point():
            try:
                px = float(pixel_var.get().replace(",", ".")); wv = float(wave_var.get().replace(",", "."))
                if not np.isfinite(px) or not np.isfinite(wv) or not 0 <= px <= width - 1 or wv <= 0: raise ValueError
            except ValueError: messagebox.showerror("Калибровка", "Введите корректный пиксель и длину волны (нм).", parent=win); return
            for i, (pp, _) in enumerate(points):
                if abs(pp - px) < 0.01: points[i] = (px, wv); break
            else: points.append((px, wv))
            points.sort(); refresh_points(); wave_var.set("")
        tk.Button(fields, text="Добавить", command=add_point, bg="#3aa55a", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=8, pady=3).pack(side="left", padx=5)
        btn_row2 = tk.Frame(win, bg="#2a2a2a"); btn_row2.pack(fill="x", padx=10, pady=4)
        def remove_points():
            for i in sorted((int(i) for i in table.selection()), reverse=True): points.pop(i)
            refresh_points()
        def clear_all(): points.clear(); refresh_points()
        tk.Button(btn_row2, text="Удалить выбранные", command=remove_points, bg="#555555", fg="white", relief="flat", padx=10, pady=4).pack(side="left", padx=3)
        tk.Button(btn_row2, text="Очистить", command=clear_all, bg="#555555", fg="white", relief="flat", padx=10, pady=4).pack(side="left", padx=3)
        def reset_calibration():
            if width < 2:
                messagebox.showwarning(
                    "Сброс калибровки",
                    "Недостаточная ширина спектра для сброса калибровки.",
                    parent=win
                )
                return

            if not messagebox.askyesno(
                "Сброс калибровки",
                "Сбросить калибровку к значениям по умолчанию?\n"
                "Текущие точки будут заменены.",
                parent=win
            ):
                return

            points.clear()

            if width == 2:
                points.extend([
                    (0.0, 380.0),
                    (1.0, 750.0),
                ])
            else:
                points.extend([
                    (0.0, 380.0),
                    (float(width // 2), 545.0),
                    (float(width - 1), 750.0),
                ])

            points.sort()
            refresh_points()

            # Сразу применяем сброс к основному приложению
            self.calib_px = [int(round(p)) for p, _ in points]
            self.calib_nm = [float(w) for _, w in points]
            self._default_calib_active = True

            self._rebuild_wavelengths()
            self._rebuild_color_cache()
            self._save_calibration()

            self.var_calib.set("Калибровка: значения по умолчанию")

            if self.intensity is not None:
                self._draw_graph()

        tk.Button(
            btn_row2,
            text="Сброс калибровки",
            command=reset_calibration,
            bg="#8a2a2a",
            fg="white",
            relief="flat",
            padx=10,
            pady=4
        ).pack(side="left", padx=3)
        def apply_calibration():
            if len(points) < 2: messagebox.showerror("Калибровка", "Добавьте минимум две известные линии.", parent=win); return
            px_arr = np.array([p for p, _ in points], dtype=float); nm_arr = np.array([w for _, w in points], dtype=float)
            if np.ptp(px_arr) < 1.0: messagebox.showerror("Калибровка", "Точки расположены слишком близко.", parent=win); return
            deg = min(len(px_arr) - 1, 3); coeffs = np.polyfit(px_arr, nm_arr, deg)
            predicted = np.polyval(coeffs, px_arr); rmse = float(np.sqrt(np.mean((predicted - nm_arr) ** 2)))
            r2 = 1.0 - np.sum((nm_arr - predicted) ** 2) / max(np.sum((nm_arr - nm_arr.mean()) ** 2), 1e-10)
            self.calib_px = [int(round(p)) for p in px_arr]; self.calib_nm = nm_arr.tolist()
            self._default_calib_active = False; self._rebuild_wavelengths(); self._rebuild_color_cache(); self._save_calibration()
            self.var_calib.set(f"Калибровка: {len(points)} точек, RMSE={rmse:.2f} нм")
            messagebox.showinfo("Калибровка", f"Применено {len(points)} точек.\nПолином {deg} порядка.\nRMSE = {rmse:.2f} нм\nR² = {r2:.6f}", parent=win); win.destroy()
        tk.Button(btn_row2, text="Применить", command=apply_calibration, bg="#3a7fd5", fg="white", font=("Segoe UI", 10, "bold"), relief="flat", padx=20, pady=5).pack(side="right", padx=3)
        refresh_points(); on_pixel_change(); canvas.draw_idle(); mag_canvas.draw_idle()

    # ================= ПОЛНЫЙ МЕТОД НАСТРОЕК КАМЕРЫ =================
    def camera_settings(self):
        if self.camera is None: messagebox.showinfo("Настройки", "Сначала подключите камеру."); return
        win = tk.Toplevel(self.root); win.title("Настройки камеры"); win.geometry("600x540")
        win.configure(bg="#2a2a2a"); win.transient(self.root)
        tk.Label(win, text="Диапазоны зависят от драйвера.\nПустое поле — не изменять.\nЗначения — OpenCV.", fg="#ccc", bg="#2a2a2a", justify="left").pack(fill="x", padx=10, pady=8)
        form = tk.Frame(win, bg="#2a2a2a"); form.pack(fill="x", padx=10, pady=4)
        properties = [("Экспозиция", cv2.CAP_PROP_EXPOSURE), ("Усиление", cv2.CAP_PROP_GAIN), ("Яркость", cv2.CAP_PROP_BRIGHTNESS), ("Контраст", cv2.CAP_PROP_CONTRAST), ("Насыщенность", cv2.CAP_PROP_SATURATION), ("Гамма", cv2.CAP_PROP_GAMMA), ("Резкость", cv2.CAP_PROP_SHARPNESS), ("Баланс белого", cv2.CAP_PROP_WB_TEMPERATURE)]
        exposure_choices = {"Ручная: 0.25 (V4L2)": 0.25, "Авто: 0.75 (V4L2)": 0.75, "Ручная: 0": 0.0, "Авто: 1": 1.0}
        props_to_read = [p for _, p in properties] + [cv2.CAP_PROP_AUTO_EXPOSURE, cv2.CAP_PROP_AUTO_WB]
        values = self.camera.get_properties(props_to_read); value_map = dict(zip(props_to_read, values))
        entries = []
        for row, (name, prop) in enumerate(properties):
            tk.Label(form, text=name, fg="#ccc", bg="#2a2a2a").grid(row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(); cur = value_map.get(prop)
            if cur is not None and np.isfinite(cur): var.set(f"{cur:g}")
            tk.Entry(form, textvariable=var, width=18, bg="#1e1e1e", fg="white").grid(row=row, column=1, sticky="ew", padx=10, pady=3)
            entries.append((name, prop, var))
        auto_exp = tk.StringVar(value="Не изменять"); cur_exp = value_map.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        if cur_exp is not None:
            for label, value in exposure_choices.items():
                if abs(cur_exp - value) < 0.05: auto_exp.set(label); break
        auto_wb = tk.StringVar(value="Не изменять"); cur_wb = value_map.get(cv2.CAP_PROP_AUTO_WB)
        if cur_wb is not None:
            if abs(cur_wb - 1.0) < 0.05: auto_wb.set("Включить")
            elif abs(cur_wb - 0.0) < 0.05: auto_wb.set("Выключить")
        row = len(properties)
        tk.Label(form, text="Автоэкспозиция", fg="#ccc", bg="#2a2a2a").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(form, textvariable=auto_exp, state="readonly", values=["Не изменять"] + list(exposure_choices.keys()), width=28).grid(row=row, column=1, sticky="ew", padx=10, pady=3)
        row += 1
        tk.Label(form, text="Автобаланс белого", fg="#ccc", bg="#2a2a2a").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(form, textvariable=auto_wb, state="readonly", values=["Не изменять", "Выключить", "Включить"], width=28).grid(row=row, column=1, sticky="ew", padx=10, pady=3)
        def apply_settings():
            if self.camera is None: messagebox.showerror("Настройки", "Камера не подключена.", parent=win); return
            commands = []; mode = auto_exp.get()
            if mode in exposure_choices: commands.append((cv2.CAP_PROP_AUTO_EXPOSURE, exposure_choices[mode]))
            wbm = auto_wb.get()
            if wbm != "Не изменять": commands.append((cv2.CAP_PROP_AUTO_WB, 1.0 if wbm == "Включить" else 0.0))
            for name, prop, var in entries:
                text = var.get().strip()
                if not text: continue
                try:
                    value = float(text.replace(",", "."))
                    if not np.isfinite(value): raise ValueError
                except ValueError: messagebox.showerror("Настройки", f"Некорректное значение поля «{name}».", parent=win); return
                commands.append((prop, value))
            for prop, value in commands: self.camera.set_property(prop, value)
            messagebox.showinfo("Настройки", "Команды отправлены камере.", parent=win)
        btn = tk.Frame(win, bg="#2a2a2a"); btn.pack(fill="x", pady=8, padx=10)
        tk.Button(btn, text="Применить", command=apply_settings, bg="#3a7fd5", fg="white", font=("Segoe UI", 9, "bold"), relief="flat", padx=14, pady=5).pack(side="right", padx=4)
        tk.Button(btn, text="Закрыть", command=win.destroy, bg="#555555", fg="white", relief="flat", padx=14, pady=5).pack(side="right", padx=4)

        # ================= ВЫДЕЛЕНИЕ ОБЛАСТИ СПЕКТРА =================
    def _toggle_select_mode(self):
        self.var_select_mode.set(not self.var_select_mode.get())
        if self.var_select_mode.get():
            self.var_status.set("Режим выделения: ВКЛ — ЛКМ + перетаскивание на графике")
            try: self.canvas.get_tk_widget().config(cursor="crosshair")
            except Exception: pass
        else:
            self.var_status.set("Режим выделения: выкл")
            try: self.canvas.get_tk_widget().config(cursor="")
            except Exception: pass

    def _clear_selection(self):
        self.selection_range = None
        if self.sel_rect is not None:
            try: self.sel_rect.remove()
            except Exception: pass
            self.sel_rect = None
        self.var_sel_label.set("Область: не выбрана")
        self.canvas.draw_idle()

    def _zoom_to_selection(self):
        if not self.selection_range:
            messagebox.showwarning("Выделение", "Сначала выделите область на графике.")
            return
        self.zoom_range = self.selection_range
        self._draw_graph()

    def _reset_zoom(self):
        self.zoom_range = None
        self._draw_graph()

    def _on_graph_press(self, event):
        if not self.var_select_mode.get():
            return
        if event.inaxes is not self.ax or event.xdata is None or event.button != 1:
            return
        self.sel_start_x = float(event.xdata)
        self.sel_dragging = True
        if self.sel_rect is None:
            from matplotlib.patches import Rectangle
            self.sel_rect = Rectangle((0, 0), 0, 0,
                                      facecolor="yellow", alpha=0.18,
                                      edgecolor="red", linewidth=1, zorder=1)
            self.ax.add_patch(self.sel_rect)
        ymax = max(1.0, float(self.ax.get_ylim()[1]))
        self.sel_rect.set_xy((self.sel_start_x, 0))
        self.sel_rect.set_width(0)
        self.sel_rect.set_height(ymax)
        self.canvas.draw_idle()

    def _on_graph_motion(self, event):
        if not self.sel_dragging or self.sel_rect is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        x0, x1 = self.sel_start_x, float(event.xdata)
        lo, hi = min(x0, x1), max(x0, x1)
        self.sel_rect.set_x(lo)
        self.sel_rect.set_width(hi - lo)
        self.canvas.draw_idle()

    def _on_graph_release(self, event):
        if not self.sel_dragging:
            return
        self.sel_dragging = False
        if event.inaxes != self.ax or event.xdata is None:
            return
        x0, x1 = self.sel_start_x, float(event.xdata)
        lo, hi = min(x0, x1), max(x0, x1)
        if hi - lo < 0.5:
            self._clear_selection()
            return
        self.selection_range = (lo, hi)
        self.var_sel_label.set(f"Область: {lo:.1f}–{hi:.1f} нм")
        self.var_status.set(f"Выделено: {lo:.1f} – {hi:.1f} нм (ширина {hi - lo:.1f} нм)")
        self.canvas.draw_idle()

    # ================= ОПТИМИЗИРОВАННЫЙ ГРАФИК =================
    def _draw_graph(self):
        if self.intensity is None or self.wavelengths is None: return
        x, y = self.wavelengths, self.intensity
        if self.var_logscale.get():
            y_plot = np.log10(np.maximum(y, 0) + 1.0); ymax = max(2.0, float(y_plot.max()) * 1.1); ylabel = "log₁₀ (Инт. + 1)"
        else:
            y_plot = y; ymax = max(255.0, float(y_plot.max()) * 1.1); ylabel = "Интенсивность"
        self.ax.set_ylabel(ylabel)
        cur_top = self.ax.get_ylim()[1]
        if ymax > cur_top or ymax < cur_top * 0.5: self.ax.set_ylim(0, ymax)
        if self.sel_rect is not None:
            self.sel_rect.set_height(max(1.0, float(self.ax.get_ylim()[1])))
        if self.fill_coll is not None:
            try: self.fill_coll.remove()
            except Exception: pass
            self.fill_coll = None
        for t in self.peak_texts:
            try: t.remove()
            except Exception: pass
        self.peak_texts = []
        self.line_main.set_data(x, y_plot)
        if self._cached_colors_float is not None and len(x) > 1:
            n = len(x); segs = np.zeros((n - 1, 2, 2))
            segs[:, 0, 0] = x[:-1]; segs[:, 1, 0] = x[:-1]; segs[:, 0, 1] = 0; segs[:, 1, 1] = y_plot[:-1]
            try:
                self.fill_coll = LineCollection(segs, colors=self._cached_colors_float[:n - 1], linewidths=1.5, zorder=2)
                self.ax.add_collection(self.fill_coll)
            except Exception: self.fill_coll = None
        
        peaks = find_peaks(y, thresh_rel=self.var_thresh.get() / 100.0,
                           min_dist=self.var_mindist.get())
        if (self.var_only_sel_peaks.get() and self.selection_range is not None
                and len(peaks) > 0):
            wl_peaks = x[peaks]
            keep = (wl_peaks >= self.selection_range[0]) & (wl_peaks <= self.selection_range[1])
            peaks = peaks[keep]
        if len(peaks) > 0:
            if self.peak_line is None:
                self.peak_line, = self.ax.plot([], [], 'rv', markersize=6, zorder=10)
            self.peak_line.set_data(x[peaks], y_plot[peaks])
            for pidx in peaks:
                wl, val_plot, val_raw = float(x[pidx]), float(y_plot[pidx]), float(y[pidx])
                ann = self.ax.annotate(f"{wl:.1f} ({val_raw:.0f})", (wl, val_plot),
                                       textcoords="offset points", xytext=(0, 8),
                                       fontsize=7, color="red", ha="center",
                                       fontweight="bold")
                self.peak_texts.append(ann)
        else:
            if self.peak_line is not None:
                self.peak_line.set_data([], [])

        if self.zoom_range is not None:
            x_lo, x_hi = self.zoom_range
        else:
            x_lo, x_hi = 250.0, 780.0
        if self.flip_spec.get():
            self.ax.set_xlim(x_hi, x_lo)
        else:
            self.ax.set_xlim(x_lo, x_hi)
        self.canvas.draw_idle()

    def save_png(self):
        if self.intensity is None: messagebox.showwarning("Сохранение", "Нет данных."); return
        fname = f"spectrum_{datetime.now():%Y%m%d_%H%M%S}.png"
        try: self.fig.savefig(fname, dpi=150, bbox_inches="tight")
        except Exception as exc: messagebox.showerror("Сохранение", f"Ошибка PNG:\n{exc}"); return
        self.var_status.set(f"Сохранено: {fname}")
    def save_csv(self):
        if self.raw_spectrum is None or self.wavelengths is None: messagebox.showwarning("Сохранение", "Нет данных."); return
        signal = self.intensity.copy() if self.intensity is not None else self.raw_spectrum.copy()
        filename = filedialog.asksaveasfilename(parent=self.root, title="Сохранить спектр", defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile=f"spectrum_{datetime.now():%Y%m%d_%H%M%S}.csv")
        if not filename: return
        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";"); writer.writerow(["pixel", "wavelength_nm", "intensity_counts", "channel"])
                for pixel in range(len(self.wavelengths)): writer.writerow([pixel, f"{float(self.wavelengths[pixel]):.8f}", f"{float(signal[pixel]):.6f}", self.var_channel.get()])
        except Exception as exc: messagebox.showerror("Сохранение", f"Ошибка CSV:\n{exc}"); return
        self.var_status.set(f"Сохранено: {filename}")
    def on_close(self):
        self.running = False
        if self.camera:
            try: self.camera.close()
            except Exception: pass
            self.camera = None
        self.root.after(100, self.root.destroy)

def main():
    root = tk.Tk(); SpectrometerApp(root); root.mainloop()

if __name__ == "__main__":
    main()