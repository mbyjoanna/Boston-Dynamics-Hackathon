"""
Smart Motor (PyScript) — main.py

Same teachable sensor-to-motor idea as smart_motor_web.py, but running
entirely in the browser tab via PyScript instead of a local Python HTTP
server. There is no fetch()/API layer anymore: this file owns the model,
the LEGO BLE connection (through Device.Element, using the same worker
patch as the Neural Network Builder), and the DOM/canvas rendering, all in
one continuous asyncio loop.
"""
import asyncio
import json
import math
from pyscript import document, window
from pyscript.ffi import create_proxy
try:
    from pyscript.ffi import to_js
except ImportError:
    from pyodide.ffi import to_js
from Device import Element

# ── Model: nearest-neighbor mapping from a sensor reading vector to positions ──
DEFAULT_SPEED = 50  # percent, 1-100

class NearestNeighborModel:
    """Stores (sensor_vector, [positions...], [speeds...]) points; predicts
    the nearest point's position(s), and a distance-weighted blend of
    speed(s) so speed varies smoothly across the input range."""

    def __init__(self):
        self._points = []
        self._features = []

    @property
    def features(self):
        return list(self._features)

    def set_features(self, features):
        features = list(features)
        if features == self._features:
            return True
        if self._points:
            return False
        self._features = features
        return True

    def add_point(self, sensor_vector, positions, speeds=None):
        vec = [float(x) for x in sensor_vector]
        pos = [float(x) for x in positions]
        spd = [float(DEFAULT_SPEED) for _ in pos] if speeds is None else [float(x) for x in speeds]
        self._points.append([vec, pos, spd])

    def clear(self):
        self._points = []

    def remove_last(self):
        if not self._points:
            return False
        self._points.pop()
        return True

    def points(self):
        return [[list(v), list(p), list(sp)] for v, p, sp in self._points]

    def __len__(self):
        return len(self._points)

    def predict(self, sensor_vector):
        if not self._points:
            return None

        def dist2(point):
            return sum((a - b) ** 2 for a, b in zip(point[0], sensor_vector))

        return list(min(self._points, key=dist2)[1])

    def predict_speed(self, sensor_vector):
        if not self._points:
            return None
        n = len(self._points[0][2])
        acc = [0.0] * n
        wsum = 0.0
        for vec, _pos, spd in self._points:
            d2 = sum((a - b) ** 2 for a, b in zip(vec, sensor_vector))
            w = 1.0 / (d2 + 1e-6)
            wsum += w
            for j in range(min(n, len(spd))):
                acc[j] += w * spd[j]
        if wsum == 0:
            return None
        return [acc[j] / wsum for j in range(n)]

    def to_json_obj(self):
        return {"features": self.features, "points": self.points()}

    def load_obj(self, data):
        pts = data.get("points", [])
        features = data.get("features", [])
        parsed = []
        for entry in pts:
            s, p = entry[0], entry[1]
            sp = entry[2] if len(entry) > 2 else None
            if isinstance(s, (int, float)):
                s = [s]
            if isinstance(p, (int, float)):
                p = [p]
            pos = [float(x) for x in p]
            if sp is None:
                spd = [float(DEFAULT_SPEED) for _ in pos]
            else:
                if isinstance(sp, (int, float)):
                    sp = [sp]
                spd = [float(x) for x in sp]
            parsed.append([[float(x) for x in s], pos, spd])
        if not features and parsed:
            features = [f"feature{i}" for i in range(len(parsed[0][0]))]
        self._features = list(features)
        self._points = parsed


# ── Backends ────────────────────────────────────────────────────────────────
SIM_FEATURES = ["valueX", "valueY"]
COLOR_FEATURES = ["Reflection", "Hue", "Color", "Value", "Saturation"]
CONTROLLER_FEATURES = ["LeftPercent", "RightPercent", "LeftAngle", "RightAngle"]
DEFAULT_FEATURES = {
    "sim": ["valueX"],
    "color": ["Reflection"],
    "controller": ["LeftPercent", "RightPercent"],
}
COLOR_NAMES = ["GREEN", "BLUE", "RED", "ORANGE", "YELLOW", "AZURE", "PURPLE", "MAGENTA"]


class SimBackend:
    name = "Simulated"

    def __init__(self):
        self.sensor_features = list(SIM_FEATURES)
        self.motor_labels = ["motor"]
        self._connected = False
        self._sensor = {"valueX": 50.0, "valueY": 50.0}
        self._motor_pos = 0.0
        self._motor_target = 0.0
        self._motor_speed = 50.0

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    def set_sim_sensor(self, feature, value):
        self._sensor[feature] = float(value)

    def set_manual_motor(self, value):
        self._motor_pos = float(value)
        self._motor_target = float(value)

    def read_sensor_vector(self, features):
        return [self._sensor.get(f, 0.0) for f in features]

    def read_motor_position(self):
        diff = self._motor_target - self._motor_pos
        frac = max(0.05, (self._motor_speed / 100.0) * 0.35)
        self._motor_pos += diff * frac
        if abs(diff) < 0.5:
            self._motor_pos = self._motor_target
        return [self._motor_pos]

    def move_motor_to(self, positions, speeds=None):
        self._motor_target = float(positions[0])
        if speeds and speeds[0] is not None:
            self._motor_speed = float(speeds[0])


class LegoBackendWrapper:
    """Owns two separate BLE connections (a motor hub and a sensor hub),
    each paired individually through the browser's device picker."""
    name = "LEGO Hardware"

    def __init__(self, sensor_kind="color", motor_kind="single"):
        self.sensor_kind = sensor_kind
        self.motor_kind = motor_kind
        self.sensor_features = COLOR_FEATURES if sensor_kind == "color" else CONTROLLER_FEATURES
        self.motor_labels = ["motor"] if motor_kind == "single" else ["left", "right"]
        self.motor_element = None
        self.sensor_element = None

    async def connect(self):
        motor_type = "Single Motor" if self.motor_kind == "single" else "Double Motor"
        sensor_type = "Color Sensor" if self.sensor_kind == "color" else "Controller"

        self.motor_element = Element()
        await self.motor_element.connect(existing_names=[])
        if not self.motor_element.connected or motor_type not in (self.motor_element.name or ""):
            got = self.motor_element.name or "nothing"
            self.motor_element = None
            raise RuntimeError(f"Expected to pair a {motor_type}, got {got}.")

        self.sensor_element = Element()
        await self.sensor_element.connect(existing_names=[self.motor_element.name])
        if not self.sensor_element.connected or sensor_type not in (self.sensor_element.name or ""):
            got = self.sensor_element.name or "nothing"
            self.sensor_element = None
            raise RuntimeError(f"Expected to pair a {sensor_type}, got {got}.")

    async def disconnect(self):
        if self.motor_element:
            await self.motor_element.disconnect()
        if self.sensor_element:
            await self.sensor_element.disconnect()
        self.motor_element = None
        self.sensor_element = None

    @property
    def connected(self):
        return bool(self.motor_element and self.motor_element.connected
                     and self.sensor_element and self.sensor_element.connected)

    def read_sensor_vector(self, features):
        if not self.sensor_element or not self.sensor_element.state:
            return [0.0] * len(features)
        return [float(self.sensor_element.state.get(f, 0.0)) for f in features]

    def read_motor_position(self):
        if not self.motor_element or not self.motor_element.state:
            return [0.0] * len(self.motor_labels)
        if self.motor_kind == "single":
            return [float(self.motor_element.state.get("Position", 0.0))]
        return [float(self.motor_element.state.get("LeftPosition", 0.0)),
                float(self.motor_element.state.get("RightPosition", 0.0))]

    def move_motor_to(self, positions, speeds=None):
        if not self.motor_element:
            return
        s0 = speeds[0] if speeds and speeds[0] is not None else DEFAULT_SPEED
        if self.motor_kind == "single":
            self.motor_element.move_to_position(positions[0], s0)
        else:
            s1 = speeds[1] if speeds and len(speeds) > 1 and speeds[1] is not None else DEFAULT_SPEED
            self.motor_element.move_to_position_double(positions[0], positions[1], s0, s1)


# ── Application state ─────────────────────────────────────────────────────────
MODE_TRAINING = "TRAINING"
MODE_RUN = "RUN"
MAP_COMBINED = "combined"
MAP_INDEPENDENT = "independent"


class SmartMotorState:
    def __init__(self):
        self.backend = SimBackend()
        self.mode = MODE_TRAINING
        self.mapping = MAP_COMBINED

        self.model = NearestNeighborModel()
        self.combined_features = []
        self.display_feature = None

        self.axis_features = []
        self.axis_models = []
        self.axis_speeds = []

        self.active_features = []

        self.latest_vec = [0.0]
        self.latest_features = []
        self.latest_pos = [0.0]
        self.latest_target = None
        self.latest_speed = None

        self._apply_defaults()

    def _backend_key(self):
        if isinstance(self.backend, SimBackend):
            return "sim"
        return self.backend.sensor_kind

    def num_axes(self):
        return len(self.backend.motor_labels)

    def supports_independent(self):
        return self.num_axes() >= 2

    def _default_axis_features(self):
        feats = self.backend.sensor_features
        return [feats[min(i, len(feats) - 1)] for i in range(self.num_axes())]

    def _apply_defaults(self):
        key = self._backend_key()
        cf = [f for f in self.backend.sensor_features if f in DEFAULT_FEATURES.get(key, [])]
        if not cf:
            cf = [self.backend.sensor_features[0]]
        self.combined_features = cf
        self.display_feature = cf[0]
        self.model = NearestNeighborModel()
        self.model.set_features(cf)

        self.axis_features = self._default_axis_features()
        self.axis_models = [NearestNeighborModel() for _ in range(self.num_axes())]
        for a, feat in enumerate(self.axis_features):
            self.axis_models[a].set_features([feat])

        self.axis_speeds = [DEFAULT_SPEED for _ in range(self.num_axes())]

        if not self.supports_independent():
            self.mapping = MAP_COMBINED
        self._sync_active_features()

    def _sync_active_features(self):
        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            seen = []
            for f in self.axis_features:
                if f not in seen:
                    seen.append(f)
            self.active_features = seen
        else:
            self.active_features = list(self.combined_features)

    def set_config(self, backend_name, motor, sensor):
        if self.backend.connected:
            raise RuntimeError("Disconnect before changing the configuration.")
        if backend_name == SimBackend.name:
            self.backend = SimBackend()
        else:
            motor_kind = "single" if motor == "Single Motor" else "double"
            sensor_kind = "color" if sensor == "Color Sensor" else "controller"
            self.backend = LegoBackendWrapper(sensor_kind=sensor_kind, motor_kind=motor_kind)
        self._apply_defaults()

    def set_mapping(self, mode):
        self.mapping = MAP_INDEPENDENT if (mode == MAP_INDEPENDENT and self.supports_independent()) else MAP_COMBINED
        self._sync_active_features()

    def set_features(self, features):
        features = [f for f in features if f in self.backend.sensor_features]
        if not features:
            features = [self.backend.sensor_features[0]]
        if not self.model.set_features(features):
            self.model.clear()
            self.model.set_features(features)
        self.combined_features = features
        if self.display_feature not in features:
            self.display_feature = features[0]
        self._sync_active_features()

    def set_axis_feature(self, axis, feature):
        if not (0 <= axis < self.num_axes()) or feature not in self.backend.sensor_features:
            return
        if self.axis_features[axis] != feature:
            self.axis_models[axis].clear()
            self.axis_models[axis].set_features([feature])
            self.axis_features[axis] = feature
        self._sync_active_features()

    def set_axis_speed(self, axis, speed):
        if not (0 <= axis < self.num_axes()):
            return
        try:
            speed = int(round(float(speed)))
        except (TypeError, ValueError):
            return
        self.axis_speeds[axis] = max(1, min(100, speed))

    def set_display_feature(self, feature):
        if feature in self.combined_features:
            self.display_feature = feature

    def set_mode(self, mode):
        self.mode = MODE_RUN if mode == MODE_RUN else MODE_TRAINING

    async def connect(self):
        await self.backend.connect()

    async def disconnect(self):
        await self.backend.disconnect()

    def record(self, axis=None):
        if not self.backend.connected:
            raise RuntimeError("Not connected.")
        vec = list(self.latest_vec)
        feats = list(self.latest_features)
        pos = list(self.latest_pos)

        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            axes = [axis] if axis is not None else range(self.num_axes())
            for a in axes:
                if not (0 <= a < self.num_axes()) or a >= len(pos):
                    continue
                feat = self.axis_features[a]
                if feat not in feats:
                    continue
                val = vec[feats.index(feat)]
                if not self.axis_models[a].set_features([feat]):
                    self.axis_models[a].clear()
                    self.axis_models[a].set_features([feat])
                self.axis_models[a].add_point([val], [pos[a]], [self.axis_speeds[a]])
        else:
            if feats and not self.model.set_features(feats):
                self.model.clear()
                self.model.set_features(feats)
            self.model.add_point(vec, pos, list(self.axis_speeds))

    def clear_points(self, axis=None):
        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            if axis is None:
                for m in self.axis_models:
                    m.clear()
            elif 0 <= axis < self.num_axes():
                self.axis_models[axis].clear()
        else:
            self.model.clear()

    def undo_last(self, axis=None):
        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            if axis is None:
                for m in self.axis_models:
                    m.remove_last()
            elif 0 <= axis < self.num_axes():
                self.axis_models[axis].remove_last()
        else:
            self.model.remove_last()

    def sim_sensor(self, feature, value):
        if isinstance(self.backend, SimBackend):
            self.backend.set_sim_sensor(feature, value)

    def sim_motor(self, value):
        if isinstance(self.backend, SimBackend) and self.mode == MODE_TRAINING:
            self.backend.set_manual_motor(value)

    def export_json(self):
        return json.dumps({
            "mapping": self.mapping,
            "combined": self.model.to_json_obj(),
            "display": self.display_feature,
            "axes": [{"feature": self.axis_features[a],
                      "speed": self.axis_speeds[a],
                      "model": self.axis_models[a].to_json_obj()}
                     for a in range(self.num_axes())],
        }, indent=2)

    def import_json(self, text):
        data = json.loads(text)
        if "combined" not in data and "points" in data:
            self.mapping = MAP_COMBINED
            self.model.load_obj(data)
            loaded = self.model.features
            feats = [f for f in self.backend.sensor_features if f in loaded]
            if feats:
                self.combined_features = feats
                self.display_feature = feats[0]
            self._sync_active_features()
            return

        self.mapping = data.get("mapping", MAP_COMBINED)
        if self.mapping == MAP_INDEPENDENT and not self.supports_independent():
            self.mapping = MAP_COMBINED

        self.model.load_obj(data.get("combined", {"features": [], "points": []}))
        cfeats = [f for f in self.backend.sensor_features if f in self.model.features]
        if cfeats:
            self.combined_features = cfeats
        disp = data.get("display")
        self.display_feature = disp if disp in self.combined_features else self.combined_features[0]

        axes = data.get("axes", [])
        for a in range(self.num_axes()):
            if a < len(axes):
                feat = axes[a].get("feature")
                if feat in self.backend.sensor_features:
                    self.axis_features[a] = feat
                spd = axes[a].get("speed")
                if spd is not None:
                    try:
                        self.axis_speeds[a] = max(1, min(100, int(round(float(spd)))))
                    except (TypeError, ValueError):
                        pass
                self.axis_models[a].load_obj(axes[a].get("model", {"features": [], "points": []}))
        self._sync_active_features()

    def _independent_targets(self, features, vec, pos):
        target, move, speeds, any_pred = [], [], [], False
        for a in range(self.num_axes()):
            feat = self.axis_features[a]
            pred = None
            spd = self.axis_speeds[a]
            if feat in features:
                val = vec[features.index(feat)]
                r = self.axis_models[a].predict([val])
                if r is not None:
                    pred = r[0]
                    blended = self.axis_models[a].predict_speed([val])
                    if blended:
                        spd = blended[0]
            target.append(pred)
            move.append(pred if pred is not None else (pos[a] if a < len(pos) else 0.0))
            speeds.append(spd)
            if pred is not None:
                any_pred = True
        return target, move, speeds, any_pred

    def tick(self):
        """One control-loop pass: read sensors/motor, and in RUN mode,
        predict + command the motor. Called every ~100ms by main_loop()."""
        if not self.backend.connected:
            return
        features = self.active_features or self.backend.sensor_features[:1]
        vec = self.backend.read_sensor_vector(features)
        pos = self.backend.read_motor_position()
        target = None
        speed = None
        if self.mode == MODE_RUN:
            if self.mapping == MAP_INDEPENDENT and self.supports_independent():
                target, move, speeds, any_pred = self._independent_targets(features, vec, pos)
                if any_pred:
                    self.backend.move_motor_to(move, speeds)
                    speed = speeds
            else:
                target = self.model.predict(vec)
                if target is not None:
                    speeds = self.model.predict_speed(vec) or list(self.axis_speeds)
                    self.backend.move_motor_to(target, speeds)
                    speed = speeds
        self.latest_vec = vec
        self.latest_features = features
        self.latest_pos = pos
        self.latest_target = target
        self.latest_speed = speed

    def _build_plot(self, vec, feats, pos, target, connected):
        labels = self.backend.motor_labels
        series = []
        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            xlabel = "input value"
            for a, lab in enumerate(labels):
                feat = self.axis_features[a]
                pts = self.axis_models[a].points()
                spoints = [[p[0][0], p[1][0]] for p in pts if p[0] and p[1]]
                live, ty = None, None
                if connected and feat in feats:
                    lx = vec[feats.index(feat)]
                    if a < len(pos):
                        live = [lx, pos[a]]
                    if target is not None and a < len(target) and target[a] is not None:
                        ty = target[a]
                series.append({"label": f"{lab} ← {feat}", "points": spoints, "live": live, "targetY": ty})
        else:
            disp = self.display_feature
            mfeats = self.model.features
            di = mfeats.index(disp) if disp in mfeats else 0
            pts = self.model.points()
            xlabel = disp or ""
            for a, lab in enumerate(labels):
                spoints = [[p[0][di], p[1][a]] for p in pts if len(p[0]) > di and len(p[1]) > a]
                live, ty = None, None
                if connected and disp in feats:
                    lx = vec[feats.index(disp)]
                    if a < len(pos):
                        live = [lx, pos[a]]
                    if target is not None and a < len(target) and target[a] is not None:
                        ty = target[a]
                series.append({"label": lab, "points": spoints, "live": live, "targetY": ty})
        return {"xlabel": xlabel, "series": series}

    def points_label(self):
        if self.mapping == MAP_INDEPENDENT and self.supports_independent():
            parts = [f"{self.backend.motor_labels[a]}={len(self.axis_models[a])}" for a in range(self.num_axes())]
            return "Recorded points — " + ", ".join(parts)
        return f"Recorded points: {len(self.model)}"

    def snapshot(self):
        vec, feats, pos = list(self.latest_vec), list(self.latest_features), list(self.latest_pos)
        target = list(self.latest_target) if self.latest_target is not None else None
        speed = list(self.latest_speed) if self.latest_speed is not None else None
        connected = self.backend.connected
        readings = {f: v for f, v in zip(feats, vec)} if connected else {}
        return {
            "backend": self.backend.name,
            "isSim": isinstance(self.backend, SimBackend),
            "connected": connected,
            "mode": self.mode,
            "mapping": self.mapping,
            "numAxes": self.num_axes(),
            "motorLabels": list(self.backend.motor_labels),
            "availableFeatures": list(self.backend.sensor_features),
            "selectedFeatures": list(self.combined_features),
            "displayFeature": self.display_feature,
            "axisFeatures": list(self.axis_features),
            "axisSpeeds": list(self.axis_speeds),
            "readings": readings,
            "positions": pos if connected else [],
            "target": target if connected else None,
            "speed": speed if connected else None,
            "plot": self._build_plot(vec, feats, pos, target, connected),
            "pointsLabel": self.points_label(),
        }


STATE = SmartMotorState()

# ── DOM helpers ────────────────────────────────────────────────────────────
def el(id_):
    return document.getElementById(id_)

def set_children_html(container, html):
    container.innerHTML = html

def make_option(text):
    op = document.createElement("option")
    op.textContent = text
    return op


# ── Config / connect wiring ────────────────────────────────────────────────
def push_config(evt=None):
    try:
        STATE.set_config(el("backendSel").value, el("motorSel").value, el("sensorSel").value)
    except Exception as e:
        window.alert(str(e))
    render()

def push_features(evt=None):
    feats = [cb.value for cb in document.querySelectorAll(".featchk") if cb.checked]
    try:
        STATE.set_features(feats)
    except Exception as e:
        window.alert(str(e))

async def on_connect(evt=None):
    if STATE.backend.connected:
        await STATE.disconnect()
        render()
        return
    try:
        await STATE.connect()
    except Exception as e:
        window.alert("Connection failed:\n" + str(e))
    render()

def on_mapping_change(evt=None):
    STATE.set_mapping(el("mappingSel").value)

def on_display_change(evt=None):
    STATE.set_display_feature(el("displaySel").value)

def on_mode_toggle(evt=None):
    STATE.set_mode(MODE_RUN if STATE.mode != MODE_RUN else MODE_TRAINING)

def on_record(evt=None):
    try:
        STATE.record()
    except Exception as e:
        window.alert(str(e))

def on_undo(evt=None):
    try:
        STATE.undo_last()
    except Exception as e:
        window.alert(str(e))

def on_clear(evt=None):
    if window.confirm("Remove all recorded points?"):
        STATE.clear_points()

def on_sim_motor(evt=None):
    STATE.sim_motor(float(el("simMotor").value))

def on_save(evt=None):
    text = STATE.export_json()
    uri = "data:application/json;charset=utf-8," + window.encodeURIComponent(text)
    a = document.createElement("a")
    a.href = uri
    a.download = "smart_motor_points.json"
    a.click()

def on_load_click(evt=None):
    el("loadFile").click()

def on_load_file_change(evt=None):
    files = evt.target.files
    if files.length == 0:
        return
    file = files.item(0)
    reader = window.FileReader.new()

    def on_loaded(e):
        try:
            STATE.import_json(reader.result)
        except Exception as ex:
            window.alert(str(ex))
        render(force=True)

    reader.addEventListener("load", create_proxy(on_loaded))
    reader.readAsText(file)
    evt.target.value = ""


# ── One-time DOM build ────────────────────────────────────────────────────
build_done = False

def build_once(s):
    global build_done
    b = el("backendSel")
    b.innerHTML = ""
    for name in ["Simulated", "LEGO Hardware"]:
        b.appendChild(make_option(name))
    b.value = s["backend"]

    b.addEventListener("change", create_proxy(push_config))
    el("motorSel").addEventListener("change", create_proxy(push_config))
    el("sensorSel").addEventListener("change", create_proxy(push_config))
    el("displaySel").addEventListener("change", create_proxy(on_display_change))
    el("mappingSel").addEventListener("change", create_proxy(on_mapping_change))
    el("modeBtn").addEventListener("click", create_proxy(on_mode_toggle))
    el("recordBtn").addEventListener("click", create_proxy(on_record))
    el("undoBtn").addEventListener("click", create_proxy(on_undo))
    el("clearBtn").addEventListener("click", create_proxy(on_clear))
    el("connectBtn").addEventListener("click", create_proxy(lambda evt: asyncio.ensure_future(on_connect(evt))))
    el("simMotor").addEventListener("input", create_proxy(on_sim_motor))
    el("saveBtn").addEventListener("click", create_proxy(on_save))
    el("loadBtn").addEventListener("click", create_proxy(on_load_click))
    el("loadFile").addEventListener("change", create_proxy(on_load_file_change))
    build_done = True


# ── Per-frame render ───────────────────────────────────────────────────────
last_feature_key = ""
last_sim_key = ""
last_indep_key = ""
last_speed_key = ""

def render(force=False):
    s = STATE.snapshot()
    global last_feature_key, last_sim_key, last_indep_key, last_speed_key

    if not build_done:
        build_once(s)

    el("connStatus").textContent = "Connected" if s["connected"] else "Not connected"
    el("connStatus").style.color = "#bce8c6" if s["connected"] else "#ffd0cd"
    el("connectBtn").textContent = "Disconnect" if s["connected"] else "Connect"

    for id_ in ["backendSel", "motorSel", "sensorSel"]:
        el(id_).disabled = s["connected"]

    show_lego = s["backend"] == "LEGO Hardware"
    el("motorRow").style.display = "" if show_lego else "none"
    el("sensorRow").style.display = "" if show_lego else "none"

    multi = s["numAxes"] > 1
    el("mappingRow").style.display = "" if multi else "none"
    el("mappingSel").value = s["mapping"]
    independent = multi and s["mapping"] == "independent"
    el("combinedInputs").style.display = "none" if independent else ""
    el("independentInputs").style.display = "" if independent else "none"
    el("combinedTraining").style.display = "none" if independent else ""
    el("independentTraining").style.display = "" if independent else "none"

    feat_key = ",".join(s["availableFeatures"]) + "|" + ",".join(s["selectedFeatures"])
    if feat_key != last_feature_key or force:
        last_feature_key = feat_key
        box = el("featureChecks")
        box.innerHTML = ""
        for f in s["availableFeatures"]:
            lab = document.createElement("label")
            cb = document.createElement("input")
            cb.type = "checkbox"
            cb.className = "featchk"
            cb.value = f
            cb.checked = f in s["selectedFeatures"]
            cb.addEventListener("change", create_proxy(push_features))
            lab.appendChild(cb)
            lab.appendChild(document.createTextNode(" " + f))
            box.appendChild(lab)

        dsel = el("displaySel")
        dsel.innerHTML = ""
        for f in s["selectedFeatures"]:
            dsel.appendChild(make_option(f))
        dsel.value = s["displayFeature"]

    indep_key = f"{s['numAxes']}|{','.join(s['motorLabels'])}|{','.join(s['availableFeatures'])}"
    if indep_key != last_indep_key or force:
        last_indep_key = indep_key
        frows = el("axisFeatureRows")
        rrows = el("axisRecordRows")
        frows.innerHTML = ""
        rrows.innerHTML = ""
        for a in range(s["numAxes"]):
            lab = s["motorLabels"][a]

            row = document.createElement("div")
            row.className = "axisrow"
            title = document.createElement("div")
            title.innerHTML = f"<b>{lab} motor</b> is driven by:"
            sel = document.createElement("select")
            sel.dataset.axis = str(a)
            for f in s["availableFeatures"]:
                sel.appendChild(make_option(f))

            def make_axis_handler(axis, select):
                def h(evt):
                    STATE.set_axis_feature(axis, select.value)
                return h
            sel.addEventListener("change", create_proxy(make_axis_handler(a, sel)))
            row.appendChild(title)
            row.appendChild(sel)
            frows.appendChild(row)

            rr = document.createElement("div")
            rr.className = "btn-row"
            rr.style.marginTop = "4px"
            rec = document.createElement("button")
            rec.className = "primary axis-rec"
            rec.textContent = "Record " + lab

            def make_rec_handler(axis):
                def h(evt):
                    try:
                        STATE.record(axis=axis)
                    except Exception as e:
                        window.alert(str(e))
                return h
            rec.addEventListener("click", create_proxy(make_rec_handler(a)))

            undo = document.createElement("button")
            undo.className = "lightbtn axis-undo"
            undo.textContent = "Undo " + lab

            def make_undo_handler(axis):
                def h(evt):
                    STATE.undo_last(axis=axis)
                return h
            undo.addEventListener("click", create_proxy(make_undo_handler(a)))

            clr = document.createElement("button")
            clr.className = "lightbtn"
            clr.textContent = "Clear " + lab

            def make_clear_handler(axis):
                def h(evt):
                    STATE.clear_points(axis=axis)
                return h
            clr.addEventListener("click", create_proxy(make_clear_handler(a)))

            rr.appendChild(rec)
            rr.appendChild(undo)
            rr.appendChild(clr)
            rrows.appendChild(rr)

    for sel in document.querySelectorAll("#axisFeatureRows select"):
        axis = int(sel.dataset.axis)
        sel.value = s["axisFeatures"][axis]
    for btn in document.querySelectorAll("#axisRecordRows .axis-rec, #axisRecordRows .axis-undo"):
        btn.disabled = (s["mode"] == "RUN") or (not s["connected"])

    speed_key = f"{s['numAxes']}|{','.join(s['motorLabels'])}"
    if speed_key != last_speed_key or force:
        last_speed_key = speed_key
        box = el("speedRows")
        box.innerHTML = ""
        for a in range(s["numAxes"]):
            lab = s["motorLabels"][a]
            wrap = document.createElement("div")
            wrap.style.marginTop = "4px"
            head = document.createElement("div")
            head.className = "muted"
            head.style.fontSize = "12px"
            val = document.createElement("span")
            val.className = "speedval"
            val.dataset.axis = str(a)
            head.textContent = lab + " motor speed: "
            head.appendChild(val)
            sl = document.createElement("input")
            sl.type = "range"
            sl.min = "1"
            sl.max = "100"
            sl.className = "slider"
            sl.dataset.axis = str(a)
            sl.value = str(s["axisSpeeds"][a])

            def make_speed_handler(axis, slider, label):
                def h(evt):
                    label.textContent = slider.value + "%"
                    STATE.set_axis_speed(axis, int(slider.value))
                return h
            sl.addEventListener("input", create_proxy(make_speed_handler(a, sl, val)))
            wrap.appendChild(head)
            wrap.appendChild(sl)
            box.appendChild(wrap)

    for sl in document.querySelectorAll("#speedRows .slider"):
        axis = int(sl.dataset.axis)
        if document.activeElement != sl:
            sl.value = str(s["axisSpeeds"][axis])
    for v in document.querySelectorAll("#speedRows .speedval"):
        v.textContent = str(s["axisSpeeds"][int(v.dataset.axis)]) + "%"

    el("simCard").style.display = "" if s["isSim"] else "none"
    if s["isSim"]:
        sim_key = ",".join(s["availableFeatures"])
        if sim_key != last_sim_key or force:
            last_sim_key = sim_key
            box = el("simSensors")
            box.innerHTML = ""
            for f in s["availableFeatures"]:
                wrap = document.createElement("div")
                lab = document.createElement("div")
                lab.className = "muted"
                lab.style.fontSize = "12px"
                lab.textContent = f
                sl = document.createElement("input")
                sl.type = "range"
                sl.min = "0"
                sl.max = "100"
                sl.value = "50"
                sl.className = "slider"

                def make_sim_handler(feature, slider):
                    def h(evt):
                        STATE.sim_sensor(feature, float(slider.value))
                    return h
                sl.addEventListener("input", create_proxy(make_sim_handler(f, sl)))
                wrap.appendChild(lab)
                wrap.appendChild(sl)
                box.appendChild(wrap)
        el("simMotorLabel").textContent = "Motor position (auto in run)" if s["mode"] == "RUN" else "Motor position (training)"
        el("simMotor").disabled = (s["mode"] == "RUN")

    pill = el("modePill")
    pill.textContent = s["mode"]
    pill.className = "pill " + ("run" if s["mode"] == "RUN" else "train")
    el("modeBtn").textContent = "Switch to TRAINING mode" if s["mode"] == "RUN" else "Switch to RUN mode"
    el("recordBtn").disabled = (s["mode"] == "RUN") or (not s["connected"])
    el("undoBtn").disabled = (s["mode"] == "RUN") or (not s["connected"])

    if s["connected"] and s["readings"]:
        el("rSensor").textContent = "Sensor: " + ", ".join(f"{k}={v:.0f}" for k, v in s["readings"].items())
    else:
        el("rSensor").textContent = "Sensor: —"
    el("rMotor").textContent = "Motor: " + ", ".join(f"{v:.0f}°" for v in s["positions"]) if s["connected"] else "Motor: —"
    if s["connected"] and s["target"]:
        el("rTarget").textContent = "Target: " + ", ".join("—" if v is None else f"{v:.0f}°" for v in s["target"])
    else:
        el("rTarget").textContent = "Target: —"
    if s["connected"] and s["speed"]:
        el("rSpeed").textContent = "Speed: " + ", ".join("—" if v is None else f"{v:.0f}%" for v in s["speed"])
    else:
        el("rSpeed").textContent = "Speed: —"
    el("rCount").textContent = s["pointsLabel"]

    draw_graph(s)


# ── Canvas graph (ported from smart_motor_web.py's drawGraph/drawStar) ──────
SERIES_COLORS = ["#1f77b4", "#ff7f0e"]

def draw_star(ctx, cx, cy, r):
    ctx.beginPath()
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        a = math.pi / 5 * i - math.pi / 2
        x = cx + math.cos(a) * rad
        y = cy + math.sin(a) * rad
        if i == 0:
            ctx.moveTo(x, y)
        else:
            ctx.lineTo(x, y)
    ctx.closePath()
    ctx.fill()

def draw_graph(s):
    cv = el("graph")
    ctx = cv.getContext("2d")
    W, H = cv.width, cv.height
    ctx.clearRect(0, 0, W, H)
    padL, padR, padT, padB = 54, 16, 14, 40
    plot = s["plot"]
    series = plot["series"]

    xs, ys = [], []
    for ser in series:
        for p in ser["points"]:
            xs.append(p[0]); ys.append(p[1])
        if ser["live"]:
            xs.append(ser["live"][0]); ys.append(ser["live"][1])
        if ser["targetY"] is not None:
            ys.append(ser["targetY"])

    xmin, xmax = (min(xs), max(xs)) if xs else (0, 100)
    ymin, ymax = (min(ys), max(ys)) if ys else (-180, 180)
    xpad = max((xmax - xmin) * 0.1, 5)
    ypad = max((ymax - ymin) * 0.1, 10)
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0

    def X(v): return padL + (v - xmin) / xspan * (W - padL - padR)
    def Y(v): return H - padB - (v - ymin) / yspan * (H - padT - padB)

    ctx.strokeStyle = "rgba(128,128,128,0.25)"
    ctx.fillStyle = window.getComputedStyle(document.body).color
    ctx.lineWidth = 1
    ctx.font = "11px system-ui, sans-serif"
    for i in range(6):
        gx = padL + i / 5 * (W - padL - padR)
        ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, H - padB); ctx.stroke()
        ctx.textAlign = "center"
        ctx.fillText(f"{xmin + i / 5 * (xmax - xmin):.0f}", gx, H - padB + 14)
    for i in range(6):
        gy = padT + i / 5 * (H - padT - padB)
        ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke()
        ctx.textAlign = "right"
        ctx.fillText(f"{ymax - i / 5 * (ymax - ymin):.0f}", padL - 6, gy + 4)
    ctx.textAlign = "center"
    ctx.fillText(f"Sensor reading ({plot['xlabel']})", (padL + W - padR) / 2, H - 6)

    for i, ser in enumerate(series):
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        ctx.fillStyle = col
        ctx.strokeStyle = col
        for p in ser["points"]:
            ctx.beginPath()
            ctx.arc(X(p[0]), Y(p[1]), 4, 0, math.pi * 2)
            ctx.fill()
        if ser["live"]:
            if s["mode"] == "RUN" and ser["targetY"] is not None:
                ctx.setLineDash(to_js([5, 4]))
                ctx.beginPath()
                ctx.moveTo(X(ser["live"][0]), Y(ser["live"][1]))
                ctx.lineTo(X(ser["live"][0]), Y(ser["targetY"]))
                ctx.stroke()
                ctx.setLineDash(to_js([]))
            draw_star(ctx, X(ser["live"][0]), Y(ser["live"][1]), 7)

    legend_html = "  &nbsp; ".join(
        f'<span style="color:{SERIES_COLORS[i % len(SERIES_COLORS)]}">● {ser["label"]}</span>'
        for i, ser in enumerate(series)
    ) + '  &nbsp; <span>★ live</span>'
    el("legend").innerHTML = legend_html


# ── Main loop ──────────────────────────────────────────────────────────────
async def main_loop():
    while True:
        try:
            STATE.tick()
            render()
        except Exception as e:
            print("render/tick error:", e)
        await asyncio.sleep(0.1)


def boot():
    el("loading-splash").style.display = "none"
    el("page-wrap").style.display = "block"
    render()
    asyncio.ensure_future(main_loop())


try:
    boot()
except Exception as e:
    import traceback
    msg = traceback.format_exc()
    print(msg)
    splash = el("loading-splash")
    splash.innerHTML = (
        '<div style="max-width:600px;padding:20px;text-align:left;font-family:monospace;'
        'font-size:12px;white-space:pre-wrap;color:#900;">'
        '<b>Startup error — this stayed on the splash screen because of an '
        'unhandled exception:</b><br><br>' + msg.replace("<", "&lt;") + '</div>'
    )
