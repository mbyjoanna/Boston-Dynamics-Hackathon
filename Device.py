import asyncio
import threading
import concurrent.futures
import legoeducation as le
import legoeducation.background_worker as bw
import legoeducation.basic_device as bd
from pyscript.js_modules import ble
from pyscript.ffi import create_proxy

# --------------------------------------------------------------------------
# Same trick used by the Neural Network Builder: legoeducation expects a
# real bleak/threaded backend. In the browser there is no thread and no
# blocking wait, so these two calls are patched to return immediately/
# non-blocking, and the worker's "thread" is really just an asyncio task
# driven by the browser's Web Bluetooth API (ble.js) via BLEDevice.
# --------------------------------------------------------------------------
def _nowait_event_wait(self, timeout=None):
    return self._flag
threading.Event.wait = _nowait_event_wait

def _nowait_future_result(self, timeout=None):
    if self._state == 'FINISHED':
        return self._Future__get_result()
    return None
concurrent.futures.Future.result = _nowait_future_result

_worker_started = False

def _wasm_start_thread(self):
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    self.loop = asyncio.get_event_loop()
    self.loop_ready.set()
    asyncio.ensure_future(_wasm_worker_loop(self))

def _wasm_put_request(self, request):
    asyncio.ensure_future(self.async_put_request(request))

bw.Worker.start_thread = _wasm_start_thread
bw.Worker.put_request  = _wasm_put_request

async def _wasm_worker_loop(worker):
    worker._js_ble_registry = {}
    while True:
        try:
            req = await worker.request_queue.get()
            if req is None:
                break
            topic = req.get('topic')
            if topic == 'send':
                device  = req.get('msg')
                message = req.get('msg2')
                js_ble  = worker._js_ble_registry.get(id(device))
                if js_ble is not None and message is not None:
                    try:
                        await js_ble.send(list(message))
                    except Exception as e:
                        print(f"BLE send error: {e}")
            elif topic == 'connect':
                cb = req.get('msg3')
                if cb:
                    cb(True)
            elif topic == 'disconnect':
                device = req.get('msg')
                js_ble = worker._js_ble_registry.pop(id(device), None)
                if js_ble is not None:
                    js_ble.disconnect()
            elif topic == 'scan':
                cb = req.get('msg2')
                if cb:
                    cb([])
        except Exception as e:
            print(f"Worker loop error: {e}")

bd.my_worker.start_thread()

_DEVICE_MAP = {
    "Double Motor": le.DoubleMotor,
    "Single Motor": le.SingleMotor,
    "Color Sensor": le.ColorSensor,
    "Controller":   le.Controller,
}

def _dedupe_name(base_name: str, existing_names: list[str]) -> str:
    """Append ' (1)', ' (2)', etc. if base_name is already in use."""
    if base_name not in existing_names:
        return base_name
    i = 1
    while f"{base_name} ({i})" in existing_names:
        i += 1
    return f"{base_name} ({i})"

NOTIFY_UUID = '0000fd02-0002-1000-8000-00805f9b34fb'


class Element:
    """One connected LEGO Education hub (a motor OR a sensor). Smart Motor
    keeps exactly one Element in the 'motor' role and one in the 'sensor'
    role -- see main.py's connect_hardware()."""

    def __init__(self):
        self.myble = ble.BLEDevice.new()
        self.hub = None
        self.name = None
        self.in_list = []
        self.out_list = []
        self.state = None

    async def connect(self, existing_names: list[str] | None = None):
        existing_names = existing_names or []
        print("Scanning...")
        try:
            await self.myble.scan()
        except Exception as e:
            print(f"Scan cancelled or failed: {e}")
            return

        raw_name = self.myble.name
        if not raw_name:
            print("No device selected.")
            return

        base_name = str(raw_name)
        self.name = _dedupe_name(base_name, existing_names)

        hub_cls = None
        for key, cls in _DEVICE_MAP.items():
            if key in self.name:
                hub_cls = cls
                self.build_state()
                break

        if hub_cls is None:
            print(f"Unknown device type: {self.name}")
            return

        try:
            self.hub = hub_cls()

            _hub = self.hub
            self._notification_proxy = create_proxy(
                lambda data: asyncio.ensure_future(
                    _hub._device_callback(NOTIFY_UUID, bytes(data.to_py()))
                )
            )
            self.myble.callback = self._notification_proxy

            SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb'
            WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb'
            success = await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID)
            if not success:
                print("BLE connect() returned false")
                self.hub = None
                return

            print("BLE GATT connected, setting up hub...")
            self.hub.device = self.hub

            registry = getattr(bd.my_worker, '_js_ble_registry', None)
            if registry is not None:
                registry[id(self.hub)] = self.myble
            else:
                print("WARNING: worker registry not found")

            self.hub.connected = True
            await asyncio.sleep(0)

            print("Sending device_notification_request...")
            self.hub.device_notification_request(100, blocking=False)

            max_wait, interval, waited = 2.0, 0.05, 0.0
            while not self.hub._first_notification_ready.is_set() and waited < max_wait:
                await asyncio.sleep(interval)
                waited += interval

            if self.hub._first_notification_ready.is_set():
                print("First notification received, sensor values ready.")
            else:
                print("Warning: timed out waiting for first notification.")

            print(f"Hub ready: {self.name}")
            self.hub.set_notification_callback(self.notif_callback)

            # Motors: zero the position counter right after connecting, so
            # recorded/predicted positions are relative to where the model
            # starts (matches smart_motor_web.py's LegoBackend.connect()).
            if "Single Motor" in self.name:
                self.hub.motor_reset_relative_position(blocking=False)
            elif "Double Motor" in self.name:
                self.hub.motor_reset_relative_position(motor=le.MOTOR_LEFT, blocking=False)
                self.hub.motor_reset_relative_position(motor=le.MOTOR_RIGHT, blocking=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Hub setup failed: {e}")
            self.hub = None

    async def disconnect(self):
        try:
            if self.hub and self.hub.connected:
                if "Motor" in (self.name or ""):
                    self.stop()
                self.hub.disconnect()
                self.hub.connected = False
        finally:
            await asyncio.sleep(0)
            self.hub = None
            self.myble = None

    @property
    def connected(self):
        return bool(self.hub and getattr(self.hub, "connected", False))

    def build_state(self):
        if "Double Motor" in self.name:
            self.state = {
                "LeftPosition": 0.0, "RightPosition": 0.0, "LeftAngle": 0.0, "RightAngle": 0.0,
                "LeftSpeed": 0.0, "RightSpeed": 0.0, "RightPower": 0.0, "LeftPower": 0.0,
            }
        elif "Single Motor" in self.name:
            self.state = {"Position": 0.0, "Speed": 0.0, "Angle": 0.0}
        elif "Controller" in self.name:
            self.state = {
                "LeftPercent": 0.0, "RightPercent": 0.0,
                "LeftAngle": 0.0, "RightAngle": 0.0,
            }
        else:  # Color Sensor
            self.state = {
                "Reflection": 0.0, "Color": 0.0,
                "Hue": 0.0, "Saturation": 0.0, "Value": 0.0,
                "Red": 0.0, "Green": 0.0, "Blue": 0.0,
            }

    def notif_callback(self, data):
        parsed_items = le.device_notification_parser(data)
        for parsed_item in parsed_items:
            if isinstance(parsed_item, le.MotorNotification):
                if "Double Motor" in self.name:
                    side = "Left" if parsed_item.motorBitMask == le.MOTOR_BITS_LEFT else "Right"
                    self.state[side + "Position"] = parsed_item.position
                    self.state[side + "Angle"] = parsed_item.absolutePosition
                    self.state[side + "Speed"] = parsed_item.speed
                    self.state[side + "Power"] = parsed_item.power
                else:
                    self.state["Position"] = parsed_item.position
                    self.state["Angle"] = parsed_item.absolutePosition
                    self.state["Speed"] = parsed_item.speed
                    self.state["Power"] = parsed_item.power

            elif isinstance(parsed_item, le.ColorSensorNotification):
                self.state["Reflection"] = parsed_item.reflection
                self.state["Color"] = parsed_item.color
                self.state["Hue"] = parsed_item.hue
                self.state["Saturation"] = parsed_item.saturation
                self.state["Value"] = parsed_item.value
                self.state["Red"] = parsed_item.rawRed
                self.state["Green"] = parsed_item.rawGreen
                self.state["Blue"] = parsed_item.rawBlue

            elif isinstance(parsed_item, le.ControllerNotification):
                self.state["LeftPercent"] = parsed_item.leftPercent
                self.state["RightPercent"] = parsed_item.rightPercent
                self.state["LeftAngle"] = parsed_item.leftAngle
                self.state["RightAngle"] = parsed_item.rightAngle

    # -- motor position control (Smart Motor drives to a position, not a
    #    raw speed, so these are new relative to the Neural Network Builder's
    #    Device.py) --
    def move_to_position(self, position, speed=50):
        """Single motor only."""
        if "Single Motor" not in (self.name or ""):
            return
        self.hub.motor_run_to_relative_position(
            int(round(position)), speed=int(round(speed)), blocking=False)

    def move_to_position_double(self, left, right, speed_left=50, speed_right=50):
        """Double motor only."""
        if "Double Motor" not in (self.name or ""):
            return
        self.hub.motor_run_to_relative_position(
            int(round(left)), motor=le.MOTOR_LEFT, speed=int(round(speed_left)), blocking=False)
        self.hub.motor_run_to_relative_position(
            int(round(right)), motor=le.MOTOR_RIGHT, speed=int(round(speed_right)), blocking=False)

    def stop(self):
        if "Single Motor" in (self.name or ""):
            self.hub.motor_stop()
        elif "Double Motor" in (self.name or ""):
            self.hub.motor_stop(motor=le.MOTOR_BOTH)
