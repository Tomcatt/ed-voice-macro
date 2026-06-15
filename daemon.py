#!/usr/bin/env python3
"""
ED Voice+Macro Daemon
Gauss stagger autofire + voice commands for Elite Dangerous VR

Config:  ~/.config/ed-voice-macro/app.yaml
Profile: ~/.config/ed-voice-macro/profiles/<name>.yaml
Active:  ~/.config/ed-voice-macro/active_profile  (one line, hot-swappable)
"""
import asyncio
import logging
import signal
import shutil
from pathlib import Path

import yaml
import evdev
from evdev import ecodes
import uinput

log = logging.getLogger("ed-macro")

CONFIG_DIR         = Path.home() / ".config" / "ed-voice-macro"
APP_CONFIG_PATH    = CONFIG_DIR / "app.yaml"
PROFILES_DIR       = CONFIG_DIR / "profiles"
ACTIVE_PROFILE_PATH = CONFIG_DIR / "active_profile"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_app_config() -> dict:
    with open(APP_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def read_active_profile_name() -> str:
    return ACTIVE_PROFILE_PATH.read_text().strip()


# ---------------------------------------------------------------------------
# Virtual keyboard (uinput)
# ---------------------------------------------------------------------------

KEY_MAP = {
    "KEY_RIGHTBRACE": uinput.KEY_RIGHTBRACE,
    "KEY_LEFTBRACE":  uinput.KEY_LEFTBRACE,
    "KEY_TAB":        uinput.KEY_TAB,
    "KEY_U":          uinput.KEY_U,
    "KEY_DELETE":     uinput.KEY_DELETE,
    "KEY_COMMA":      uinput.KEY_COMMA,
    "KEY_PERIOD":     uinput.KEY_PERIOD,
    "KEY_SLASH":      uinput.KEY_SLASH,
    "KEY_N":          uinput.KEY_N,
    "KEY_M":          uinput.KEY_M,
}

class VirtualInput:
    def __init__(self):
        self._dev = uinput.Device(list(KEY_MAP.values()), name="ed-macro-virtual-kbd")
        log.info("Virtual keyboard created")

    def tap(self, key_name: str, hold_ms: int = 50):
        key = KEY_MAP.get(key_name)
        if not key:
            log.warning(f"Unknown key: {key_name}")
            return
        import time
        self._dev.emit(key, 1)
        time.sleep(hold_ms / 1000)
        self._dev.emit(key, 0)

    def close(self):
        self._dev.destroy()


# ---------------------------------------------------------------------------
# Gauss stagger autofire
# ---------------------------------------------------------------------------

class GaussAutofireLoop:
    """
    Cycles fire groups every charge_hold_ms while the physical trigger is held.
    ED reads the trigger directly — daemon only injects the cycle_group_key.

    Timeline (3 groups, 1300ms):
      t=0.0s  trigger held, group 1 starts charging
      t=1.3s  group 1 fires → cycle to group 2
      t=2.6s  group 2 fires → cycle to group 3
      t=3.9s  group 3 fires → cycle to group 1  (repeat)
    """

    def __init__(self, profile: dict, vinput: VirtualInput):
        self.cfg = profile["autofire"]
        self.vinput = vinput
        self._task: asyncio.Task | None = None

    def reload(self, profile: dict):
        self.cfg = profile["autofire"]

    @property
    def running(self):
        return self._task is not None and not self._task.done()

    async def start(self):
        if self.running:
            return
        log.info("Autofire START")
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Autofire STOP")

    async def _loop(self):
        charge_s  = self.cfg["charge_hold_ms"] / 1000
        inter_s   = self.cfg.get("inter_shot_ms", 150) / 1000
        cycle_key = self.cfg["cycle_group_key"]
        await asyncio.sleep(charge_s)
        while True:
            self.vinput.tap(cycle_key, hold_ms=50)
            log.debug("Fire group cycled")
            await asyncio.sleep(charge_s + inter_s)


# ---------------------------------------------------------------------------
# Joystick watcher
# ---------------------------------------------------------------------------

class StickWatcher:
    def __init__(self, device_path: str, label: str, callbacks: dict):
        self.dev = evdev.InputDevice(device_path)
        self.label = label
        self.callbacks = callbacks   # {button_code: {"down": coro, "up": coro}}
        log.info(f"Watching {label}: {self.dev.name}")

    async def run(self):
        async for event in self.dev.async_read_loop():
            if event.type != ecodes.EV_KEY:
                continue
            cb = self.callbacks.get(event.code)
            if not cb:
                continue
            if event.value == 1 and "down" in cb:
                asyncio.create_task(cb["down"]())
            elif event.value == 0 and "up" in cb:
                asyncio.create_task(cb["up"]())


# ---------------------------------------------------------------------------
# Voice listener (faster-whisper + PTT)
# ---------------------------------------------------------------------------

class VoiceListener:
    def __init__(self, app_cfg: dict, command_handler):
        self.cfg = app_cfg["voice"]
        self.handler = command_handler
        self._listening = False
        self._frames: list = []
        self._model = None

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.cfg["model"],
                device=self.cfg["compute_device"],
                compute_type="float16",
            )
            log.info(f"Whisper loaded: {self.cfg['model']}")
        return self._model

    async def ptt_down(self):
        log.info("PTT: recording")
        self._listening = True
        self._frames = []
        asyncio.create_task(self._record())

    async def ptt_up(self):
        self._listening = False
        log.info("PTT: processing")
        await asyncio.sleep(0.05)
        await self._transcribe_and_dispatch()

    async def _record(self):
        import sounddevice as sd
        import numpy as np
        RATE, CHUNK = 16000, 1024
        devices = sd.query_devices()
        mic_idx = next(
            (i for i, d in enumerate(devices)
             if "Valve VR Radio" in d["name"] and d["max_input_channels"] > 0),
            None
        )
        if mic_idx is None:
            log.warning("Valve Index mic not found — using default")
        with sd.InputStream(samplerate=RATE, channels=1, device=mic_idx,
                            blocksize=CHUNK, dtype="float32") as stream:
            while self._listening:
                chunk, _ = stream.read(CHUNK)
                self._frames.append(chunk)
                await asyncio.sleep(0)

    async def _transcribe_and_dispatch(self):
        if not self._frames:
            return
        import numpy as np
        audio = np.concatenate(self._frames).flatten()
        if len(audio) < 3200:
            return
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._run_whisper, audio)
        if text:
            log.info(f"Voice: '{text}'")
            await self.handler(text)

    def _run_whisper(self, audio):
        model = self._get_model()
        segments, _ = model.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip().lower()


# ---------------------------------------------------------------------------
# Command dispatcher (profile-aware, hot-reloadable)
# ---------------------------------------------------------------------------

class CommandDispatcher:
    def __init__(self, autofire: GaussAutofireLoop, profile_manager):
        self.autofire = autofire
        self.pm = profile_manager
        self._commands: list[dict] = []

    def reload(self, profile: dict):
        self._commands = profile.get("commands", [])

    async def __call__(self, text: str):
        for cmd in self._commands:
            if any(p in text for p in cmd["phrases"]):
                action = cmd["action"]
                log.info(f"Command: {cmd['name']} → {action}")
                if action == "stop_autofire":
                    await self.autofire.stop()
                elif action == "load_profile":
                    await self.pm.switch(cmd["profile"])
                elif action == "keypress":
                    self.autofire.vinput.tap(cmd["key"])
                return
        log.debug(f"No match: '{text}'")


# ---------------------------------------------------------------------------
# Profile manager (hot-swap)
# ---------------------------------------------------------------------------

class ProfileManager:
    def __init__(self, app_cfg: dict, autofire: GaussAutofireLoop,
                 dispatcher: CommandDispatcher):
        self.app_cfg    = app_cfg
        self.autofire   = autofire
        self.dispatcher = dispatcher
        self._current   = ""

    async def switch(self, name: str):
        if name == self._current:
            return
        log.info(f"Loading profile: {name}")
        try:
            profile = load_profile(name)
        except FileNotFoundError as e:
            log.error(e)
            return
        await self.autofire.stop()
        self.autofire.reload(profile)
        self.dispatcher.reload(profile)
        self._current = name
        ACTIVE_PROFILE_PATH.write_text(name + "\n")
        log.info(f"Profile active: {profile['name']}")

    async def watch(self):
        """Poll active_profile file and hot-swap on change."""
        while True:
            await asyncio.sleep(1)
            name = read_active_profile_name()
            if name != self._current:
                await self.switch(name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    app_cfg = load_app_config()
    active  = read_active_profile_name()
    profile = load_profile(active)

    vinput     = VirtualInput()
    autofire   = GaussAutofireLoop(profile, vinput)
    dispatcher = CommandDispatcher(autofire, None)   # pm injected below
    pm         = ProfileManager(app_cfg, autofire, dispatcher)
    dispatcher.pm = pm
    dispatcher.reload(profile)
    pm._current = active

    voice = VoiceListener(app_cfg, dispatcher)

    dev_map = app_cfg["devices"]
    af_cfg  = profile["autofire"]
    v_cfg   = app_cfg["voice"]

    trigger_code = getattr(ecodes, af_cfg["trigger_button"])
    ptt_code     = getattr(ecodes, v_cfg["ptt_button"])

    callbacks = {
        trigger_code: {"down": autofire.start, "up": autofire.stop},
        ptt_code:     {"down": voice.ptt_down, "up": voice.ptt_up},
    }

    trigger_dev = dev_map[af_cfg["trigger_device"]]
    watcher     = StickWatcher(trigger_dev, af_cfg["trigger_device"], callbacks)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(_shutdown(autofire, vinput))
        )

    log.info(f"ED macro daemon ready — profile: {profile['name']}")
    await asyncio.gather(watcher.run(), pm.watch())


async def _shutdown(autofire, vinput):
    await autofire.stop()
    vinput.close()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
