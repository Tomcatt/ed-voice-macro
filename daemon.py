#!/usr/bin/env python3
"""
ED Voice+Macro Daemon
Gauss stagger autofire + voice commands for Elite Dangerous VR

Autofire logic:
  - User holds physical trigger → ED reads it directly and starts charging group 1
  - Daemon cycles fire groups every charge_hold_ms
  - Each cycle: group fires (charge complete), next group immediately starts charging
  - Release trigger → daemon stops cycling, ED stops firing
  - Net result: 3 Gauss groups fire in rotation with no gap
"""
import asyncio
import logging
import signal
import yaml
import evdev
from evdev import ecodes
import uinput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ed-macro")


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class VirtualInput:
    """Injects keyboard events via uinput (kernel-level, works in VR/Wayland)."""

    KEY_MAP = {
        "KEY_RIGHTBRACE": uinput.KEY_RIGHTBRACE,
        "KEY_LEFTBRACE":  uinput.KEY_LEFTBRACE,
        "KEY_COMMA":      uinput.KEY_COMMA,
        "KEY_PERIOD":     uinput.KEY_PERIOD,
        "KEY_SLASH":      uinput.KEY_SLASH,
        "KEY_N":          uinput.KEY_N,
        "KEY_M":          uinput.KEY_M,
    }

    def __init__(self, keys_needed: list[str]):
        keys = [self.KEY_MAP[k] for k in keys_needed if k in self.KEY_MAP]
        self._dev = uinput.Device(keys, name="ed-macro-virtual-kbd")
        self._key_map = self.KEY_MAP
        log.info("Virtual keyboard created")

    def tap(self, key_name: str, hold_ms: int = 50):
        key = self._key_map.get(key_name)
        if not key:
            log.warning(f"Unknown key: {key_name}")
            return
        self._dev.emit(key, 1)
        import time; time.sleep(hold_ms / 1000)
        self._dev.emit(key, 0)

    def close(self):
        self._dev.destroy()


class GaussAutofireLoop:
    """
    Stagger-fires 3 Gauss groups while the physical trigger is held.

    ED reads the physical trigger directly (no injection needed for the shot).
    Daemon only injects the "Next Fire Group" key every charge_hold_ms.

    Timeline with 3 groups, 1300ms charge:
      t=0.0s   trigger held, group 1 starts charging
      t=1.3s   group 1 fires → daemon cycles to group 2
      t=2.6s   group 2 fires → daemon cycles to group 3
      t=3.9s   group 3 fires → daemon cycles to group 1
      (repeat)
    """

    def __init__(self, cfg: dict, vinput: VirtualInput):
        self.cfg = cfg
        self.vinput = vinput
        self._task: asyncio.Task | None = None

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
        charge_s = self.cfg["charge_hold_ms"] / 1000
        cycle_key = self.cfg["cycle_group_key"]

        # Wait for first charge, then cycle repeatedly
        await asyncio.sleep(charge_s)
        while True:
            self.vinput.tap(cycle_key, hold_ms=50)
            log.debug(f"Fire group cycled")
            await asyncio.sleep(charge_s)


class StickWatcher:
    """Reads evdev events from one joystick and calls registered callbacks."""

    def __init__(self, device_path: str, label: str, callbacks: dict):
        self.dev = evdev.InputDevice(device_path)
        self.label = label
        self.callbacks = callbacks  # {button_code: {"down": coro_fn, "up": coro_fn}}
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


class VoiceListener:
    """
    Push-to-talk voice recognition via faster-whisper (CUDA).
    PTT: hold configured button → speak → release → transcribe → dispatch.
    Always-on: set ptt_enabled: false in config (uses voice activity detection).
    """

    def __init__(self, cfg: dict, command_handler):
        self.cfg = cfg
        self.handler = command_handler
        self._listening = False
        self._frames: list = []
        self._model = None  # loaded lazily on first use

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.cfg["model"],
                device=self.cfg["compute_device"],
                compute_type="float16",
            )
            log.info(f"Whisper model loaded: {self.cfg['model']}")
        return self._model

    async def ptt_down(self):
        log.info("PTT: recording")
        self._listening = True
        self._frames = []
        asyncio.create_task(self._record())

    async def ptt_up(self):
        self._listening = False
        log.info("PTT: processing")
        # Give the record loop one tick to finish
        await asyncio.sleep(0.05)
        await self._transcribe_and_dispatch()

    async def _record(self):
        import sounddevice as sd
        import numpy as np
        SAMPLE_RATE = 16000
        CHUNK = 1024

        # Find Valve Index HMD mic
        devices = sd.query_devices()
        mic_idx = None
        for i, d in enumerate(devices):
            if "Valve VR Radio" in d["name"] and d["max_input_channels"] > 0:
                mic_idx = i
                break
        if mic_idx is None:
            log.warning("Valve Index mic not found — falling back to default")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            device=mic_idx, blocksize=CHUNK, dtype="float32") as stream:
            while self._listening:
                chunk, _ = stream.read(CHUNK)
                self._frames.append(chunk)
                await asyncio.sleep(0)

    async def _transcribe_and_dispatch(self):
        if not self._frames:
            return
        import numpy as np
        audio = np.concatenate(self._frames).flatten()
        if len(audio) < 3200:  # < 0.2s — too short, skip
            return

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._run_whisper, audio)
        if text:
            log.info(f"Voice: '{text}'")
            await self.handler(text)

    def _run_whisper(self, audio):
        import numpy as np
        model = self._get_model()
        segments, _ = model.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip().lower()


class CommandHandler:
    def __init__(self, autofire: GaussAutofireLoop, cfg: dict):
        self.autofire = autofire
        self.cfg = cfg

    async def __call__(self, text: str):
        for name, cmd in self.cfg["voice"]["commands"].items():
            if any(p in text for p in cmd["phrases"]):
                action = cmd["action"]
                log.info(f"Command matched: {name} → {action}")
                if action == "stop_autofire":
                    await self.autofire.stop()
                elif action == "keypress":
                    pass  # TODO: inject cmd["key"] via vinput
                return
        log.debug(f"No command matched: '{text}'")


async def main():
    cfg = load_config()
    af_cfg = cfg["autofire"]
    voice_cfg = cfg["voice"]

    vinput = VirtualInput([af_cfg["cycle_group_key"]])
    autofire = GaussAutofireLoop(af_cfg, vinput)
    handler = CommandHandler(autofire, cfg)
    voice = VoiceListener(voice_cfg, handler)

    trigger_code = getattr(ecodes, af_cfg["trigger_button"])
    ptt_code     = getattr(ecodes, voice_cfg["ptt_button"])

    callbacks = {
        trigger_code: {"down": autofire.start, "up": autofire.stop},
        ptt_code:     {"down": voice.ptt_down, "up": voice.ptt_up},
    }

    device_map = cfg["devices"]
    trigger_dev = device_map[af_cfg["trigger_device"]]

    watcher = StickWatcher(trigger_dev, af_cfg["trigger_device"], callbacks)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(autofire, vinput)))

    log.info("ED macro daemon ready — hold trigger to autofire, PTT for voice")
    await watcher.run()


async def _shutdown(autofire, vinput):
    await autofire.stop()
    vinput.close()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    asyncio.run(main())
