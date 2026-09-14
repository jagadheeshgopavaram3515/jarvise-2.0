"""
HealthMonitor (Phase 3 + 4) — heartbeat + automatic thread recovery.

A daemon thread that, every ``HEALTH_INTERVAL_S`` seconds:

  * checks ``is_alive()`` for every registered service thread,
  * records and publishes a heartbeat line:

        [HEALTH] STT alive=true TTS alive=true Dispatcher alive=true ...

  * if a *critical* service has died (and we're not shutting down), logs at
    CRITICAL and restarts it via its factory — without restarting Jarvis:

        [HEALTH]   STT thread stopped
        [RECOVERY] Restarting STT
        [RECOVERY] STT recovered

Each service is registered with a ``factory`` that builds **and starts** a fresh
thread (the pipeline supplies these, re-wiring any bus references). Restarts are
capped per service (``HEALTH_MAX_RESTARTS``) so a permanently-broken dependency
(no microphone, missing model) can't spin in a tight loop.

The monitor never raises into anything else; a bad check or a failed restart is
logged and the loop continues. Itself supervised: its own loop can't die.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from assistant import config
from assistant.core.log import get

log = get("health")

# A factory builds AND starts a brand-new daemon thread for the service and
# returns it (re-wiring any shared references as a side effect).
ServiceFactory = Callable[[], threading.Thread]


@dataclass
class _Supervised:
    name: str
    factory: ServiceFactory
    critical: bool
    thread: threading.Thread
    restarts: int = 0
    last_restart: float = 0.0
    # A service that exits by design (e.g. backchannel disabled) is recorded but
    # never restarted; `critical=False` covers that.


class HealthMonitor(threading.Thread):
    def __init__(self, bus, interval: Optional[float] = None):
        super().__init__(name="HealthMonitor", daemon=True)
        self.bus = bus
        self.interval = float(interval if interval is not None else config.HEALTH_INTERVAL_S)
        self._services: List[_Supervised] = []
        self._status: Dict[str, bool] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ registration
    def register(self, name: str, thread: threading.Thread,
                 factory: ServiceFactory, critical: bool = True) -> None:
        """Register a running service thread + how to recreate it."""
        with self._lock:
            self._services.append(_Supervised(name, factory, critical, thread))
            self._status[name] = thread.is_alive()
        log.info("[HEALTH] registered %s (critical=%s)", name, critical)

    def status(self) -> Dict[str, bool]:
        """Snapshot of last-seen alive flags (for the GUI / external probes)."""
        with self._lock:
            return dict(self._status)

    def restart_counts(self) -> Dict[str, int]:
        with self._lock:
            return {s.name: s.restarts for s in self._services}

    # -------------------------------------------------------------------- loop
    def run(self) -> None:
        if not config.HEALTH_MONITOR_ENABLED:
            log.info("[HEALTH] monitor disabled by config")
            return
        log.info("[HEALTH] monitor started (interval=%.0fs, auto_recovery=%s)",
                 self.interval, config.HEALTH_AUTO_RECOVERY)
        # Supervised loop: a bad check or restart can NEVER kill the monitor.
        while not self.bus.shutdown.is_set():
            try:
                self._check_once()
            except Exception:
                log.exception("[HEALTH] check cycle error — monitor stays alive")
            self._sleep_interval()
        log.info("[HEALTH] monitor stopped (shutdown)")

    def _sleep_interval(self) -> None:
        # Sleep in small slices so shutdown stays responsive.
        slices = max(1, int(self.interval / 0.5))
        for _ in range(slices):
            if self.bus.shutdown.is_set():
                return
            time.sleep(0.5)

    def _check_once(self) -> None:
        parts: List[str] = []
        changed = False
        any_down = False
        with self._lock:
            services = list(self._services)
        for sup in services:
            alive = sup.thread.is_alive()
            with self._lock:
                prev = self._status.get(sup.name)
                self._status[sup.name] = alive
            if prev is not None and prev != alive:
                changed = True
            if not alive:
                any_down = True
            parts.append(f"{sup.name} alive={str(alive).lower()}")
            if not alive and not self.bus.shutdown.is_set():
                if sup.critical and config.HEALTH_AUTO_RECOVERY:
                    self._recover(sup)
                elif not sup.critical:
                    log.debug("[HEALTH] %s not alive (non-critical, no restart)", sup.name)
        # Steady-state is SILENT: only log a heartbeat when a state changed, a
        # critical service is down, or the user explicitly opts into every-cycle
        # logging. This keeps the monitor's I/O footprint at zero when all is well.
        if config.HEALTH_LOG_HEARTBEAT or changed or (any_down and config.HEALTH_AUTO_RECOVERY):
            log.info("[HEALTH] " + " ".join(parts))
        else:
            log.debug("[HEALTH] " + " ".join(parts))

    # ---------------------------------------------------------------- recovery
    def _recover(self, sup: _Supervised) -> None:
        if sup.restarts >= config.HEALTH_MAX_RESTARTS:
            log.error("[RECOVERY] %s exceeded max restarts (%d) — giving up; "
                      "service stays down (rest of Jarvis keeps running)",
                      sup.name, config.HEALTH_MAX_RESTARTS)
            return
        # CRITICAL is the ONLY level that triggers recovery (Phase 7).
        log.critical("[HEALTH] %s thread stopped", sup.name)
        log.warning("[RECOVERY] Restarting %s (attempt %d/%d)",
                    sup.name, sup.restarts + 1, config.HEALTH_MAX_RESTARTS)
        try:
            new_thread = sup.factory()
            if new_thread is None or not isinstance(new_thread, threading.Thread):
                raise RuntimeError("factory did not return a Thread")
            with self._lock:
                sup.thread = new_thread
                sup.restarts += 1
                sup.last_restart = time.monotonic()
                self._status[sup.name] = new_thread.is_alive()
            # Give it a moment to come up before the next health check.
            time.sleep(config.HEALTH_RESTART_BACKOFF_S)
            log.warning("[RECOVERY] %s recovered", sup.name)
            self.bus.post_gui("notification", f"{sup.name} recovered")
        except Exception:
            log.exception("[RECOVERY] %s restart failed — will retry next cycle", sup.name)
