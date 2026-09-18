"""
MQTT Telemetry Receiver for Ground Truth Rain Gauge Stations (Phase 2).

Subscribes to sensors/rainfall/# on Mosquitto broker (localhost:1883)
and caches real-time precipitation intensity for bias correction in terrain_stream.py.
"""

import json
import logging
import threading
from typing import Optional, Dict, Any
import paho.mqtt.client as mqtt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TelemetryReceiver")


class RainGaugeTelemetryReceiver:
    """Listens for FreeRTOS edge telemetry over local MQTT broker."""

    def __init__(self, host: str = "localhost", port: int = 1883, topic: str = "sensors/rainfall/#"):
        self.host = host
        self.port = port
        self.topic = topic
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self._thread: Optional[threading.Thread] = None

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        logger.info("Connected to Mosquitto MQTT broker at %s:%d (rc: %s)", self.host, self.port, reason_code)
        client.subscribe(self.topic)
        logger.info("Subscribed to MQTT topic: %s", self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            with self._lock:
                self.latest_telemetry = payload
            logger.info("Received edge telemetry from %s: rate=%.2f mm/hr, 6hr=%.2f mm",
                        payload.get("station_id"),
                        payload.get("rate_mm_hr", 0.0),
                        payload.get("accum_6hr_mm", 0.0))
        except Exception as ex:
            logger.warning("Failed to decode telemetry payload: %s", ex)

    def start(self):
        """Starts background MQTT network loop."""
        try:
            self.client.connect(self.host, self.port, keepalive=60)
            self.client.loop_start()
            logger.info("MQTT Telemetry listener started in background.")
        except Exception as ex:
            logger.warning("Could not connect to MQTT broker (%s). Falling back to synthetic baseline.", ex)

    def stop(self):
        """Stops background MQTT network loop."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT Telemetry listener stopped.")
        except Exception:
            pass

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent received rain gauge telemetry record."""
        with self._lock:
            return dict(self.latest_telemetry) if self.latest_telemetry else None


if __name__ == "__main__":
    import time
    receiver = RainGaugeTelemetryReceiver()
    receiver.start()
    print("Listening for 5 seconds...")
    time.sleep(5)
    print("Latest telemetry:", receiver.get_latest())
    receiver.stop()
