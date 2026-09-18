"""
Cedar Policy Evaluator for Automated Disaster Broadcast (Phase 5).

Validates statutory flood alerting safety thresholds and system modes
using the official AWS Cedar Policy Engine CLI.
"""

import os
import sys
import json
import logging
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CedarPolicyEvaluator")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

CEDAR_BIN = os.path.join(PROJECT_ROOT, "bin", "cedar.exe")
if not os.path.exists(CEDAR_BIN):
    # Fallback to cedar in PATH or Linux binary
    CEDAR_BIN = "cedar"

DEFAULT_POLICY_PATH = os.path.join(CURRENT_DIR, "alert_policy.cedar")
DEFAULT_ENTITIES_PATH = os.path.join(CURRENT_DIR, "entities.json")


@dataclass
class CedarDecision:
    decision: str  # "ALLOW" or "DENY"
    allowed: bool
    diagnostics: List[str]
    raw_output: str
    context: Dict[str, Any]


class CedarPolicyEvaluator:
    """Evaluates disaster broadcast authorization using AWS Cedar Engine."""

    def __init__(
        self,
        cedar_bin: str = CEDAR_BIN,
        policy_path: str = DEFAULT_POLICY_PATH,
        entities_path: str = DEFAULT_ENTITIES_PATH
    ):
        self.cedar_bin = cedar_bin
        self.policy_path = policy_path
        self.entities_path = entities_path

        if not os.path.exists(self.policy_path):
            raise FileNotFoundError(f"Cedar policy file not found at: {self.policy_path}")

        if not os.path.exists(self.entities_path):
            raise FileNotFoundError(f"Cedar entities file not found at: {self.entities_path}")

    def evaluate(
        self,
        flood_depth_cm: float,
        time_to_peak_hours: int,
        affected_parcels_count: int,
        system_mode: str = "LIVE",
        principal: str = 'Role::"MLHydrologyAgent"',
        action: str = 'Action::"BroadcastAlert"',
        resource: str = 'System::"AlertBroadcastSystem"',
    ) -> CedarDecision:
        """
        Executes Cedar authorization check against statutory criteria:
        - principal == Role::"MLHydrologyAgent"
        - action == Action::"BroadcastAlert"
        - flood_depth_cm >= 15
        - time_to_peak_hours <= 6
        - affected_parcels_count > 0
        - system_mode != "DRY_RUN"
        """
        # Cedar integers are 64-bit signed integers
        context = {
            "flood_depth_cm": int(round(flood_depth_cm)),
            "time_to_peak_hours": int(time_to_peak_hours),
            "affected_parcels_count": int(affected_parcels_count),
            "system_mode": str(system_mode).strip().upper()
        }

        request_payload = {
            "principal": principal,
            "action": action,
            "resource": resource,
            "context": context
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(request_payload, tf, indent=2)
            temp_req_path = tf.name

        try:
            cmd = [
                self.cedar_bin,
                "authorize",
                "--policies", self.policy_path,
                "--entities", self.entities_path,
                "--request-json", temp_req_path,
                "--verbose"
            ]

            logger.debug("Executing Cedar command: %s", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode not in (0, 2):
                logger.error("Cedar CLI execution error (code %d): %s", proc.returncode, stderr or stdout)
                raise RuntimeError(f"Cedar CLI error (code {proc.returncode}): {stderr or stdout}")

            # Cedar output contains ALLOW or DENY
            lines = stdout.splitlines()
            decision_str = lines[0].strip() if lines else "DENY"
            is_allowed = (decision_str == "ALLOW" and proc.returncode == 0)

            diagnostics = lines[1:] if len(lines) > 1 else []

            if is_allowed:
                logger.info("Cedar Authorization Decision: ALLOW (Broadcast permitted)")
            else:
                logger.warning("Cedar Authorization Decision: DENY (Audit: %s)", ", ".join(diagnostics) or "Policy requirements not satisfied")

            return CedarDecision(
                decision=decision_str,
                allowed=is_allowed,
                diagnostics=diagnostics,
                raw_output=stdout,
                context=context
            )

        finally:
            if os.path.exists(temp_req_path):
                try:
                    os.remove(temp_req_path)
                except OSError:
                    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Alert Policy with AWS Cedar")
    parser.add_argument("--depth", type=float, default=28.0, help="Flood depth in cm")
    parser.add_argument("--peak-hours", type=int, default=3, help="Time to peak in hours")
    parser.add_argument("--parcels", type=int, default=5, help="Affected parcels count")
    parser.add_argument("--mode", default="LIVE", choices=["LIVE", "DRY_RUN"], help="System operational mode")

    args = parser.parse_args()

    evaluator = CedarPolicyEvaluator()
    res = evaluator.evaluate(
        flood_depth_cm=args.depth,
        time_to_peak_hours=args.peak_hours,
        affected_parcels_count=args.parcels,
        system_mode=args.mode
    )

    print(f"\nFinal Cedar Result: {res.decision}")
    print(f"Context: {res.context}")
    if res.diagnostics:
        print("Diagnostics:\n" + "\n".join(res.diagnostics))
