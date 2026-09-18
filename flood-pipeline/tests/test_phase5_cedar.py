"""
Unit and Integration Test Suite for Phase 5 (AWS Cedar Policy Engine).
Tests statutory criteria, safeguard policies, and context evaluation.
"""

import os
import sys
import unittest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from policies.evaluator import CedarPolicyEvaluator


class TestPhase5CedarPolicy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.evaluator = CedarPolicyEvaluator()

    def test_valid_broadcast_is_allowed(self):
        """Test statutory conditions met in LIVE mode: should return ALLOW."""
        res = self.evaluator.evaluate(
            flood_depth_cm=25.0,
            time_to_peak_hours=3,
            affected_parcels_count=12,
            system_mode="LIVE"
        )
        self.assertTrue(res.allowed)
        self.assertEqual(res.decision, "ALLOW")

    def test_dry_run_is_forbidden(self):
        """Test safeguard rule: DRY_RUN mode must be explicitly forbidden."""
        res = self.evaluator.evaluate(
            flood_depth_cm=30.0,
            time_to_peak_hours=2,
            affected_parcels_count=5,
            system_mode="DRY_RUN"
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.decision, "DENY")

    def test_shallow_depth_is_denied(self):
        """Test threshold rule: flood depth < 15 cm must not trigger broadcast."""
        res = self.evaluator.evaluate(
            flood_depth_cm=14.0,
            time_to_peak_hours=2,
            affected_parcels_count=10,
            system_mode="LIVE"
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.decision, "DENY")

    def test_distant_time_to_peak_is_denied(self):
        """Test arrival rule: time to peak > 6 hours must not trigger broadcast."""
        res = self.evaluator.evaluate(
            flood_depth_cm=40.0,
            time_to_peak_hours=7,
            affected_parcels_count=10,
            system_mode="LIVE"
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.decision, "DENY")

    def test_zero_affected_parcels_is_denied(self):
        """Test affected parcels rule: 0 affected parcels must not trigger broadcast."""
        res = self.evaluator.evaluate(
            flood_depth_cm=50.0,
            time_to_peak_hours=1,
            affected_parcels_count=0,
            system_mode="LIVE"
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.decision, "DENY")

    def test_unauthorized_principal_is_denied(self):
        """Test authorization guard: calling principal other than MLHydrologyAgent must be denied."""
        res = self.evaluator.evaluate(
            flood_depth_cm=30.0,
            time_to_peak_hours=2,
            affected_parcels_count=5,
            system_mode="LIVE",
            principal='Role::"UnauthorizedAgent"'
        )
        self.assertFalse(res.allowed)
        self.assertEqual(res.decision, "DENY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
