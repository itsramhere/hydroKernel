"""
Unit and Integration Test Suite for Phase 6 (Isolated MicroVM Dispatcher & Gateway Emulator).
"""

import os
import sys
import unittest
import requests

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from dispatch.dispatcher import (
    MockGatewayServer,
    DialectFormatter,
    MicroVMDispatcher
)


class TestPhase6Dispatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = MockGatewayServer(port=8089)
        cls.server.start()
        cls.dispatcher = MicroVMDispatcher(gateway_url="http://127.0.0.1:8089")

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        self.server.clear()

    def test_dialect_formatter_all_languages(self):
        """Test dialect formatter across all target regional dialects."""
        subdistricts = {
            "Riverbank_North": "as",
            "Eastern_Paddy": "bn",
            "Southern_Slope": "hi",
            "Highland_Ridge": "en"
        }

        for sub_dist, expected_lang in subdistricts.items():
            alert = DialectFormatter.format_alert(
                farmer_name="Test Farmer",
                sub_district=sub_dist,
                depth_cm=35.0,
                peak_hours=4
            )
            self.assertEqual(alert["lang"], expected_lang)
            self.assertIn("35.0", alert["message"])
            self.assertIn("4", alert["message"])

    def test_gateway_server_receipt(self):
        """Test mock gateway receives and stores payloads correctly."""
        url = "http://127.0.0.1:8089/api/v1/sms/send"
        resp = requests.post(url, json={"to": "+919999999999", "message": "Test Alert"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "DELIVERED")

        messages = self.server.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["payload"]["to"], "+919999999999")
        self.assertEqual(messages[0]["channel"], "sms")

    def test_microvm_dispatch_batch(self):
        """Test full batch dispatch to multiple threatened farmers."""
        test_recipients = [
            {
                "farmer_id": "FARMER_001",
                "farmer_name": "Ramesh Das",
                "phone_number": "+919876543210",
                "sub_district": "Riverbank_North"
            },
            {
                "farmer_id": "FARMER_002",
                "farmer_name": "Anita Baruah",
                "phone_number": "+919876543211",
                "sub_district": "Riverbank_North"
            },
            {
                "farmer_id": "FARMER_003",
                "farmer_name": "Biren Kalita",
                "phone_number": "+919876543212",
                "sub_district": "Lowland_Basin"
            }
        ]

        result = self.dispatcher.dispatch_batch(
            alert_id="ALERT_BATCH_001",
            recipients=test_recipients,
            flood_depth_cm=32.0,
            time_to_peak_hours=2,
            channel="both"
        )

        self.assertEqual(result["dispatched_count"], 6)  # 3 farmers * 2 channels
        self.assertEqual(result["error_count"], 0)
        self.assertLess(result["boot_latency_ms"], 10.0)  # MicroVM boot <10ms
        self.assertLess(result["fanout_duration_ms"], 1000.0)  # Fan-out < 1s

        gateway_msgs = self.server.get_messages()
        self.assertEqual(len(gateway_msgs), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
