import unittest
from unittest.mock import patch

from travel_agent.schemas import PlanRequest, PlanningMode
from travel_agent.workflow import (
    build_initial_state,
    complete_pending_segment_node,
    distance_calculator_node,
    fallback_local_spot_node,
    mid_segment_rest_node,
    route_from_distance,
)


class WorkflowDistanceGuardTests(unittest.TestCase):
    def _base_state(self):
        request = PlanRequest(
            mode=PlanningMode.LOCAL_EXPLORE,
            city="Hangzhou",
            destination="West Lake",
            origin="",
            trip_duration_days=1,
            interests=["parks"],
            fatigue_threshold_meters=3000,
            max_spots=3,
            selected_waypoints=[],
        )
        state = build_initial_state(request)
        state["anchor_location"] = "West Lake"
        state["itinerary_items"] = [
            {
                "id": "spot-test-1",
                "name": "Far Spot",
                "kind": "spot",
                "sequence": 1,
                "anchor_segment": "West Lake",
                "distance_from_previous_meters": 0,
                "cumulative_distance_meters": 0,
                "reason": "test",
                "confirmed": False,
            }
        ]
        return state

    def test_rejects_overlong_segment_and_requests_replan(self):
        state = self._base_state()
        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.route.return_value = {"distance_meters": 7000}
            updates = distance_calculator_node(state)

        self.assertTrue(updates["replan_required"])
        self.assertEqual(updates["distance_replan_count"], 1)
        self.assertEqual(updates["itinerary_items"], [])
        self.assertIn("Far Spot", updates["rejected_spot_names"])
        self.assertEqual(updates["emitted_events"][0]["event"], "itinerary_item_rejected")

    def test_accepts_segment_within_limit_and_updates_total_distance(self):
        state = self._base_state()
        state["total_distance_meters"] = 200
        state["cumulative_distance_meters"] = 100

        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.route.return_value = {
                "distance_meters": 2500,
                "origin_coords": "120.100000,30.100000",
                "destination_coords": "120.120000,30.120000",
            }
            updates = distance_calculator_node(state)

        self.assertFalse(updates["replan_required"])
        self.assertEqual(updates["distance_replan_count"], 0)
        self.assertEqual(updates["current_segment_distance_meters"], 2500)
        self.assertEqual(updates["cumulative_distance_meters"], 2600)
        self.assertEqual(updates["total_distance_meters"], 2700)
        self.assertEqual(updates["emitted_events"][0]["event"], "segment_distance_updated")

    def test_long_but_reachable_segment_requires_mid_segment_rest(self):
        state = self._base_state()
        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.route.return_value = {
                "distance_meters": 5000,
                "origin_coords": "120.100000,30.100000",
                "destination_coords": "120.200000,30.200000",
            }
            updates = distance_calculator_node(state)

        self.assertEqual(updates["itinerary_items"], [])
        self.assertEqual(updates["pending_destination_item"]["name"], "Far Spot")
        self.assertEqual(updates["pending_remaining_distance_meters"], 2000)
        self.assertEqual(updates["emitted_events"][0]["event"], "mid_segment_rest_required")

    def test_mid_segment_rest_node_inserts_rest_stop(self):
        state = self._base_state()
        state["itinerary_items"] = []
        state["pending_destination_item"] = {
            "id": "spot-test-1",
            "name": "Far Spot",
            "kind": "spot",
            "sequence": 1,
            "anchor_segment": "West Lake",
            "distance_from_previous_meters": 0,
            "cumulative_distance_meters": 0,
            "reason": "test",
            "confirmed": False,
        }
        state["pending_midpoint_coords"] = "120.150000,30.150000"
        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.nearby_pois_by_location.return_value = [
                {"name": "Mid Cafe", "address": "X", "distance_meters": 180}
            ]
            updates = mid_segment_rest_node(state)

        self.assertEqual(updates["itinerary_items"][0]["kind"], "rest_stop")
        self.assertEqual(updates["itinerary_items"][0]["distance_from_previous_meters"], 3000)
        self.assertEqual(updates["total_distance_meters"], 3000)
        self.assertEqual(updates["emitted_events"][0]["event"], "rest_stop_added")

    def test_complete_pending_segment_appends_destination_after_rest(self):
        state = self._base_state()
        state["itinerary_items"] = [
            {
                "id": "rest-mid",
                "name": "Mid Cafe",
                "kind": "rest_stop",
                "sequence": 1,
                "anchor_segment": "West Lake",
                "distance_from_previous_meters": 3000,
                "cumulative_distance_meters": 3000,
                "reason": "rest",
                "confirmed": True,
            }
        ]
        state["total_distance_meters"] = 3000
        state["pending_destination_item"] = {
            "id": "spot-test-1",
            "name": "Far Spot",
            "kind": "spot",
            "sequence": 1,
            "anchor_segment": "West Lake",
            "distance_from_previous_meters": 0,
            "cumulative_distance_meters": 0,
            "reason": "test",
            "confirmed": False,
        }
        state["pending_remaining_distance_meters"] = 1800
        updates = complete_pending_segment_node(state)

        self.assertEqual(updates["itinerary_items"][-1]["name"], "Far Spot")
        self.assertEqual(updates["itinerary_items"][-1]["distance_from_previous_meters"], 1800)
        self.assertEqual(updates["cumulative_distance_meters"], 1800)
        self.assertEqual(updates["total_distance_meters"], 4800)
        self.assertEqual(updates["emitted_events"][0]["event"], "segment_distance_updated")

    def test_stops_replanning_after_retry_limit(self):
        state = self._base_state()
        state["distance_replan_count"] = 2
        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.route.return_value = {"distance_meters": 9000}
            updates = distance_calculator_node(state)

        self.assertFalse(updates["replan_required"])
        self.assertTrue(updates["distance_guard_exhausted"])
        self.assertEqual(updates["distance_replan_count"], 3)
        self.assertEqual(updates["itinerary_items"], [])
        self.assertEqual(updates["emitted_events"][0]["event"], "itinerary_item_rejected")

    def test_route_from_distance_goes_to_fallback_when_exhausted(self):
        state = self._base_state()
        state["distance_guard_exhausted"] = True
        state["fallback_attempted"] = False
        self.assertEqual(route_from_distance(state), "fallback_local_spot")

    def test_fallback_local_spot_selects_nearby_candidate(self):
        state = self._base_state()
        state["itinerary_items"] = []
        with patch("travel_agent.workflow.AMapClient") as mock_client:
            mock_client.return_value.nearby_pois.return_value = [
                {"name": "Hubin Park", "address": "West Lake", "distance_meters": 350}
            ]
            updates = fallback_local_spot_node(state)

        self.assertEqual(len(updates["itinerary_items"]), 1)
        self.assertEqual(updates["itinerary_items"][0]["name"], "Hubin Park")
        self.assertTrue(updates["fallback_attempted"])
        self.assertFalse(updates["distance_guard_exhausted"])
        self.assertEqual(updates["emitted_events"][0]["event"], "fallback_spot_selected")


if __name__ == "__main__":
    unittest.main()
