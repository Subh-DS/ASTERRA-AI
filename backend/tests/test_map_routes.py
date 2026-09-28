import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend.main import app


class FakeCatalog:
    def descriptors(self):
        return [{"name": "public", "label": "Public", "resolution_m": 10, "synthetic": False}]

    def search(self, aoi, provider, max_cloud):
        return {"provider": "public", "resolution_m": 10, "attribution": "test", "items": []}

    def get(self, provider):
        return Mock()


class MapRouteTests(unittest.TestCase):
    def test_imagery_routes_return_contracts(self):
        with patch("backend.main._imagery_catalog", return_value=FakeCatalog()):
            with TestClient(app) as client:
                providers = client.get("/api/imagery/providers")
                self.assertEqual(providers.status_code, 200)
                self.assertEqual(providers.json()["providers"][0]["name"], "public")

                search = client.get(
                    "/api/imagery/search",
                    params={"north": 20.305, "south": 20.295, "east": 85.81, "west": 85.80},
                )
                self.assertEqual(search.status_code, 200)
                self.assertIn("items", search.json())

    def test_map_job_route_persists_map_metadata_and_queues(self):
        fake_job = {"job_id": "a" * 32, "file_token": "token"}
        fake_manager = Mock()
        fake_manager.create.return_value = fake_job
        fake_manager.store.update.return_value = fake_job
        with patch("backend.main._imagery_catalog", return_value=FakeCatalog()), patch("backend.main.manager", fake_manager):
            with TestClient(app) as client:
                response = client.post(
                    "/api/map-jobs",
                    json={
                        "aoi": {"north": 20.305, "south": 20.295, "east": 85.81, "west": 85.80},
                        "provider": "auto",
                        "quality": "medium",
                    },
                )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], fake_job["job_id"])
        fake_manager.store.update.assert_called_once()
        update = fake_manager.store.update.call_args.kwargs
        self.assertEqual(update["source_type"], "map")
        self.assertEqual(update["aoi"]["north"], 20.305)
        fake_manager.start.assert_called_once_with(fake_job["job_id"])


if __name__ == "__main__":
    unittest.main()
