"""Unit and integration tests for the AI Business Intelligence Report Assistant."""

import http.cookiejar
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import server as srv
from Ai_assistant import BusinessIntelligenceModel

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


class TestBusinessIntelligenceModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = BusinessIntelligenceModel()
        self.model.ingest_file(SAMPLES_DIR / "regional_sales_q1_q4.csv")
        self.model.ingest_file(SAMPLES_DIR / "annual_business_performance_report.md")

    def test_ingestion_counts(self) -> None:
        self.assertEqual(len(self.model.documents), 2)
        self.assertEqual(len(self.model.records), 32)
        self.assertGreater(len(self.model.chunks), 32)

    def test_best_performing_region_challenge(self) -> None:
        result = self.model.identify_best_region()
        self.assertEqual(result["best_region"], "North America")
        self.assertGreater(result["metrics"]["revenue"], 3_000_000)
        reasons_joined = " ".join(result["reasons"])
        self.assertIn("North America", reasons_joined)
        self.assertTrue(
            any("94% enterprise customer retention" in r or "bundled Cloud Suite" in r for r in result["reasons"]),
            f"Expected narrative reasons in {result['reasons']}",
        )

    def test_retrieval_and_qa(self) -> None:
        ans = self.model.answer_question("What operational risks or bottlenecks are mentioned?")
        self.assertIn("Operational Risks", ans["answer"])
        self.assertGreater(len(ans["evidence"]), 0)

    def test_summary_and_recommendations(self) -> None:
        analysis = self.model.full_analysis()
        kpis = analysis["summary"]["kpis"]
        self.assertEqual(kpis["documents_loaded"], 2)
        self.assertGreater(kpis["total_revenue"], 0)
        self.assertGreater(kpis["profit_margin_pct"], 0)
        self.assertGreaterEqual(len(analysis["recommendations"]), 3)

    def test_json_report_ingestion(self) -> None:
        m = BusinessIntelligenceModel()
        payload = json.dumps([
            {"Region": "EMEA", "Revenue": "$500,000", "Profit": "$150,000", "Category": "Core", "Growth": "14%"},
            {"Region": "APAC", "Revenue": "$800,000", "Profit": "$320,000", "Category": "AI", "Growth": "28%"},
        ])
        info = m.ingest_text("summary.json", payload)
        self.assertEqual(info["records_added"], 2)
        best = m.identify_best_region()
        self.assertEqual(best["best_region"], "APAC")


class TestWebServerAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        srv.USERS = srv.UserStore(Path(self.tmp_dir.name) / "test_users.db")
        srv.SESSIONS = srv.SessionStore()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.BIRequestHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp_dir.cleanup()

    @staticmethod
    def _new_client() -> urllib.request.OpenerDirector:
        jar = http.cookiejar.CookieJar()
        return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def test_health_and_auth_isolation(self) -> None:
        # Health check and HTML index should return 200 without auth
        with urllib.request.urlopen(f"{self.base}/health") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(json.loads(resp.read().decode("utf-8")), {"status": "ok"})

        with urllib.request.urlopen(f"{self.base}/") as resp:
            self.assertEqual(resp.status, 200)
            html = resp.read().decode("utf-8")
            self.assertIn("1-Click Demo Workspace", html)

        # Invalid JSON on login should return 400
        bad_req = urllib.request.Request(
            f"{self.base}/api/auth/login",
            data=b"{not-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as bad_ctx:
            urllib.request.urlopen(bad_req)
        self.assertEqual(bad_ctx.exception.code, 400)
        bad_ctx.exception.close()

        # Unauthenticated API call should return 401
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(f"{self.base}/api/analysis")
        self.assertEqual(ctx.exception.code, 401)
        ctx.exception.close()

        # User A registers and loads sample reports
        client_a = self._new_client()
        reg_req = urllib.request.Request(
            f"{self.base}/api/auth/register",
            data=json.dumps({"username": "alice", "password": "password123"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with client_a.open(reg_req) as resp:
            self.assertTrue(json.loads(resp.read().decode("utf-8"))["authenticated"])

        load_req = urllib.request.Request(f"{self.base}/api/load-samples", data=b"{}", method="POST")
        with client_a.open(load_req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["analysis"]["best_region_analysis"]["best_region"], "North America")

        q_req = urllib.request.Request(
            f"{self.base}/api/query",
            data=json.dumps({"query": "Which region performed best and why?"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with client_a.open(q_req) as resp:
            q_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(q_data["best_region"], "North America")

        # User B uses Demo login and must have an empty isolated workspace
        client_b = self._new_client()
        demo_req = urllib.request.Request(f"{self.base}/api/auth/demo", data=b"{}", method="POST")
        with client_b.open(demo_req) as resp:
            self.assertTrue(json.loads(resp.read().decode("utf-8"))["is_demo"])

        with client_b.open(f"{self.base}/api/analysis") as resp:
            b_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(b_data["documents"]), 0)
            self.assertFalse(b_data["summary"]["has_data"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
