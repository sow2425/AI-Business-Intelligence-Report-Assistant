"""Zero-dependency HTTP server for the AI Business Intelligence Report Assistant.

Production-ready for Render and multi-visitor online hosting:
- Binds to 0.0.0.0:$PORT (configurable via HOST / PORT env vars)
- Health check endpoint at GET /health
- Per-session BusinessIntelligenceModel isolation via HttpOnly cookie
- Lightweight SQLite user auth (PBKDF2-SHA256) + 1-click Demo login
- Request size limits (5 MB) and security headers
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from Ai_assistant import BusinessIntelligenceModel

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "samples"
DEFAULT_DB_PATH = Path(os.environ.get("BI_DB_PATH", str(BASE_DIR / "bi_users.db")))

MAX_BODY_BYTES = 5 * 1024 * 1024  # 5 MB upload limit
MAX_SESSIONS = 200
SESSION_COOKIE = "bi_session"
SESSION_TTL_SECONDS = 60 * 60 * 12  # 12 hours


class PayloadTooLargeError(Exception):
    """Raised when an HTTP request body exceeds MAX_BODY_BYTES."""


class SessionStore:
    """Thread-safe bounded store mapping session tokens to user & BI model state."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_sessions = max_sessions

    def create(self, username: str, is_demo: bool = False) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge_expired()
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)
            self._sessions[token] = {
                "username": username,
                "is_demo": is_demo,
                "model": BusinessIntelligenceModel(),
                "expires_at": time.time() + SESSION_TTL_SECONDS,
            }
        return token

    def get(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self._lock:
            entry = self._sessions.get(token)
            if not entry:
                return None
            if entry["expires_at"] < time.time():
                del self._sessions[token]
                return None
            entry["expires_at"] = time.time() + SESSION_TTL_SECONDS
            self._sessions.move_to_end(token)
            return entry

    def delete(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if v["expires_at"] < now]
        for k in expired:
            del self._sessions[k]


class UserStore:
    """SQLite-backed user store with salted PBKDF2-SHA256 password hashing."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            120_000,
        )
        return digest.hex()

    def register(self, username: str, password: str) -> tuple[bool, str]:
        clean_user = username.strip().lower()
        if len(clean_user) < 3 or len(clean_user) > 32:
            return False, "Username must be between 3 and 32 characters."
        if len(password) < 6:
            return False, "Password must be at least 6 characters."

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)
        try:
            with self._lock, closing(self._connect()) as conn:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                    (clean_user, pw_hash, salt, time.time()),
                )
                conn.commit()
            return True, clean_user
        except sqlite3.IntegrityError:
            return False, "Username already exists."

    def verify(self, username: str, password: str) -> bool:
        clean_user = username.strip().lower()
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT password_hash, salt FROM users WHERE username = ?",
                (clean_user,),
            ).fetchone()
        if not row:
            return False
        expected_hash, salt = row
        candidate_hash = self._hash_password(password, salt)
        return hmac.compare_digest(expected_hash, candidate_hash)


SESSIONS = SessionStore()
USERS = UserStore()


class BIRequestHandler(BaseHTTPRequestHandler):
    """Serves the BI dashboard UI, auth endpoints, and per-session BI API."""

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")

    def _send_json(
        self,
        payload: Any,
        status: int = HTTPStatus.OK,
        set_cookie: str | None = None,
        clear_cookie: bool = False,
    ) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self._send_security_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={set_cookie}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_TTL_SECONDS}",
            )
        elif clear_cookie:
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0",
            )
        self.end_headers()
        self.wfile.write(body)

    def _get_session_token(self) -> str | None:
        cookie_header = self.headers.get("Cookie")
        if not cookie_header:
            return None
        cookie = SimpleCookie()
        cookie.load(cookie_header)
        morsel = cookie.get(SESSION_COOKIE)
        return morsel.value if morsel else None

    def _require_session(self) -> dict[str, Any] | None:
        session = SESSIONS.get(self._get_session_token())
        if not session:
            self._send_json({"error": "Authentication required"}, status=HTTPStatus.UNAUTHORIZED)
            return None
        return session

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            raise PayloadTooLargeError("Payload exceeds 5 MB limit")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise json.JSONDecodeError("Expected JSON object", raw.decode("utf-8"), 0)
        return data

    def _resolve_index_file(self) -> Path | None:
        for candidate in (STATIC_DIR / "index.html", BASE_DIR / "index.html"):
            if candidate.is_file():
                return candidate
        return None

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"status": "ok"})
            return

        if self.path in ("/", "/index.html"):
            index_file = self._resolve_index_file()
            if not index_file:
                self.send_error(HTTPStatus.NOT_FOUND, "UI file not found. Make sure static/index.html is uploaded.")
                return
            data = index_file.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._send_security_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/api/auth/me":
            session = SESSIONS.get(self._get_session_token())
            if not session:
                self._send_json({"authenticated": False})
                return
            self._send_json({
                "authenticated": True,
                "username": session["username"],
                "is_demo": session["is_demo"],
            })
            return

        if self.path == "/api/analysis":
            session = self._require_session()
            if not session:
                return
            self._send_json(session["model"].full_analysis())
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/auth/register":
                payload = self._read_json_body()
                username = str(payload.get("username", ""))
                password = str(payload.get("password", ""))
                ok, msg = USERS.register(username, password)
                if not ok:
                    self._send_json({"error": msg}, status=HTTPStatus.BAD_REQUEST)
                    return
                token = SESSIONS.create(msg, is_demo=False)
                self._send_json(
                    {"authenticated": True, "username": msg, "is_demo": False},
                    set_cookie=token,
                )
                return

            if self.path == "/api/auth/login":
                payload = self._read_json_body()
                username = str(payload.get("username", "")).strip().lower()
                password = str(payload.get("password", ""))
                if not USERS.verify(username, password):
                    self._send_json(
                        {"error": "Invalid username or password."},
                        status=HTTPStatus.UNAUTHORIZED,
                    )
                    return
                token = SESSIONS.create(username, is_demo=False)
                self._send_json(
                    {"authenticated": True, "username": username, "is_demo": False},
                    set_cookie=token,
                )
                return

            if self.path == "/api/auth/demo":
                token = SESSIONS.create("demo-guest", is_demo=True)
                self._send_json(
                    {"authenticated": True, "username": "demo-guest", "is_demo": True},
                    set_cookie=token,
                )
                return

            if self.path == "/api/auth/logout":
                SESSIONS.delete(self._get_session_token())
                self._send_json({"authenticated": False}, clear_cookie=True)
                return

            session = self._require_session()
            if not session:
                return
            model: BusinessIntelligenceModel = session["model"]

            if self.path == "/api/upload":
                payload = self._read_json_body()
                files = payload.get("files", [])
                if payload.get("replace", False):
                    model.clear()
                ingested = []
                for item in files:
                    name = item.get("name", "report.txt")
                    content = item.get("content", "")
                    ingested.append(model.ingest_text(name, content))
                self._send_json({"ingested": ingested, "analysis": model.full_analysis()})
                return

            if self.path == "/api/load-samples":
                model.clear()
                ingested = []
                if SAMPLES_DIR.exists():
                    for file_path in sorted(SAMPLES_DIR.iterdir()):
                        if file_path.is_file():
                            ingested.append(model.ingest_file(file_path))
                if not ingested:
                    for name in ("regional_sales_q1_q4.csv", "annual_business_performance_report.md"):
                        root_sample = BASE_DIR / name
                        if root_sample.is_file():
                            ingested.append(model.ingest_file(root_sample))
                self._send_json({"ingested": ingested, "analysis": model.full_analysis()})
                return

            if self.path == "/api/clear":
                model.clear()
                self._send_json({"analysis": model.full_analysis()})
                return

            if self.path == "/api/query":
                payload = self._read_json_body()
                query = str(payload.get("query", "")).strip()
                if not query:
                    self._send_json({"error": "Query cannot be empty"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._send_json(model.answer_question(query))
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
        except PayloadTooLargeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON payload"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        return


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def run_server(host: str | None = None, port: int | None = None) -> None:
    bind_host = host or os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("RENDER") or os.environ.get("PORT") else "127.0.0.1")
    bind_port = port or int(os.environ.get("PORT", "8000"))
    server = ReusableThreadingHTTPServer((bind_host, bind_port), BIRequestHandler)
    display_host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
    print(f"AI Business Intelligence Report Assistant running at http://{display_host}:{bind_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    run_server()
