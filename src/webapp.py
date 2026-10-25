"""Local web interface for French text to a randomly selected target language."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse

from .translation import CHECKPOINT, LANG_NAMES, MultilingualTranslator

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
try:
    TRANSLATOR = MultilingualTranslator(CHECKPOINT)
except (FileNotFoundError, RuntimeError, KeyError):
    TRANSLATOR = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            langs = list(TRANSLATOR.targets_from_french) if TRANSLATOR else []
            self.respond({"model_ready": TRANSLATOR is not None,
                          "targets": [{"code": x, "name": LANG_NAMES[x]} for x in langs]})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/translate":
            self.respond({"error": "Route inconnue."}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.respond({"error": "Requête invalide."}, 400)
            return
        if length < 1 or length > 16_384:
            self.respond({"error": "Requête vide ou trop volumineuse."}, 413)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.respond({"error": "JSON invalide."}, 400)
            return
        if not isinstance(payload, dict) or "target_lang" in payload or "language" in payload:
            self.respond({"error": "La langue cible est choisie automatiquement."}, 400)
            return
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            self.respond({"error": "Saisis un texte à traduire."}, 400)
            return
        if len(text) > 2000:
            self.respond({"error": "Texte trop long (limite : 2 000 caractères)."}, 413)
            return
        if TRANSLATOR is None:
            self.respond({"error": "Modèle absent. Prépare les données et entraîne le Transformer."}, 503)
            return
        try:
            result = TRANSLATOR.random_translation(text)
            self.respond(result)
        except ValueError as exc:
            self.respond({"error": str(exc)}, 422)

    def respond(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traducteur expérimental français multilingue")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Application : http://127.0.0.1:{args.port} (modèle chargé : {TRANSLATOR is not None})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
