"""Start the local translator and open its web interface."""
import argparse
import webbrowser
from http.server import ThreadingHTTPServer

from src.webapp import Handler, TRANSLATOR


def main():
    parser = argparse.ArgumentParser(description="Lancer le traducteur local")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Traducteur : {url} (modèle chargé : {TRANSLATOR is not None})")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Arrêt du serveur.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
