"""Serve the source client locally, with native HTTP Brotli for centroid indexes."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class LocalHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        path = Path(self.translate_path(self.path))
        if path.suffix == '.br' and path.is_file():
            self.send_header('Content-Encoding', 'br')
            self.send_header('Cache-Control', 'no-cache, no-transform')
        super().end_headers()

    def guess_type(self, path):
        return 'application/octet-stream' if path.endswith('.br') else super().guess_type(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(LocalHandler, directory=str(root)))
    print(f'Serving {root} at http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
