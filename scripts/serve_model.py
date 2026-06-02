import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ask2know.deployment import load_deployment_bundle, predict_with_loaded_bundle


def make_handler(model, weights, bundle, model_path, default_top_k):
    class PredictHandler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/health':
                self._send_json(200, {'ok': True, 'model': str(Path(model_path).expanduser().resolve())})
                return
            if parsed.path == '/predict':
                query = parse_qs(parsed.query)
                image = (query.get('image') or [''])[0]
                top_k = int((query.get('top_k') or [default_top_k])[0])
                self._predict(image, top_k)
                return
            self._send_json(404, {'error': 'Not found. Use GET /health or POST /predict.'})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != '/predict':
                self._send_json(404, {'error': 'Not found. Use POST /predict.'})
                return
            length = int(self.headers.get('Content-Length', '0') or '0')
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode('utf-8')) if raw else {}
            except json.JSONDecodeError:
                self._send_json(400, {'error': 'Invalid JSON body.'})
                return
            image = payload.get('image') or payload.get('image_path')
            top_k = int(payload.get('top_k') or default_top_k)
            self._predict(image, top_k)

        def _predict(self, image, top_k):
            if not image:
                self._send_json(400, {'error': 'Missing image path.'})
                return
            image_path = Path(image).expanduser()
            if not image_path.exists():
                self._send_json(404, {'error': f'Image not found: {image}'})
                return
            try:
                result = predict_with_loaded_bundle(model, weights, bundle, model_path, image_path, top_k=top_k)
            except Exception as exc:
                self._send_json(500, {'error': str(exc)})
                return
            self._send_json(200, result)

        def log_message(self, fmt, *args):
            print('%s - %s' % (self.address_string(), fmt % args))

    return PredictHandler


def main():
    parser = argparse.ArgumentParser(description='Serve an exported Ask2Know model bundle over a small local HTTP API.')
    parser.add_argument('--model', required=True, help='Path to .a2kmodel.json')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host')
    parser.add_argument('--port', type=int, default=8000, help='Bind port')
    parser.add_argument('--top-k', type=int, default=5, help='Default number of predictions to return')
    parser.add_argument('--cache-dir', help='Optional CLIP feature cache directory for deployment runtime')
    args = parser.parse_args()

    model, weights, bundle = load_deployment_bundle(args.model, cache_dir=args.cache_dir)
    handler = make_handler(model, weights, bundle, args.model, args.top_k)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f'Serving Ask2Know model at http://{args.host}:{args.port}')
    print('Health: GET /health')
    print('Predict: POST /predict with JSON {"image": "path/to/image.jpg", "top_k": 5}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('Stopping server.')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
