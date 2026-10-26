import os
import sys
import json
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote

# Set base directory to repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Nested challenge folder fallback for imports if needed
CHALLENGE_DIR = os.path.join(BASE_DIR, "statoil-iceberg-classifier-challenge")
if CHALLENGE_DIR not in sys.path:
    sys.path.insert(0, CHALLENGE_DIR)

# Optional PyTorch detection
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    HAS_TORCH = False
    DEVICE = "cpu (analytical-fallback)"

from maritime_engine import MaritimeThreatSimulator, MARITIME_PORTS, ROUTE_PRESETS

SIMULATOR = None
STATIC_DIR = BASE_DIR

def init_server_resources():
    global SIMULATOR
    print("==================================================================")
    print("   MARITIME AI THREAT MONITOR & CYBER-RESILIENT ENGINE SERVER    ")
    print("   Based on: Maritime AI Cyclone & Iceberg Threat Detection       ")
    print(f"   Execution Device: {DEVICE}")
    print(f"   Base Directory:   {BASE_DIR}")
    print("==================================================================")
    
    checkpoint_candidates = [
        os.path.join(BASE_DIR, "best_custom_cnn.pth"),
        os.path.join(CHALLENGE_DIR, "best_custom_cnn.pth")
    ]
    checkpoint_path = None
    for cand in checkpoint_candidates:
        if os.path.exists(cand):
            checkpoint_path = cand
            break

    SIMULATOR = MaritimeThreatSimulator(model_checkpoint_path=checkpoint_path)
    print("[ONLINE] MaritimeThreatSimulator initialized successfully.")

class MaritimeHTTPRequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Universal CORS headers for web APIs and static resources
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Range')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "OK")
        self.end_headers()

    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2 if status_code != 200 else None).encode("utf-8"))

    def _send_file(self, file_path):
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return
            
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".html": "text/html; charset=utf-8",
            ".htm": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".woff": "font/woff",
            ".woff2": "font/woff2",
            ".ttf": "font/ttf",
            ".pdf": "application/pdf"
        }
        content_type = mime_map.get(ext)
        if not content_type:
            content_type, _ = mimetypes.guess_type(file_path)
        if not content_type:
            content_type = "application/octet-stream"
            
        try:
            file_len = os.path.getsize(file_path)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_len))
            self.send_header("Cache-Control", "public, max-age=3600" if ext not in [".html", ".json"] else "no-cache")
            self.end_headers()

            # Stream in 64KB chunks to minimize memory footprint
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            # Client disconnected or stream broken
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

        if path == "/api/status" or path == "/healthz":
            # Health check endpoint for Render & monitoring
            resp = {
                "status": "online",
                "device": DEVICE,
                "torch_available": HAS_TORCH,
                "model": "Statoil Custom CNN + Cyclone LSTM Trajectory Predictor",
                "blockchain_blocks": len(SIMULATOR.blockchain.chain) if SIMULATOR else 0,
                "sim_step": SIMULATOR.step_count if SIMULATOR else 0,
                "sim_hours": round(SIMULATOR.time_hours, 1) if SIMULATOR else 0.0,
                "current_route": {
                    "source": SIMULATOR.source_port_id if SIMULATOR else "CA_STJ",
                    "destination": SIMULATOR.dest_port_id if SIMULATOR else "NL_RTM"
                }
            }
            self._send_json(resp)

        elif path == "/api/state":
            if not SIMULATOR:
                self._send_json({"error": "Simulator uninitialized"}, 500)
                return
            self._send_json(SIMULATOR.get_full_state())

        elif path == "/api/blockchain":
            if not SIMULATOR:
                self._send_json({"error": "Simulator uninitialized"}, 500)
                return
            self._send_json({
                "status": "success",
                "blocks": SIMULATOR.blockchain.get_ledger(),
                "total_blocks": len(SIMULATOR.blockchain.chain),
                "validators": SIMULATOR.blockchain.validators
            })

        elif path == "/api/ports":
            self._send_json({
                "ports": MARITIME_PORTS,
                "presets": ROUTE_PRESETS,
                "current_source": SIMULATOR.source_port_id if SIMULATOR else "CA_STJ",
                "current_dest": SIMULATOR.dest_port_id if SIMULATOR else "NL_RTM"
            })

        elif path == "/api/ablation":
            if not SIMULATOR:
                self._send_json({"error": "Simulator uninitialized"}, 500)
                return
            self._send_json(SIMULATOR.run_ablation_study())

        else:
            # Static file resolution
            if path in ["", "/", "/index", "/index.html"]:
                target_file = os.path.join(STATIC_DIR, "index.html")
            elif path in ["/docs", "/landing", "/documentation", "/landing_documentation.html"]:
                target_file = os.path.join(STATIC_DIR, "landing_documentation.html")
            else:
                clean_path = unquote(parsed.path).lstrip("/").replace("/", os.sep)
                
                # Check root directory first
                candidate = os.path.join(STATIC_DIR, clean_path)
                if os.path.exists(candidate) and not os.path.isdir(candidate):
                    target_file = candidate
                else:
                    # Fallback checks
                    fallbacks = [
                        os.path.join(STATIC_DIR, "assets", clean_path),
                        os.path.join(CHALLENGE_DIR, "web", clean_path),
                        os.path.join(CHALLENGE_DIR, clean_path)
                    ]
                    target_file = candidate
                    for fb in fallbacks:
                        if os.path.exists(fb) and not os.path.isdir(fb):
                            target_file = fb
                            break

            self._send_file(target_file)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        content_len = int(self.headers.get('Content-Length', 0))
        post_body = self.rfile.read(content_len) if content_len > 0 else b"{}"

        try:
            payload = json.loads(post_body.decode('utf-8')) if post_body else {}
        except Exception:
            payload = {}

        if not SIMULATOR:
            self._send_json({"error": "Simulator uninitialized"}, 500)
            return

        if path == "/api/step":
            dt = float(payload.get("dt_hours", 0.5))
            SIMULATOR.step(dt_hours=dt)
            self._send_json({
                "status": "stepped",
                "step_count": SIMULATOR.step_count,
                "sim_time_hours": SIMULATOR.time_hours,
                "state": SIMULATOR.get_full_state()
            })

        elif path == "/api/reset":
            scenario = payload.get("scenario", "Grand Banks Iceberg Belt & Cyclone Collision Course")
            SIMULATOR.scenario_name = scenario
            SIMULATOR.step_count = 0
            SIMULATOR.time_hours = 0.0
            SIMULATOR._init_entities()
            self._send_json({
                "status": "reset",
                "scenario": scenario,
                "state": SIMULATOR.get_full_state()
            })

        elif path == "/api/set_route":
            source = payload.get("source") or payload.get("source_id") or "CA_STJ"
            dest = payload.get("dest") or payload.get("dest_id") or payload.get("destination") or "NL_RTM"
            SIMULATOR.set_route(source, dest)
            self._send_json({
                "status": "route_updated",
                "source": SIMULATOR.source_port_id,
                "destination": SIMULATOR.dest_port_id,
                "state": SIMULATOR.get_full_state()
            })

        elif path == "/api/classify":
            b1 = payload.get("band_1")
            b2 = payload.get("band_2")
            s_id = payload.get("id", "target_live")
            if not b1 or not b2 or len(b1) != 5625 or len(b2) != 5625:
                self._send_json({"status": "error", "message": "Expected band_1 and band_2 with 5625 floats each"}, 400)
                return
            res = SIMULATOR.classify_sar_sample(b1, b2, sample_id=s_id)
            self._send_json(res)

        elif path == "/api/cyberattack":
            attack_type = payload.get("type", "GPS_SPOOF")
            enabled = bool(payload.get("enabled", True))
            
            if attack_type == "GPS_SPOOF":
                d_lat = float(payload.get("d_lat", 0.85))
                d_lon = float(payload.get("d_lon", -1.2))
                SIMULATOR.cyber_manager.set_gps_spoof(enabled, d_lat=d_lat, d_lon=d_lon)
            elif attack_type == "WIND_SCALING":
                alpha = float(payload.get("alpha", 1.95))
                SIMULATOR.cyber_manager.set_wind_scaling(enabled, alpha=alpha)
            elif attack_type == "REPLAY_ATTACK":
                SIMULATOR.cyber_manager.set_replay_attack(enabled)
            elif attack_type == "CLEAR_ALL":
                SIMULATOR.cyber_manager.set_gps_spoof(False)
                SIMULATOR.cyber_manager.set_wind_scaling(False)
                SIMULATOR.cyber_manager.set_replay_attack(False)
                
            SIMULATOR.recompute_all()
            self._send_json({
                "status": "updated",
                "cyber_state": SIMULATOR.threat_state.get("cyber_status", {})
            })

        elif path == "/api/replan":
            SIMULATOR.recompute_all()
            self._send_json({
                "status": "replanned",
                "safe_route": SIMULATOR.safe_route,
                "threat_state": SIMULATOR.threat_state
            })

        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8000, host="0.0.0.0"):
    init_server_resources()
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, MaritimeHTTPRequestHandler)
    print("==================================================================")
    print(" [ONLINE] Maritime AI Real-Time System running for Render:")
    print(f" -> http://{host}:{port}/")
    print(f" -> Healthcheck: http://{host}:{port}/api/status")
    print("==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Maritime Server.")
        httpd.server_close()

if __name__ == "__main__":
    # Render binds dynamic port via $PORT
    port_env = os.environ.get("PORT") or os.environ.get("SERVER_PORT")
    port = int(port_env) if port_env else 8000
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    run_server(port=port, host="0.0.0.0")
