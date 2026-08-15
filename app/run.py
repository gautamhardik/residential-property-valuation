"""Start the valuation app on a free local port.

This avoids stale port collisions when a previous UVicorn process is still running.
It chooses the requested port if available, otherwise increments until it finds a free one.
"""

import os
import socket
import subprocess
import sys


def find_free_port(preferred: int, max_port: int = 8100) -> int:
    port = preferred
    while port <= max_port:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"No free port found between {preferred} and {max_port}.")


def main() -> None:
    preferred = int(os.getenv("PORT", "8000"))
    port = find_free_port(preferred)

    if preferred != port:
        print(f"Port {preferred} was busy; starting the app on http://127.0.0.1:{port}")
    else:
        print(f"Starting API on http://127.0.0.1:{port}")

    env = os.environ.copy()
    env["PORT"] = str(port)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.index:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if "--reload" in sys.argv[1:]:
        cmd.append("--reload")

    try:
        raise SystemExit(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:
        print("\nStopping app.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
