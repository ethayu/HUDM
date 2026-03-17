from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hudm.experiment_review import serve_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Review experiment results and render replay media on demand.")
    parser.add_argument("--run-dir", required=True, help="Experiment output directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    args = parser.parse_args()
    serve_review(args.run_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
