"""Build balanced L2 clusters and LVQ6 centroid files for the website."""

import argparse
import json

from balanced_l2.build import build


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="*", default=["vectors"])
    parser.add_argument("--output", default="clusters")
    parser.add_argument("--clusters", type=int, default=64000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=250000)
    parser.add_argument("--sample-size", type=int, default=2500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--cpu-assignment", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--brotli-centroids", action="store_true",
                        help="Require a whole Brotli-11 centroid index below 25 MiB before full assignment")
    parser.add_argument("--heads-from",
                        help="Reuse validated native heads from the matching completed centroid comparison")
    args = parser.parse_args()
    result = build(args.input, args.output, args.clusters, dim=args.dim, threads=args.threads,
                   batch_size=args.batch_size, sample_size=args.sample_size, seed=args.seed,
                   max_iter=args.max_iter, gpu=not args.cpu_assignment, resume=args.resume,
                   export_only=args.export_only, centroid_brotli=args.brotli_centroids,
                   heads_from=args.heads_from)
    print(json.dumps(result, indent=2))
