import argparse
from pathlib import Path

from .condition import NetworkCondition
from .request import NetworkRequest
from .generator import ORANGenerator


def build_parser():
    parser = argparse.ArgumentParser(
        prog="oran-control",
        description=(
            "Generate Open RAN KPM traces from operator conditions "
            "and executable service constraints."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    generate = subparsers.add_parser(
        "generate",
        help="Generate and verify an Open RAN KPM trace.",
    )

    generate.add_argument(
        "--scheduler",
        choices=["PF", "RR"],
        required=True,
    )

    generate.add_argument(
        "--embb-prb",
        type=int,
        required=True,
    )

    generate.add_argument(
        "--urllc-prb",
        type=int,
        required=True,
    )

    generate.add_argument(
        "--embb-traffic",
        type=float,
        required=True,
        help="Offered eMBB traffic in Mbps.",
    )

    generate.add_argument(
        "--urllc-traffic",
        type=float,
        required=True,
        help="Offered URLLC traffic in Mbps.",
    )

    generate.add_argument(
        "--embb-flows",
        type=int,
        required=True,
    )

    generate.add_argument(
        "--urllc-flows",
        type=int,
        required=True,
    )

    generate.add_argument(
        "--min-throughput",
        type=float,
        default=None,
        help="Minimum mean eMBB throughput in Mbps.",
    )

    generate.add_argument(
        "--max-buffer-kb",
        type=float,
        default=None,
        help="Maximum mean eMBB buffer in KB.",
    )

    generate.add_argument(
        "--output",
        type=str,
        default="generated_trace.csv",
    )

    generate.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser


def run_generate(args):
    condition = (
        NetworkCondition()
        .use_scheduler(
            args.scheduler
        )
        .allocate_prbs(
            embb=args.embb_prb,
            urllc=args.urllc_prb,
        )
        .set_traffic(
            embb_mbps=args.embb_traffic,
            urllc_mbps=args.urllc_traffic,
            embb_flows=args.embb_flows,
            urllc_flows=args.urllc_flows,
        )
    )

    request = NetworkRequest()

    if args.min_throughput is not None:
        request.require_embb_throughput(
            args.min_throughput
        )

    if args.max_buffer_kb is not None:
        request.limit_embb_buffer(
            args.max_buffer_kb
        )

    if not request.constraints:
        raise SystemExit(
            "At least one service constraint is required."
        )

    generator = ORANGenerator.from_pretrained()

    result = generator.generate(
        condition=condition,
        request=request,
        seed=args.seed,
    )

    output_path = result.save(
        Path(
            args.output
        )
    )

    print(
        result.report
    )

    print()

    print(
        "Saved trace:",
        output_path,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate":
        run_generate(args)
        return

    parser.error(
        "Unknown command."
    )


if __name__ == "__main__":
    main()
