from oran_control import (
    ORANGenerator,
    NetworkCondition,
    NetworkRequest,
)


def main():
    generator = (
        ORANGenerator
        .from_pretrained()
    )

    condition = (
        NetworkCondition()
        .use_scheduler("PF")
        .allocate_prbs(
            embb=30,
            urllc=20,
        )
        .set_traffic(
            embb_mbps=12.0,
            urllc_mbps=0.04,
            embb_flows=3,
            urllc_flows=2,
        )
    )

    request = (
        NetworkRequest()
        .require_embb_throughput(
            10.0
        )
        .limit_embb_buffer(
            50.0
        )
    )

    result = generator.generate(
        condition=condition,
        request=request,
    )

    print(
        result.report
    )

    result.save(
        "generated_trace.csv"
    )


if __name__ == "__main__":
    main()
