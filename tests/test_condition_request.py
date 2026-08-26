import pytest

from oran_control import (
    NetworkCondition,
    NetworkRequest,
)


def test_valid_condition():
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

    condition.validate()


def test_invalid_prb_total():
    # Validation may occur eagerly in allocate_prbs() or later in validate().
    # Both behaviors are correct, so wrap the full construction/validation.
    with pytest.raises(ValueError):
        condition = (
            NetworkCondition()
            .use_scheduler("PF")
            .allocate_prbs(
                embb=30,
                urllc=30,
            )
            .set_traffic(
                embb_mbps=12.0,
                urllc_mbps=0.04,
                embb_flows=3,
                urllc_flows=2,
            )
        )

        condition.validate()


def test_composite_request():
    request = (
        NetworkRequest()
        .require_embb_throughput(
            10.0
        )
        .limit_embb_buffer(
            50.0
        )
    )

    request.validate_composite_v1()

    assert len(
        request.constraints
    ) == 2
