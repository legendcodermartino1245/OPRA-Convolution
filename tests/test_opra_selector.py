import pytest

from fir_dsp.opra_selector import eq_to_txt, select_eq


def _eq(*, preamp=0.0, band=None):
    return {
        "id": "vendor:product::target",
        "data": {
            "parameters": {
                "gain_db": preamp,
                "bands": [
                    band
                    or {
                        "type": "peak_dip",
                        "frequency": 1000.0,
                        "gain_db": 3.0,
                        "q": 0.7,
                    }
                ],
            }
        },
    }


def test_eq_to_txt_rejects_invalid_preamp_values():
    with pytest.raises(TypeError, match="preamp_db must be numeric, not bool"):
        eq_to_txt(_eq(preamp=True))

    with pytest.raises(ValueError, match="preamp_db must be finite"):
        eq_to_txt(_eq(preamp=float("nan")))


def test_eq_to_txt_rejects_bands_that_generation_would_reject():
    with pytest.raises(ValueError, match="band gain_db must be within the safe range"):
        eq_to_txt(_eq(band={"type": "peak_dip", "frequency": 1000.0, "gain_db": 10_000.0, "q": 0.7}))

    with pytest.raises(ValueError, match="band q must be a positive finite value"):
        eq_to_txt(_eq(band={"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.0}))

    with pytest.raises(ValueError, match="band frequency must be a positive finite value"):
        eq_to_txt(_eq(band={"type": "peak_dip", "frequency": 0.0, "gain_db": 3.0, "q": 0.7}))


def test_eq_to_txt_supports_low_pass_bands():
    text = eq_to_txt(
        _eq(
            band={
                "type": "low_pass",
                "frequency": 12500.0,
                "slope": 12.0,
            }
        )
    )

    assert "Filter 1: ON LP Fc 12500.0 Hz Slope 12.0 dB/oct" in text


def test_eq_to_txt_rejects_low_pass_slope_that_generation_rejects():
    with pytest.raises(ValueError, match="band slope must be one of"):
        eq_to_txt(
            _eq(
                band={
                    "type": "low_pass",
                    "frequency": 12500.0,
                    "slope": 24.0,
                }
            )
        )


def test_select_eq_target_filter_ignores_invalid_nonmatching_siblings():
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    valid_eq = {
        "id": "vendor:product::autoeq_valid",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Example Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                ],
            },
        },
    }
    invalid_sibling = {
        "id": "vendor:product::oratory1990_harman_target",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Example Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "band_stop", "frequency": 1000.0, "gain_db": 0.0, "q": 0.7},
                ],
            },
        },
    }

    selected = select_eq(
        product,
        {product["id"]: [valid_eq, invalid_sibling]},
        vendors,
        target="autoeq_valid",
    )

    assert selected["id"] == valid_eq["id"]


def test_select_eq_rejects_matching_eq_with_invalid_generation_parameters():
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    invalid_eq = {
        "id": "vendor:product::oratory1990_harman_target",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Example Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "low_pass", "frequency": 1000.0, "slope": 24.0},
                ],
            },
        },
    }

    with pytest.raises(ValueError, match="band slope must be one of"):
        select_eq(
            product,
            {product["id"]: [invalid_eq]},
            vendors,
            target="oratory1990_harman_target",
        )


def test_select_eq_still_reports_invalid_matching_candidate():
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}
    invalid_eq = {
        "id": "vendor:product::oratory1990_harman_target",
        "data": {
            "type": "parametric_eq",
            "product_id": "vendor::product",
            "details": "Measured by Example Rig",
            "parameters": {
                "gain_db": 0.0,
                "bands": [
                    {"type": "band_stop", "frequency": 1000.0, "gain_db": 0.0, "q": 0.7},
                ],
            },
        },
    }

    with pytest.raises(ValueError, match="invalid band type band_stop"):
        select_eq(
            product,
            {product["id"]: [invalid_eq]},
            vendors,
            target="oratory1990_harman_target",
        )


def test_select_eq_prefers_ranked_default_target_when_multiple_valid_eqs_exist():
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}

    def _valid_eq(eq_id: str) -> dict:
        return {
            "id": eq_id,
            "data": {
                "type": "parametric_eq",
                "product_id": "vendor::product",
                "details": "Measured by Example Rig",
                "parameters": {
                    "gain_db": 0.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                    ],
                },
            },
        }

    selected = select_eq(
        product,
        {
            product["id"]: [
                _valid_eq("vendor:product::autoeq_crinacle"),
                _valid_eq("vendor:product::autoeq_rtings"),
            ]
        },
        vendors,
    )

    assert selected["id"] == "vendor:product::autoeq_crinacle"


def test_select_eq_keeps_meaningful_variant_ties_ambiguous():
    product = {
        "id": "vendor::product",
        "data": {"subtype": "in_ear", "vendor_id": "vendor", "name": "Product"},
    }
    vendors = {"vendor": {"id": "vendor", "data": {"name": "Vendor"}}}

    def _valid_eq(eq_id: str) -> dict:
        return {
            "id": eq_id,
            "data": {
                "type": "parametric_eq",
                "product_id": "vendor::product",
                "details": "Measured by Example Rig",
                "parameters": {
                    "gain_db": 0.0,
                    "bands": [
                        {"type": "peak_dip", "frequency": 1000.0, "gain_db": 3.0, "q": 0.7},
                    ],
                },
            },
        }

    with pytest.raises(ValueError, match="Ambiguous EQ set"):
        select_eq(
            product,
            {
                product["id"]: [
                    _valid_eq("vendor:product::autoeq_crinacle_blue"),
                    _valid_eq("vendor:product::autoeq_crinacle_red"),
                ]
            },
            vendors,
        )
