"""Unit tests for the FoodAPS-calibrated household parameters (no database or mesa needed)."""
from decimal import Decimal

import pytest

from food_access_model.abm import foodaps_calibration as fc

MED_SPM, MED_CSPM = 0.81, 0.40  # FoodAPS median distances (miles)
MED_STORES_1MI = 7               # FoodAPS median SNAP-authorized stores within 1 mile


def test_poverty_guideline_values():
    assert fc.poverty_guideline(1, 2026) == 15960
    assert fc.poverty_guideline(4, 2026) == 15960 + 3 * 5680
    assert fc.poverty_guideline(1, 2025) == 15650
    # years outside the table use the nearest available year
    assert fc.poverty_guideline(1, 2030) == fc.poverty_guideline(1, 2026)


def test_poverty_ratio_bounds_and_low_income():
    assert fc.poverty_ratio(0, 3, 2022) == pytest.approx(0.1)
    assert fc.poverty_ratio(10_000_000, 1, 2022) == pytest.approx(10.0)
    r = fc.poverty_ratio(1.2 * fc.poverty_guideline(3, 2022), 3, 2022)
    assert fc.is_low_income(r)
    assert not fc.is_low_income(fc.poverty_ratio(2 * fc.poverty_guideline(3, 2022), 3, 2022))


def test_trip_probability_reproduces_foodaps_by_vehicle():
    # At median distances and a typical household, the logit should land near the
    # survey-weighted FoodAPS trip shares (0.76 with a vehicle, 0.61 without).
    p_car = fc.prob_spm_trip(MED_SPM, MED_CSPM, True, 2.0, 3, MED_STORES_1MI)
    p_nocar = fc.prob_spm_trip(MED_SPM, MED_CSPM, False, 1.0, 2, MED_STORES_1MI)
    assert p_car == pytest.approx(0.76, abs=0.04)
    assert p_nocar == pytest.approx(0.61, abs=0.04)


def test_trip_probability_directions():
    base = dict(has_vehicle=True, pov_ratio=2.0, household_size=3)
    assert fc.prob_spm_trip(0.5, 0.4, **base) > fc.prob_spm_trip(5.0, 0.4, **base)
    assert fc.prob_spm_trip(0.8, 2.0, **base) > fc.prob_spm_trip(0.8, 0.2, **base)
    assert fc.prob_spm_trip(0.8, 0.4, True, 2.0, 3) > fc.prob_spm_trip(0.8, 0.4, False, 2.0, 3)
    assert fc.prob_spm_trip(0.8, 0.4, stores_within_1mi=1, **base) > \
        fc.prob_spm_trip(0.8, 0.4, stores_within_1mi=20, **base)
    assert fc.prob_spm_trip(0.8, 0.4, rural=0, **base) > fc.prob_spm_trip(0.8, 0.4, rural=1, **base)
    # unknown rural status lies between the urban and rural predictions
    p_u, p_r = (fc.prob_spm_trip(0.8, 0.4, rural=r, **base) for r in (0, 1))
    assert p_r < fc.prob_spm_trip(0.8, 0.4, rural=None, **base) < p_u
    for spm in (0.01, 0.5, 5, 50):
        assert 0 < fc.prob_spm_trip(spm, 0.4, **base) < 1


def test_monthly_trips_poisson():
    assert fc.monthly_trips(1, 0) == 11
    assert fc.monthly_trips(3, 1) == 13
    assert fc.monthly_trips(5, 3) > fc.monthly_trips(5, 1) > fc.monthly_trips(2, 1)
    # FoodAPS weighted mean household (~2.4 people, one vehicle) is close to 12.8 trips
    assert fc.expected_monthly_trips(2.4, 1) == pytest.approx(12.8, abs=1.0)


def test_rural_setting(monkeypatch):
    monkeypatch.delenv("FEAST_RURAL", raising=False)
    assert fc.rural_setting(None) is None
    assert fc.rural_setting(1) == 1.0 and fc.rural_setting(Decimal("0")) == 0.0
    monkeypatch.setenv("FEAST_RURAL", "1")
    assert fc.rural_setting(None) == 1.0
    assert fc.rural_setting(0) == 0.0  # a per-household value overrides the county setting


def test_assets_and_food_insecurity_directions():
    poor = fc.prob_food_insecure(0.5, 3, False, 0)
    rich = fc.prob_food_insecure(4.0, 3, True, 2)
    assert poor > 0.3 and rich < 0.10 and poor > 5 * rich
    assert fc.prob_low_assets(0.5, 3, False, 0) > fc.prob_low_assets(4.0, 3, True, 0)


def test_accepts_decimal_inputs_from_database():
    # Households are rebuilt from DB records whose numerics may be Decimal
    r = fc.poverty_ratio(Decimal("45000"), Decimal("3"), 2022)
    assert 0 < fc.prob_spm_trip(Decimal("0.8"), Decimal("0.4"), True, r, Decimal("3"), Decimal("5")) < 1
    assert fc.monthly_trips(Decimal("3"), Decimal("1")) == 13
    assert 0 < fc.prob_food_insecure(r, Decimal("3"), True, Decimal("1")) < 1
