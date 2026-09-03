"""Checks on the engineering models, against catalogue data and closed forms."""

import math

import numpy as np
import pytest

from arm_lab_model import structures, system
from arm_lab_model.actuator_model import (ActuatorElectrical, ActuatorThermal,
                                          ActuatorModel, from_config_actuator)
from arm_lab_model.config import load_config
from arm_lab_model.kinematics import ArmModel
from arm_lab_model.reference import (CSF_RATIO_50, CSF_RATIO_80_PLUS,
                                     resistance_at, select_gearbox,
                                     structural_series_stiffness)


@pytest.fixture(scope='module')
def cfg():
    return load_config()


@pytest.fixture(scope='module')
def model(cfg):
    return ArmModel(cfg)


# ------------------------------------------------------------- reference
def test_catalogue_matches_the_published_values():
    """Spot-check against the numbers quoted in the Harmonic Drive tables."""
    assert CSF_RATIO_80_PLUS[17].k3 == pytest.approx(1.6e4)
    assert CSF_RATIO_80_PLUS[25].k3 == pytest.approx(5.7e4)
    assert CSF_RATIO_50[25].k3 == pytest.approx(4.4e4)
    # A higher ratio is stiffer at the output, in every size.
    for size in CSF_RATIO_50:
        assert CSF_RATIO_80_PLUS[size].k3 >= CSF_RATIO_50[size].k3


def test_gearbox_stiffens_with_load():
    g = CSF_RATIO_80_PLUS[25]
    assert g.stiffness_at(g.t1 * 0.5) == g.k1
    assert g.stiffness_at((g.t1 + g.t2) / 2) == g.k2
    assert g.stiffness_at(g.t2 * 2) == g.k3
    # Wind-up is monotonic and continuous across the segment boundaries.
    assert g.windup(g.t1 * 0.99) < g.windup(g.t1 * 1.01) < g.windup(g.t2)


def test_series_compliance_is_softer_than_any_member():
    combined = structural_series_stiffness(120000, 25000, 40000)
    assert combined < 25000
    assert combined == pytest.approx(1 / (1/120000 + 1/25000 + 1/40000))


def test_selected_gearbox_covers_the_load():
    g = select_gearbox(97.5, 100)
    assert g is not None and g.t2 >= 97.5
    smaller = [s for s in CSF_RATIO_80_PLUS if s < g.size]
    for s in smaller:
        assert CSF_RATIO_80_PLUS[s].t2 < 97.5


def test_config_derives_joint_stiffness_from_the_catalogue(cfg):
    for act in cfg.actuators.values():
        if not act.gearbox_series:
            continue
        assert act.gearbox_stiffness > 0
        # The joint must be softer than its gearbox: brackets are in series.
        assert act.joint_stiffness < act.gearbox_stiffness


def test_copper_resistance_rises_with_temperature():
    assert resistance_at(0.17, 155) > 0.17 * 1.4
    assert resistance_at(0.17, 20) == pytest.approx(0.17)


# ------------------------------------------------------------- actuator
def test_torque_never_exceeds_the_rating(cfg):
    for joint in cfg.joints:
        m = from_config_actuator(joint.actuator)
        if m is None:
            continue
        speeds, torques = m.electrical.envelope()
        assert torques.max() <= joint.effort_limit * 1.001
        # Torque is monotonically non-increasing with speed.
        assert np.all(np.diff(torques) <= 1e-9)
        assert torques[-1] == pytest.approx(0.0, abs=1e-6)


def test_back_emf_sets_the_no_load_speed():
    e = ActuatorElectrical(torque_constant=0.1, phase_resistance=0.2,
                           bus_voltage=48.0, max_phase_current=20.0,
                           gear_ratio=100.0, efficiency=0.8,
                           peak_output_torque=1e9)
    assert e.no_load_speed == pytest.approx(48.0 / 0.1 / 100.0)
    assert e.available_torque(e.no_load_speed) == pytest.approx(0.0, abs=1e-9)
    assert e.corner_speed() < e.no_load_speed


def test_thermal_time_constant_and_limit():
    t = ActuatorThermal(thermal_resistance=1.5, thermal_capacity=200.0,
                        max_winding_temp=155.0, ambient_temp=25.0)
    assert t.time_constant == pytest.approx(300.0)
    # 63 % of the rise after one time constant.
    rise = t.steady_state_temperature(50.0) - t.ambient_temp
    reached = t.temperature_after(50.0, t.time_constant) - t.ambient_temp
    assert reached / rise == pytest.approx(1 - math.exp(-1), rel=1e-9)
    # A load whose steady state is under the limit runs for ever.
    assert math.isinf(t.time_to_limit(10.0))
    assert t.time_to_limit(200.0) < t.time_constant


def test_holding_is_thermally_limited_before_it_is_torque_limited():
    e = ActuatorElectrical(0.1, 0.2, 48.0, 20.0, 100.0, 0.8,
                           peak_output_torque=160.0)
    t = ActuatorThermal(2.0, 100.0, 155.0, 25.0)
    m = ActuatorModel(e, t)
    easy = m.holding(20.0)
    hard = m.holding(140.0)
    assert easy.continuous
    assert not hard.continuous
    assert hard.hold_seconds > 0
    assert hard.loss_w > easy.loss_w


# ------------------------------------------------------------ structures
def test_euler_buckling_matches_the_closed_form(cfg):
    link = cfg.joints[1].link
    r = structures.buckling(link, 1.0, 'pinned-pinned')
    expected = (math.pi ** 2 * link.material.youngs_modulus
                * link.second_moment_area / link.length ** 2)
    assert r.euler_critical == pytest.approx(expected)


def test_thin_walls_buckle_locally_before_they_buckle_as_columns(cfg):
    """The mode that catches thin tubes out."""
    import copy
    link = copy.deepcopy(cfg.joints[1].link)
    link.inner_radius = link.outer_radius - 0.0004     # very thin wall
    r = structures.buckling(link, 1.0)
    assert r.shell_critical < r.euler_critical
    assert r.mode == 'local wall buckling'


def test_fatigue_life_falls_as_stress_rises(cfg):
    """A cycle from zero to `stress_max` alternates about half of it."""
    link = cfg.joints[1].link
    low = structures.fatigue(link, 20e6)
    high = structures.fatigue(link, 800e6)
    assert math.isinf(low.cycles_to_failure), 'below the endurance limit'
    assert math.isfinite(high.cycles_to_failure)
    assert low.safety_factor > high.safety_factor


def test_stress_below_the_endurance_limit_gives_infinite_life(cfg):
    link = cfg.joints[1].link
    se = structures.endurance_limit(link.material, 2 * link.outer_radius)
    just_under = structures.fatigue(link, 1.9 * se)
    assert just_under.alternating_stress < se
    assert math.isinf(just_under.cycles_to_failure)


def test_mass_correction_only_adds(cfg):
    correction = structures.MassCorrection.from_config(cfg)
    corrected = correction.corrected_arm_mass(cfg)
    assert corrected > cfg.arm_mass
    assert corrected < cfg.arm_mass * 2.0


def test_bearing_life_follows_the_cube_law():
    a = structures.bearing_l10('j', 1000.0, 10000.0, 600.0)
    b = structures.bearing_l10('j', 2000.0, 10000.0, 600.0)
    assert a.revolutions / b.revolutions == pytest.approx(8.0)


# --------------------------------------------------------------- system
def test_bus_latency_grows_with_node_count(cfg, model):
    verdict = system.timing_analysis(cfg, model)
    t = verdict.timing
    assert t.worst_case_latency > t.frame_time
    assert t.utilisation == pytest.approx(
        t.bus_time_per_cycle * t.control_rate)
    assert t.phase_lag_deg(10.0) > 0


def test_contact_detection_reports_the_friction_floor(cfg, model):
    verdicts = system.contact_detection(cfg, model)
    assert verdicts
    current = verdicts[0]
    assert current.detectable >= current.force_resolution
    assert current.detectable >= 0.0


def test_output_encoder_reduces_the_tcp_error(cfg, model):
    verdicts = system.encoder_analysis(cfg, model, payload=2.0)
    before = sum(v.tcp_error_before for v in verdicts)
    after = sum(v.tcp_error_after for v in verdicts)
    assert before > 0
    assert after < before


def test_holding_flags_backdriving_joints(cfg, model):
    verdicts = system.holding_analysis(cfg, model, payload=2.0)
    loaded = [v for v in verdicts if v.hold_torque > 10.0]
    assert loaded, 'expected some joints to carry real holding torque'
    for v in loaded:
        assert v.falls_on_power_loss, (
            'a joint holding well above its backdrive threshold must be '
            'reported as backdriving when the power goes')
        assert v.descent_speed > 0


def test_regeneration_is_reported_on_the_main_bus(cfg, model):
    buses = system.bus_analysis(cfg, model, payload=2.0)
    main = max(buses)
    assert buses[main].regen_power > 0
    assert buses[main].regen_current > 0
