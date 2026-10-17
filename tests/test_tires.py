from src.vehicle.tires import TireModel
from src.vehicle.vehicle import Vehicle
import numpy as np
import tempfile, os


def test_tire_max_force_scales_with_mu_and_load():
    tire = TireModel(mu=1.0, load_sensitivity=0.0)
    assert tire.max_force(1000.0) == 1000.0
    tire2 = TireModel(mu=1.5, load_sensitivity=0.0)
    assert tire2.max_force(1000.0) == 1500.0


def test_tire_grip_modifier_reduces_effective_mu():
    tire = TireModel(mu=1.2, grip_modifier=0.5)
    assert tire.effective_mu() == 0.6


def test_tire_max_force_nonnegative_for_negative_load():
    tire = TireModel(mu=1.2)
    assert tire.max_force(-500.0) == 0.0


def test_load_sensitivity_drops_mu_above_reference_load():
    tire = TireModel(mu=1.2, load_sensitivity=0.2)
    n_ref = 12000.0
    assert tire.mu_at_load(n_ref, n_ref) == tire.effective_mu()
    assert tire.mu_at_load(2.0 * n_ref, n_ref) < tire.effective_mu()
    # More load still yields more force, just not linearly.
    assert tire.max_force(2.0 * n_ref, n_ref=n_ref) > tire.max_force(n_ref, n_ref=n_ref)


def test_combined_slip_uses_pythagorean_leftover():
    from src.vehicle.tires import combined_slip_fx, friction_utilization
    assert combined_slip_fx(5.0, 0.0) == 5.0
    assert combined_slip_fx(5.0, 5.0) == 0.0
    assert combined_slip_fx(5.0, 3.0) == 4.0
    assert friction_utilization(3.0, 4.0, 5.0) == 1.0


def test_magic_formula_is_zero_at_zero_slip_and_odd():
    from src.vehicle.tires import magic_formula
    y0 = magic_formula(0.0, 10.0, 1.3, 8000.0, -0.5)
    yp = magic_formula(0.08, 10.0, 1.3, 8000.0, -0.5)
    yn = magic_formula(-0.08, 10.0, 1.3, 8000.0, -0.5)
    assert abs(y0) < 1e-9
    assert yp > 0.0
    assert np.isclose(yn, -yp)


def test_magic_formula_peak_is_near_d():
    from src.vehicle.tires import magic_formula
    B, C, D, E = 10.0, 1.3, 5000.0, -0.5
    alpha = np.linspace(0.0, 0.35, 200)
    peak = float(np.max(magic_formula(alpha, B, C, D, E)))
    assert peak > 0.95 * D
    assert peak <= D * 1.02


def test_pure_fy_matches_max_force_near_peak_slip():
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    n = 6000.0
    d = tire.max_force(n)
    alpha = np.linspace(0.0, 0.30, 80)
    peak = float(np.max(tire.fy_pure(alpha, n)))
    assert peak > 0.95 * d


def test_ellipse_leftover_uses_dx_and_dy():
    from src.vehicle.tires import combined_slip_fx
    # Circle leftover is still 3-4-5.
    assert combined_slip_fx(5.0, 3.0) == 4.0
    # Taller Dy than Dx: same Fy uses a smaller fraction of the ellipse.
    assert combined_slip_fx(5.0, 3.0, 6.0) > 4.0


def test_from_vehicle_reads_pacejka_block():
    v = Vehicle(tire_mu=1.15, pacejka={"By": 11.0, "mu_x": 1.4})
    tire = TireModel.from_vehicle(v)
    assert tire.mu == 1.15
    assert tire.By == 11.0
    assert tire.mu_x == 1.4
    assert tire.effective_mu_x() == 1.4


def test_vehicle_json_round_trip():
    v = Vehicle(name="Test Car", mass=999.0, max_power=100000.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "car.json")
        v.to_json(path)
        v2 = Vehicle.from_json(path)
    assert v2.name == "Test Car"
    assert v2.mass == 999.0
    assert v2.max_power == 100000.0
