from src.vehicle.tires import TireModel
from src.vehicle.vehicle import Vehicle
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


def test_vehicle_json_round_trip():
    v = Vehicle(name="Test Car", mass=999.0, max_power=100000.0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "car.json")
        v.to_json(path)
        v2 = Vehicle.from_json(path)
    assert v2.name == "Test Car"
    assert v2.mass == 999.0
    assert v2.max_power == 100000.0
