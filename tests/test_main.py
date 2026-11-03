import pytest
from fastapi.testclient import TestClient
from main import app, compute_noaa_solar_geometry, compute_pv_yield_engine

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_tools():
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "tools" in data

def test_solar_geometry_endpoint():
    response = client.get("/tools/get_solar_geometry?lat=2.0442&lon=102.5689")
    assert response.status_code == 200
    data = response.json()
    assert "solar_elevation_deg" in data
    assert "solar_azimuth_deg" in data
    assert "solar_zenith_deg" in data

def test_pv_yield_endpoint():
    response = client.get("/tools/get_pv_yield?lat=2.0442&lon=102.5689")
    assert response.status_code == 200
    data = response.json()
    assert "expected_generation_kw" in data
    assert "panel_efficiency" in data

def test_space_weather_endpoint():
    response = client.get("/tools/get_space_weather")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)

def test_charging_opportunity_endpoint():
    response = client.get("/tools/get_charging_opportunity?lat=2.0442&lon=102.5689")
    assert response.status_code == 200
    data = response.json()
    assert "charging_score" in data
    assert "charge_rate_w" in data

def test_circadian_light_endpoint():
    response = client.get("/tools/get_circadian_light?lat=2.0442&lon=102.5689")
    assert response.status_code == 200
    data = response.json()
    assert "circadian_light_score" in data
    assert "activity_potential" in data

def test_sunos_signal_endpoint():
    response = client.get("/tools/get_sunos_signal?lat=2.0442&lon=102.5689")
    assert response.status_code == 200
    data = response.json()
    assert "sun_state_score" in data
    assert "energy_score" in data

def test_offline_solar_calculation():
    geom = compute_noaa_solar_geometry(2.0442, 102.5689)
    assert isinstance(geom, dict)
    assert "solar_elevation_deg" in geom
    
    pv = compute_pv_yield_engine(2.0442, 102.5689)
    assert isinstance(pv, dict)
    assert "expected_generation_kw" in pv

def test_anti_placeholder_audit():
    geom = compute_noaa_solar_geometry(2.0442, 102.5689)
    # Ensure values are not hardcoded placeholder defaults
    assert geom["solar_elevation_deg"] != 45.0
