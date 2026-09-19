# Quantum-Enhanced Adaptive Urban Traffic Optimization

Software-only research prototype for comparing fixed-time, adaptive, and
hybrid quantum-classical traffic signal strategies on a deterministic NetworkX
road topology. The project includes seeded demand, event-driven simulation,
virtual ambulance routing, emergency priority coordination, performance and
environmental estimates, and a Streamlit dashboard.

## Architecture

```text
NetworkX topology
		-> seeded demand and traffic events
		-> ExperimentRunner
		-> TrafficSimulator and selected controller
		-> MetricsCollector
		-> ScenarioExperimentResult
		-> Streamlit and Plotly dashboard
```

The simulator is the single source of truth for vehicle movement and time.
Events provide temporary effective road conditions; they do not mutate the
base `RoadSegment` objects. The Emergency Green Corridor currently exposes a
safe controller orchestration API, but it is not automatically invoked by
`ExperimentRunner` and is not coupled to simulator signal-state updates.

## Installation

```text
pip install -e ".[test]"
```

For the optional Qiskit-backed hybrid controller:

```text
pip install -e ".[quantum]"
```

## Tests

```text
PYTHONPATH=src python -m pytest -q
```

## Dashboard

```text
streamlit run src/traffic_opt/dashboard/app.py
```

The dashboard configures an experiment, generates demand and events, runs the
existing simulator, collects metrics, displays scheduled events and topology,
and compares measured results kept in the current Streamlit session.

## Example workflow

```text
Configure experiment
    -> generate seeded demand/events
    -> run existing simulation
    -> collect snapshots and aggregate metrics
    -> inspect dashboard
    -> compare experiments
```

## Simulation limitations

- Congestion, accident, closure, and emergency effects are deterministic
  simulation approximations, not calibrated traffic engineering models.
- Fuel and CO2 values are configurable estimates, not physical measurements.
- The project is software-only and has no physical signal, sensor, or IoT
  validation.
- Experiment results expose scheduled event data and measured metrics, but do
  not attribute individual metric changes causally to particular events.
- Emergency corridor requests use the existing signal controller, while full
  corridor-to-simulator signal-state integration remains an explicit boundary.
