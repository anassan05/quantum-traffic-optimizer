"""Pure data adapters for dashboard rendering and export."""

import csv
import io
import json
from dataclasses import asdict
from typing import Any

from traffic_opt.domain.models import TrafficEvent
from traffic_opt.experiments.results import ComparisonRow, ScenarioExperimentResult


def result_to_kpis(result: ScenarioExperimentResult) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "Average waiting time (s)": metrics.average_waiting_time_seconds,
        "Average queue length": metrics.average_queue_length,
        "Throughput (vehicles/s)": metrics.throughput_vehicles_per_second,
        "Average travel time (s)": metrics.average_travel_time_seconds,
        "Fuel estimate (L)": metrics.total_fuel_consumed_litres,
        "CO2 estimate (g)": metrics.total_co2_emissions_grams,
        "Generated vehicles": result.number_of_generated_vehicles,
        "Completed vehicles": metrics.completed_vehicles,
        "Generated events": result.number_of_generated_events,
    }


def result_to_snapshot_rows(result: ScenarioExperimentResult) -> list[dict[str, Any]]:
    return [
        {
            "time_seconds": snapshot.simulation_time_seconds,
            "waiting_time_seconds": snapshot.average_waiting_time_seconds,
            "queue_length": snapshot.total_queue_length,
            "throughput_vehicles_per_second": snapshot.throughput_vehicles_per_second,
            "travel_time_seconds": snapshot.average_travel_time_seconds,
            "fuel_litres": snapshot.fuel_consumed_litres,
            "co2_grams": snapshot.co2_emissions_grams,
        }
        for snapshot in result.snapshots
    ]


def result_to_event_rows(result: ScenarioExperimentResult) -> list[dict[str, Any]]:
    return [event_to_row(event) for event in result.events]


def event_to_row(event: TrafficEvent) -> dict[str, Any]:
    metadata = dict(event.metadata)
    row = {
        "event_id": event.id,
        "event_type": event.event_type.value,
        "start_time_seconds": event.start_time_seconds,
        "end_time_seconds": event.start_time_seconds + event.duration_seconds,
        "duration_seconds": event.duration_seconds,
        "affected_roads": ", ".join(event.affected_road_ids) or "All roads",
    }
    row.update({f"metadata_{key}": value for key, value in metadata.items()})
    return row


def result_to_history_row(result: ScenarioExperimentResult) -> dict[str, Any]:
    return {
        "experiment": result.experiment_name,
        "controller": result.controller_name,
        "scenario": result.scenario,
        "seed": result.seed,
        "duration_seconds": result.simulation_duration_seconds,
        "vehicles": result.number_of_generated_vehicles,
        "events": result.number_of_generated_events,
        "completed": result.metrics.completed_vehicles,
        "average_wait_seconds": result.metrics.average_waiting_time_seconds,
        "throughput_vehicles_per_second": result.metrics.throughput_vehicles_per_second,
        "co2_grams": result.metrics.total_co2_emissions_grams,
    }


def comparison_rows_to_dicts(rows: tuple[ComparisonRow, ...]) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]


def result_to_json(result: ScenarioExperimentResult) -> str:
    payload = {
        "experiment_name": result.experiment_name,
        "controller_name": result.controller_name,
        "scenario": result.scenario,
        "seed": result.seed,
        "simulation_duration_seconds": result.simulation_duration_seconds,
        "demands": [asdict(demand) | {"vehicle_type": demand.vehicle_type.value} for demand in result.demands],
        "events": result_to_event_rows(result),
        "metrics": asdict(result.metrics),
        "snapshots": result_to_snapshot_rows(result),
        "ambulance_count": result.ambulance_count,
        "completed_ambulances": result.completed_ambulances,
        "average_ambulance_travel_time_seconds": result.average_ambulance_travel_time_seconds,
        "average_ambulance_waiting_time_seconds": result.average_ambulance_waiting_time_seconds,
    }
    return json.dumps(payload, indent=2)


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()