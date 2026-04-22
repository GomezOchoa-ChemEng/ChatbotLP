import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "examples"))

from scenario_sweep_figure_demo import build_supplier_utilization_plot, build_total_welfare_plot, run_demo


def test_scenario_sweep_demo_builds_plot_table_from_figure_data():
    demo = run_demo()

    scenario_sweep_df = demo["scenario_sweep_df"]
    plot_table = demo["plot_table"]

    assert not scenario_sweep_df.empty
    assert not plot_table.empty
    assert set(["run_id", "scenario_id", "parameter_name", "parameter_value", "metric_name", "metric_value"]).issubset(
        scenario_sweep_df.columns
    )
    assert set(plot_table["metric_name"]) == {"total_welfare"}
    assert list(plot_table["parameter_value"]) == [4.0, 8.0, 10.0]
    assert list(plot_table["metric_value"]) == [28.0, 56.0, 56.0]


def test_scenario_sweep_demo_plot_call_uses_filtered_table():
    demo = run_demo()
    fig, ax, plot_table = build_total_welfare_plot(demo["scenario_sweep_df"])

    line = ax.lines[0]
    assert list(line.get_xdata()) == list(plot_table["parameter_value"])
    assert list(line.get_ydata()) == list(plot_table["metric_value"])
    assert ax.get_xlabel() == "Transport Capacity"
    assert ax.get_ylabel() == "Total Welfare"
    fig.clf()


def test_utilization_demo_builds_supplier_plot_from_figure_data():
    demo = run_demo()

    utilization_df = demo["utilization_df"]
    plot_table = demo["utilization_plot_table"]

    assert not utilization_df.empty
    assert not plot_table.empty
    assert set(["asset_type", "asset_id", "parameter_value", "utilization_fraction"]).issubset(utilization_df.columns)
    assert set(plot_table["asset_type"]) == {"supplier"}
    assert list(plot_table["parameter_value"]) == [4.0, 8.0, 10.0]
    assert list(plot_table["utilization_fraction"]) == [0.4, 0.8, 0.8]


def test_utilization_demo_plot_call_uses_supplier_utilization_table():
    demo = run_demo()
    fig, ax, plot_table = build_supplier_utilization_plot(demo["utilization_df"])

    bars = ax.patches
    heights = [bar.get_height() for bar in bars]
    assert heights == list(plot_table["utilization_fraction"])
    assert ax.get_xlabel() == "Transport Capacity"
    assert ax.get_ylabel() == "Supplier Utilization Fraction"
    fig.clf()
