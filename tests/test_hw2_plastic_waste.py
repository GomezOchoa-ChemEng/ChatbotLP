import sys
from pathlib import Path

import pytest
from pyomo.environ import SolverFactory

sys.path.insert(0, str(Path.cwd()))

from src.design_model import (
    build_fixed_charge_design_model_from_state,
    build_fixed_design_management_model_from_state,
    compute_balance_residuals,
    compute_objective_decomposition,
    extract_design_solution,
    extract_node_product_prices,
    find_minimum_price_for_target,
    solve_design_problem,
    solve_fixed_design_management_problem,
    validate_design_state,
)
from src.hw2_plastic_waste import (
    ETHYLENE,
    PLASTIC_WASTE,
    build_hw2_plastic_waste_state,
    total_available_plastic_waste,
    total_recycled_plastic_waste,
)
from src.network_graph import build_problem_graph_spec, build_solution_graph_spec
from src.network_visualizer import render_mermaid
from src.solver_results import SolverResults


def glpk_available() -> bool:
    return SolverFactory("glpk").available(exception_flag=False)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_hw2_fixture_problem_state_construction_and_zero_supply_costs():
    state = build_hw2_plastic_waste_state()

    assert state.node_ids() == ["MAD", "MKE"]
    assert set(state.product_ids()) == {PLASTIC_WASTE, "PyrolysisOil", ETHYLENE}
    assert len(state.suppliers) == 2
    assert len(state.consumers) == 2
    assert len(state.transport_links) == 6
    assert len(state.technologies) == 4
    assert total_available_plastic_waste(state) == 110400.0

    supplier_bids = [bid for bid in state.bids if bid.owner_type == "supplier"]
    assert [bid.price for bid in supplier_bids] == [0.0, 0.0]
    assert all(bid.price is not None for bid in supplier_bids)


def test_hw2_missing_investment_cost_blocks_design_solving():
    state = build_hw2_plastic_waste_state(missing_investment_cost=True)
    diagnostics = validate_design_state(state)

    assert diagnostics["solver_ready"] is False
    assert any("missing fixed_cost" in item for item in diagnostics["missing_parameters"])
    assert any(technology.fixed_cost is None for technology in state.technologies)
    with pytest.raises(ValueError, match="missing fixed_cost"):
        build_fixed_charge_design_model_from_state(state)


def test_hw2_explicit_zero_fixed_cost_is_allowed():
    state = build_hw2_plastic_waste_state()
    for technology in state.technologies:
        technology.fixed_cost = 0.0

    diagnostics = validate_design_state(state)
    model = build_fixed_charge_design_model_from_state(state)

    assert diagnostics["solver_ready"] is True
    assert all(technology.fixed_cost == 0.0 for technology in state.technologies)
    assert hasattr(model, "y")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda state: setattr(state.technologies[0], "capacity", None), "missing capacity"),
        (
            lambda state: state.technologies[0].yield_coefficients.__setitem__(PLASTIC_WASTE, None),
            "missing yield",
        ),
        (lambda state: setattr(state.transport_links[0], "cost", None), "missing cost"),
        (lambda state: setattr(state.transport_links[0], "capacity", None), "missing capacity"),
        (lambda state: setattr(state.bids[0], "price", None), "missing price"),
        (lambda state: setattr(state.bids[0], "quantity", None), "missing quantity"),
    ],
)
def test_hw2_required_missing_values_block_design_model(mutate, message):
    state = build_hw2_plastic_waste_state()
    mutate(state)

    diagnostics = validate_design_state(state)

    assert diagnostics["solver_ready"] is False
    assert any(message in item for item in diagnostics["missing_parameters"])
    with pytest.raises(ValueError):
        build_fixed_charge_design_model_from_state(state)


def test_hw2_problem_graph_has_expected_products_nodes_and_technologies():
    state = build_hw2_plastic_waste_state()
    before = state.to_dict()

    spec = build_problem_graph_spec(state)
    mermaid = render_mermaid(spec)

    assert set(spec.product_ids) == {PLASTIC_WASTE, "PyrolysisOil", ETHYLENE}
    assert spec.metadata["node_count"] == 2
    assert spec.metadata["technology_count"] == 4
    assert spec.metadata["transport_edge_count"] == 6
    assert "flowchart LR" in mermaid
    assert "Pyrolysis_MKE" in {tech for node in spec.nodes for tech in node.attributes["technology_ids"]}
    assert state.to_dict() == before


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_design_milp_solves_and_selects_expected_technologies():
    state = build_hw2_plastic_waste_state()
    result = solve_design_problem(state, solver_name="glpk")
    summary = extract_design_solution(state, result)
    decomposition = compute_objective_decomposition(state, result)

    assert result.success
    assert result.objective_value == pytest.approx(6978000.0)
    assert summary["selected_technologies"] == {
        "Pyrolysis_MKE": 1.0,
        "SteamCracking_MKE": 1.0,
    }
    assert summary["technology_activities"]["Pyrolysis_MKE"] == pytest.approx(100000.0)
    assert summary["technology_activities"]["SteamCracking_MKE"] == pytest.approx(70000.0)
    assert total_recycled_plastic_waste(state, result) == pytest.approx(100000.0)
    assert decomposition["investment_cost"] == pytest.approx(4800000.0)
    assert decomposition["total_profit"] == pytest.approx(result.objective_value)


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_selected_technology_capacity_logic():
    state = build_hw2_plastic_waste_state()
    result = solve_design_problem(state, solver_name="glpk")
    summary = extract_design_solution(state, result)

    for technology in state.technologies:
        activity = summary["technology_activities"][technology.id]
        installed = summary["technology_installations"][technology.id]
        assert activity <= technology.capacity * installed + 1e-6
    assert summary["technology_activities"]["Pyrolysis_MAD"] == pytest.approx(0.0)
    assert summary["technology_installations"]["Pyrolysis_MAD"] == pytest.approx(0.0)


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_fixed_cost_selection_logic_can_turn_off_all_technologies():
    state = build_hw2_plastic_waste_state()
    for technology in state.technologies:
        technology.fixed_cost = 1_000_000_000.0

    result = solve_design_problem(state, solver_name="glpk")
    summary = extract_design_solution(state, result)

    assert result.success
    assert result.objective_value == pytest.approx(0.0)
    assert summary["selected_technologies"] == {}
    assert total_recycled_plastic_waste(state, result) == pytest.approx(0.0)


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_product_balances_hold_for_all_nodes_and_products():
    state = build_hw2_plastic_waste_state()
    result = solve_design_problem(state, solver_name="glpk")
    rows = compute_balance_residuals(state, result)

    assert {(row["node"], row["product"]) for row in rows} == {
        ("MAD", PLASTIC_WASTE),
        ("MAD", "PyrolysisOil"),
        ("MAD", ETHYLENE),
        ("MKE", PLASTIC_WASTE),
        ("MKE", "PyrolysisOil"),
        ("MKE", ETHYLENE),
    }
    assert all(row["holds"] for row in rows)
    assert max(abs(row["residual"]) for row in rows) <= 1e-6


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_solution_graph_shows_selected_technologies_and_active_flows():
    state = build_hw2_plastic_waste_state()
    result = solve_design_problem(state, solver_name="glpk")
    solver_results = SolverResults.from_solve_result(result, state)
    spec = build_solution_graph_spec(state, solver_results)

    nodes = {node.id: node for node in spec.nodes}
    edges = {edge.id: edge for edge in spec.edges}

    assert nodes["MKE"].solution["technology_installation"]["Pyrolysis_MKE"] == pytest.approx(1.0)
    assert nodes["MKE"].solution["technology_installation"]["SteamCracking_MKE"] == pytest.approx(1.0)
    assert nodes["MAD"].solution["technology_installation"]["Pyrolysis_MAD"] == pytest.approx(0.0)
    assert edges["T_MAD_to_MKE_PlasticWaste"].active is True
    assert edges["T_MAD_to_MKE_PlasticWaste"].flow == pytest.approx(22700.0)
    assert edges["T_MKE_to_MAD_Ethylene"].active is False


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_fixed_design_management_model_is_lp_and_imports_duals():
    state = build_hw2_plastic_waste_state()
    design_result = solve_design_problem(state, solver_name="glpk")
    selected = extract_design_solution(state, design_result)["selected_technologies"]

    model = build_fixed_design_management_model_from_state(state, selected)
    management_result = solve_fixed_design_management_problem(state, selected, solver_name="glpk")
    prices = extract_node_product_prices(management_result)

    assert not hasattr(model, "y")
    assert hasattr(model, "dual")
    assert management_result.success
    assert prices["MKE:Ethylene"] == pytest.approx(1050.0)
    assert prices["MKE:PlasticWaste"] == pytest.approx(10.0)


def test_hw2_missing_installation_value_is_not_treated_as_zero():
    state = build_hw2_plastic_waste_state()

    with pytest.raises(ValueError, match="installed technology value"):
        build_fixed_design_management_model_from_state(state, {"Pyrolysis_MAD": None})


@pytest.mark.skipif(not glpk_available(), reason="GLPK is not installed")
def test_hw2_wtp_sweep_identifies_threshold_for_full_recycling():
    state = build_hw2_plastic_waste_state()
    target = total_available_plastic_waste(state)
    result = find_minimum_price_for_target(
        state,
        product_id=ETHYLENE,
        target_quantity=target,
        quantity_getter=total_recycled_plastic_waste,
        low=500.0,
        high=2500.0,
        coarse_steps=9,
        bisection_iterations=12,
        solver_name="glpk",
    )

    assert result["threshold"] == pytest.approx(1465.48, abs=1.0)
    threshold_summary = extract_design_solution(
        result["threshold_state"],
        result["threshold_result"],
    )
    assert total_recycled_plastic_waste(result["threshold_state"], result["threshold_result"]) == pytest.approx(target)
    assert set(threshold_summary["selected_technologies"]) == {
        "Pyrolysis_MAD",
        "Pyrolysis_MKE",
        "SteamCracking_MKE",
    }


def test_hw2_graph_construction_does_not_mutate_problem_state():
    state = build_hw2_plastic_waste_state()
    before = state.to_dict()

    build_problem_graph_spec(state)

    assert state.to_dict() == before
