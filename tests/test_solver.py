"""Unit tests for the solver module.

Tests cover:
- Solving a simple model with glpk (or a mock solver if unavailable)
- Status reporting (optimal, infeasible, etc.)
- Objective value extraction
- Decision variable extraction (indexed and scalar)
- Error handling for missing solvers
- Result serialization
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from pyomo.environ import ConcreteModel, Set, Var, NonNegativeReals, Objective, Constraint, SolverFactory

from src.solver import (
    solve_model,
    SolveResult,
    extract_variable_values,
    save_result,
    load_result,
)


# ------------------------------------------------
# Fixtures
# ------------------------------------------------


@pytest.fixture
def simple_lp_model():
    """A simple linear program: maximize 3*x + 2*y subject to x + y <= 5, x,y >= 0."""
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)
    model.y = Var(domain=NonNegativeReals)

    def obj_rule(m):
        return 3 * m.x + 2 * m.y

    model.obj = Objective(rule=obj_rule, sense=-1)  # maximize

    def constraint_rule(m):
        return m.x + m.y <= 5

    model.constraint = Constraint(rule=constraint_rule)
    return model


@pytest.fixture
def indexed_model():
    """A model with indexed variables: maximize sum of x[i] for i in 1..3, subject to sum(x) <= 6."""
    model = ConcreteModel()
    model.I = Set(initialize=[1, 2, 3])
    model.x = Var(model.I, domain=NonNegativeReals)

    def obj_rule(m):
        return sum(m.x[i] for i in m.I)

    model.obj = Objective(rule=obj_rule, sense=-1)  # maximize

    def constraint_rule(m):
        return sum(m.x[i] for i in m.I) <= 6

    model.constraint = Constraint(rule=constraint_rule)
    return model


@pytest.fixture
def infeasible_model():
    """An infeasible model: x >= 5 and x <= 2."""
    model = ConcreteModel()
    model.x = Var(domain=NonNegativeReals)

    def obj_rule(m):
        return m.x

    model.obj = Objective(rule=obj_rule, sense=-1)

    def constraint1(m):
        return m.x >= 5

    def constraint2(m):
        return m.x <= 2

    model.constraint1 = Constraint(rule=constraint1)
    model.constraint2 = Constraint(rule=constraint2)
    return model


# ------------------------------------------------
# Tests
# ------------------------------------------------


class TestSolveModel:
    """Tests for the solve_model function."""

    @patch('src.solver.SolverFactory')
    def test_solve_simple_lp(self, mock_factory, simple_lp_model):
        """Test solving a simple LP with mock solver."""
        # Create mock solver
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        # Create mock results
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        # Set variable values on the model
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")

        assert result.success, f"Solve failed: {result.message}"
        assert result.status == "optimal"
        assert result.objective_value is not None
        # Expected: maximize 3*x + 2*y s.t. x + y <= 5
        # Solution: x=5, y=0, obj=15
        assert abs(result.objective_value - 15.0) < 1e-4

    @patch('src.solver.SolverFactory')
    def test_solve_indexed_model(self, mock_factory, indexed_model):
        """Test solving a model with indexed variables."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        # Set variable values for each index in the set data
        for i in [1, 2, 3]:
            indexed_model.x[i].set_value(2.0)

        result = solve_model(indexed_model, solver_name="glpk")

        assert result.success, f"Solve failed: {result.message}"
        assert result.objective_value is not None
        # Expected: maximize sum(x[i]) s.t. sum(x[i]) <= 6
        # Solution: sum = 6
        assert abs(result.objective_value - 6.0) < 1e-4

    @patch('src.solver.SolverFactory')
    def test_infeasible_model(self, mock_factory, infeasible_model):
        """Test handling of infeasible problems."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "infeasible"
        mock_results.solver.termination_condition = "infeasible"
        mock_solver.solve.return_value = mock_results
        
        # Set a value so extraction doesn't fail
        infeasible_model.x.set_value(0.0)

        result = solve_model(infeasible_model, solver_name="glpk")

        assert not result.success
        assert result.status == "infeasible"
        assert result.objective_value is None

    @patch('src.solver.SolverFactory')
    def test_solver_options(self, mock_factory, simple_lp_model):
        """Test passing solver options."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        options = {}
        result = solve_model(
            simple_lp_model,
            solver_name="glpk",
            solver_options=options,
        )
        assert result.success

    @patch('src.solver.SolverFactory')
    def test_result_to_dict(self, mock_factory, simple_lp_model):
        """Test converting SolveResult to dictionary."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "status" in result_dict
        assert "objective_value" in result_dict
        assert "solver_time" in result_dict
        assert "success" in result_dict
        assert "solution" in result_dict


class TestResultExtraction:
    """Tests for extraction of solution components."""

    @patch('src.solver.SolverFactory')
    def test_extract_scalar_variable(self, mock_factory, simple_lp_model):
        """Test extracting a scalar variable value."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")

        # Extract x and y values
        x_vals = extract_variable_values(result.model, "x")
        y_vals = extract_variable_values(result.model, "y")

        assert "scalar" in x_vals
        assert "scalar" in y_vals
        assert abs(x_vals["scalar"] - 5.0) < 1e-4

    @patch('src.solver.SolverFactory')
    def test_extract_indexed_variable(self, mock_factory, indexed_model):
        """Test extracting indexed variable values."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        for i in [1, 2, 3]:
            indexed_model.x[i].set_value(2.0)

        result = solve_model(indexed_model, solver_name="glpk")

        x_vals = extract_variable_values(result.model, "x")

        assert isinstance(x_vals, dict)
        # All three indices should be present
        assert len(x_vals) == 3
        # Sum should be approximately 6
        total = sum(x_vals.values())
        assert abs(total - 6.0) < 1e-4

    @patch('src.solver.SolverFactory')
    def test_extract_with_threshold(self, mock_factory, indexed_model):
        """Test extracting variables with value threshold."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        for i in [1, 2, 3]:
            indexed_model.x[i].set_value(2.0)

        result = solve_model(indexed_model, solver_name="glpk")

        x_vals = extract_variable_values(result.model, "x", threshold=1.0)

        # Values should be >= 1.0
        for val in x_vals.values():
            assert val >= 1.0 - 1e-6

    @patch('src.solver.SolverFactory')
    def test_extract_nonexistent_variable(self, mock_factory, simple_lp_model):
        """Test error handling for nonexistent variables."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")

        with pytest.raises(KeyError):
            extract_variable_values(result.model, "nonexistent_var")


class TestSerialization:
    """Tests for result serialization and loading."""

    @patch('src.solver.SolverFactory')
    def test_save_and_load_result(self, mock_factory, simple_lp_model):
        """Test saving and loading results from JSON."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = str(Path(tmpdir) / "result.json")

            # Save
            save_result(result, filepath)
            assert Path(filepath).exists()

            # Load
            loaded = load_result(filepath)

            # Verify structure
            assert loaded["status"] == result.status
            assert loaded["success"] == result.success
            if result.objective_value is not None:
                assert abs(loaded["objective_value"] - result.objective_value) < 1e-6

    @patch('src.solver.SolverFactory')
    def test_result_dict_json_serializable(self, mock_factory, simple_lp_model):
        """Test that result dict is JSON-serializable."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)
        simple_lp_model.y.set_value(0.0)

        result = solve_model(simple_lp_model, solver_name="glpk")
        result_dict = result.to_dict()

        # Should not raise
        json_str = json.dumps(result_dict)
        reloaded = json.loads(json_str)

        assert reloaded["status"] == result.status


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_unavailable_solver_fallback(self, simple_lp_model):
        """If neither the requested nor fallback solver exist we return a
        structured result rather than raising an exception.
        """
        result = solve_model(
            simple_lp_model,
            solver_name="nonexistent_solver_xyz",
            fallback_solver="also_nonexistent",
        )
        assert result.status == "solver_unavailable"
        assert not result.success
        assert result.solver_time == 0.0

    def test_both_solvers_unavailable(self, simple_lp_model):
        """Same behaviour when both names are bogus."""
        result = solve_model(
            simple_lp_model,
            solver_name="nonexistent_1",
            fallback_solver="nonexistent_2",
        )
        assert result.status == "solver_unavailable"
        assert not result.success
        assert result.message.startswith("No available solver")

    @patch('src.solver.SolverFactory')
    def test_mock_no_solver_available(self, mock_factory, simple_lp_model):
        """Simulate both primary and fallback being unavailable via mocks."""
        # factory returns a solver object whose available() is False
        fake_solver = Mock()
        fake_solver.available.return_value = False
        mock_factory.return_value = fake_solver

        result = solve_model(
            simple_lp_model,
            solver_name="any",
            fallback_solver="other",
        )
        assert result.status == "solver_unavailable"
        assert not result.success
        assert "any" in result.message and "other" in result.message

    @patch('src.solver.SolverFactory')
    def test_solve_result_repr(self, mock_factory, simple_lp_model):
        """Test string representation of SolveResult."""
        mock_solver = Mock()
        mock_solver.available.return_value = True
        mock_factory.return_value = mock_solver
        
        mock_results = Mock()
        mock_results.solver.status = "ok"
        mock_results.solver.termination_condition = "optimal"
        mock_solver.solve.return_value = mock_results
        
        simple_lp_model.x.set_value(5.0)

        result = solve_model(simple_lp_model, solver_name="glpk")
        repr_str = repr(result)

        assert "SolveResult" in repr_str
        assert result.status in repr_str




# optional integration test, only run if a real solver is detected
@pytest.mark.skipif(
    not SolverFactory("glpk").available(False) and not SolverFactory("ipopt").available(False),
    reason="No real external solver available for integration test",
)
def test_integration_real_solver(simple_lp_model):
    """Attempt a real solve using whichever solver is installed.

    This test is skipped in the CI environment when no solver is present.
    """
    result = solve_model(simple_lp_model, solver_name="glpk", fallback_solver="ipopt")
    # if a solver really ran we expect either optimal or infeasible behaviour
    assert result.status in ("optimal", "infeasible", "feasible", "unbounded")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
