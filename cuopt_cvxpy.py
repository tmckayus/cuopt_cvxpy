import argparse
import json
import sys
import cvxpy as cp
import numpy as np
from scipy import sparse
import pickle

import time

def process_bounds(bounds_array, inf_str='inf'):
    """Efficiently process bounds array whether it's list or numpy array."""
    if isinstance(bounds_array, np.ndarray):
        # If already numpy array, just replace strings with inf values
        if bounds_array.dtype.kind in {'U', 'S'}:  # if string array
            result = np.full_like(bounds_array, np.inf, dtype=float)
            ninf_mask = bounds_array == 'ninf'
            inf_mask = bounds_array == 'inf'
            result[ninf_mask] = -np.inf
            result[~(ninf_mask | inf_mask)] = bounds_array[~(ninf_mask | inf_mask)].astype(float)
            return result
        return bounds_array
    else:
        # If list, convert directly to float array
        return np.array([float('-inf') if x == 'ninf' else 
                        float('inf') if x == 'inf' else float(x) 
                        for x in bounds_array])

def create_variables(var_types, var_lb=None, var_ub=None):
    # Check if all types are the same
    if all(t == 'C' for t in var_types):
        # All continuous - create in one call
        if var_lb is not None and var_ub is not None:
            # CVXPY expects bounds as [lower_bounds, upper_bounds]
            bounds = [var_lb, var_ub]
            return cp.Variable(len(var_types), name='x', bounds=bounds)
        else:
            return cp.Variable(len(var_types), name='x')
    elif all(t == 'I' for t in var_types):
        # All integer - create in one call
        if var_lb is not None and var_ub is not None:
            # CVXPY expects bounds as [lower_bounds, upper_bounds]
            bounds = [var_lb, var_ub]
            return cp.Variable(len(var_types), integer=True, name='x', bounds=bounds)
        else:
            return cp.Variable(len(var_types), integer=True, name='x')
    else:
        # Mixed types - create individually
        variables = []
        for i, var_type in enumerate(var_types):
            if var_lb is not None and var_ub is not None:
                bounds = [var_lb[i], var_ub[i]]
            else:
                bounds = None
                
            if var_type == 'C':
                var = cp.Variable(name=f'x_{i}', bounds=bounds)
            elif var_type == 'I':
                var = cp.Variable(integer=True, name=f'x_{i}', bounds=bounds)
            else:
                raise ValueError(f"Unknown variable type: {var_type}")
            variables.append(var)
        return variables
    
def solve_lp_from_dict(problem_dict, solver, pass_variable_bounds, matrix_variable_bounds, solver_mode, solver_method, fname="", use_service=False):
    """Create and solve LP from dictionary representation."""

    start = time.time()
    
    # Get dimensions (these are fast regardless of type)
    n_variables = len(problem_dict['variable_types'])
    n_constraints = len(problem_dict['constraint_bounds']['lower_bounds'])
    print(n_variables)
    print(n_constraints)
    
    # Process variable bounds early
    var_lb = process_bounds(problem_dict['variable_bounds']['lower_bounds'])
    var_ub = process_bounds(problem_dict['variable_bounds']['upper_bounds'])
    
    # Create variables with bounds
    if pass_variable_bounds:
        x = create_variables(problem_dict['variable_types'],
                             var_lb,
                             var_ub)
    else:
        x = create_variables(problem_dict['variable_types'])

    a = time.time()
    # Create objective (avoid unnecessary conversion if already numpy array)
    obj_coeffs = (np.array(problem_dict['objective_data']['coefficients']) 
                 if not isinstance(problem_dict['objective_data']['coefficients'], np.ndarray)
                 else problem_dict['objective_data']['coefficients'])
    obj_offset = problem_dict['objective_data']['offset']
    objective = (cp.Minimize if not problem_dict['maximize'] else cp.Maximize)(obj_coeffs @ x + obj_offset)
    print(f"objective {time.time() - a}")

    a = time.time()
    # Create constraint matrix (already efficient with CSR format)
    A = sparse.csr_matrix(
        (problem_dict['csr_constraint_matrix']['values'],
         problem_dict['csr_constraint_matrix']['indices'],
         problem_dict['csr_constraint_matrix']['offsets']),
        shape=(n_constraints, n_variables)
    )
    
    print(f"csr {time.time() - a}")    
    a = time.time()
    # Process bounds efficiently
    lower_bounds = process_bounds(problem_dict['constraint_bounds']['lower_bounds'])
    upper_bounds = process_bounds(problem_dict['constraint_bounds']['upper_bounds'])
    print(f"bounds {time.time() - a}")    

    
    a = time.time()
    # Create constraints efficiently
    constraints = []
    
    # Pre-compute masks for different constraint types
    equality_mask = (lower_bounds == upper_bounds) & (lower_bounds != -np.inf) & (upper_bounds != np.inf)
    upper_mask = (upper_bounds != np.inf) & ~equality_mask
    lower_mask = (lower_bounds != -np.inf) & ~equality_mask
    
    # Check if x is a list (mixed variable types) or single variable
    if isinstance(x, list):
        # Handle mixed variable types - create constraints manually
        for i in range(n_constraints):
            if equality_mask[i]:
                row = A[i, :]
                expr = sum(row[0, j] * x[j] for j in row.indices)
                constraints.append(expr == lower_bounds[i])
            else:
                if upper_mask[i]:
                    row = A[i, :]
                    expr = sum(row[0, j] * x[j] for j in row.indices)
                    constraints.append(expr <= upper_bounds[i])
                if lower_mask[i]:
                    row = A[i, :]
                    expr = sum(row[0, j] * x[j] for j in row.indices)
                    constraints.append(expr >= lower_bounds[i])
    else:
        # Handle uniform variable types - use vectorized operations where possible
        if np.any(equality_mask):
            eq_indices = np.where(equality_mask)[0]
            for i in eq_indices:
                constraints.append(A[i, :] @ x == lower_bounds[i])
        
        if np.any(upper_mask):
            up_indices = np.where(upper_mask)[0]
            constraints.append(A[up_indices, :] @ x <= upper_bounds[up_indices])
        
        if np.any(lower_mask):
            low_indices = np.where(lower_mask)[0]
            constraints.append(A[low_indices, :] @ x >= lower_bounds[low_indices])

    # Add variable bounds efficiently (var_lb and var_ub already processed earlier)
    if matrix_variable_bounds:
        # Pre-compute masks for variable bounds
        var_eq_mask = (var_lb == var_ub) & (var_lb != -np.inf) & (var_ub != np.inf)
        var_lb_mask = (var_lb != -np.inf) & ~var_eq_mask
        var_ub_mask = (var_ub != np.inf) & ~var_eq_mask
        
        # Add equality constraints for variables
        if np.any(var_eq_mask):
            eq_indices = np.where(var_eq_mask)[0]
            for i in eq_indices:
                constraints.append(x[i] == var_lb[i])
        
        # Add bound constraints
        if isinstance(x, list):
            # Mixed variable types - individual constraints
            if np.any(var_lb_mask):
                lb_indices = np.where(var_lb_mask)[0]
                for i in lb_indices:
                    constraints.append(x[i] >= var_lb[i])
            if np.any(var_ub_mask):
                ub_indices = np.where(var_ub_mask)[0]
                for i in ub_indices:
                    constraints.append(x[i] <= var_ub[i])
        else:
            # Uniform variable types - vectorized
            if np.any(var_lb_mask):
                constraints.append(x[var_lb_mask] >= var_lb[var_lb_mask])
            if np.any(var_ub_mask):
                constraints.append(x[var_ub_mask] <= var_ub[var_ub_mask])
    print(f"rest {time.time() - a}")
            
    # Create and solve problem
    prob = cp.Problem(objective, constraints)
    
    # Use specified solver if provided
    call_solver = time.time()
    print(f"\n\nproblem setup {call_solver - start}")
    import datetime
    print(datetime.datetime.now())
    for i in range(1):
        if solver == "CUOPT":
            try: 
                result = prob.solve(solver=solver,
                                    verbose=True,
                                    pdlp_solver_mode=solver_mode,
                                    solver_method=solver_method,
                                    use_service=use_service)
                print(prob.value)
                print(prob.status)
            except Exception:
                import traceback
                traceback.print_exc()
        else:
            try:
                result = prob.solve(solver_verbose=True)
            except Exception:
                pass
        
    done_solver = time.time()
    print(f"time in cvxpy solve {done_solver - call_solver}\n\n")
    solvetime = done_solver - call_solver
    overhead = call_solver - start
    if fname: 
        import pickle
        with open(f"{args.file}_{str(solver)}_.pickle", "wb") as f:
            pickle.dump(solver, f)
            pickle.dump([overhead, solvetime], f) 
            pickle.dump(prob.status, f)
            pickle.dump(prob.value, f)
    
    return prob, x

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process LP file with solver options.')
    
    # Required FILE argument
    parser.add_argument('file', metavar='FILE', type=str,
                       help='cuopt LP json file')
    
    # Optional solver argument
    parser.add_argument('-s', '--solver', type=str, default=None,
                       help='solver to use (default: None)')
    
    # Optional boolean argument
    parser.add_argument('--pass_variable_bounds', action="store_true", default=False,
                        help='Pass variable bounds as extra arguments for CUOPT. Default is False.')

    parser.add_argument('--matrix_variable_bounds', action="store_true", default=False,
                        help='Add variable bounds to the constraint matrix. Default is False.')

    parser.add_argument("--use_service", action="store_true", default=False)
    
    # 0 Stable1
    # 1 Stable2
    # 2 Methodical1
    # 3 Fast1
    parser.add_argument('-m', "--solver_mode", type=str, default="Stable2",
                        help='PDLP solver mode for CUOPT. Default is Stable2')
                        
    # 0 concurrent
    # 1 PDLP
    # 2 DualSimplex
    parser.add_argument('-e', "--solver_method", type=str, default="",
                        help='solver mode for CUOPT.')    
    args = parser.parse_args()    
    
    # Read problem from JSON file
    if args.file.endswith(".pickle"):
        import pickle
        with open(args.file, 'rb') as f:
            problem_dict = pickle.load(f)
    else:            
        with open(args.file, "r") as f:
            problem_dict = json.load(f)
    
    # Solve problem
    start_time = time.time()
    try:
        prob, x = solve_lp_from_dict(problem_dict, args.solver, args.pass_variable_bounds, args.matrix_variable_bounds, args.solver_mode, args.solver_method, args.file, args.use_service)
    except Exception:
        import traceback
        traceback.print_exc()
    
    # Print results
    print('Status:', prob.status)
    print('Optimal value:', prob.value)
    
    # Handle solution printing for both uniform and mixed variable types
    if isinstance(x, list):
        # Mixed variable types - extract only non-zero values
        solution = [(i, var.value) for i, var in enumerate(x) if var.value is not None and abs(var.value) > 1e-10]
        print('Solution (non-zero variables):', solution)
    else:
        # Uniform variable types - single variable
        print('Solution:', x.value)



