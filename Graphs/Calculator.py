import re
from sympy import symbols, Eq, solve, diff, integrate, sympify, Derivative, Integral, solveset, S, SympifyError
from sympy.parsing.latex import parse_latex

def extract_and_solve(equation, target_variable):
    # Extract all variables in the equation
    variables = re.findall(r'[a-zA-Z_]\w*', equation)

    # Convert to sympy symbols
    symbol_dict = {var: symbols(var) for var in variables}
    
    # Replace variables in the equation with sympy symbols
    for var in variables:
        equation = equation.replace(var, f'{symbol_dict[var]}')

    # Parse the equation into a sympy expression
    lhs, rhs = equation.split('=')
    eq = Eq(sympify(lhs), sympify(rhs))

    # Solve for the target variable
    solution = solve(eq, symbol_dict[target_variable])

    return solution[0]


def clean_latex(equation_latex):
    """
    Cleans the LaTeX string by removing unnecessary delimiters like \\( and \\).
    """
    # Remove LaTeX math delimiters (\\( and \\))
    return equation_latex.replace('\\(', '').replace('\\)', '')

def solve_physics_equation_with_latex_list(equation_latex, target_variable_latex, variables_list):
    """
    Solves a physics equation given in LaTeX format for a target variable, with known variables provided as a list of strings.

    Parameters:
    - equation_latex: The equation as a LaTeX string, e.g., r'F = m \times a'
    - target_variable_latex: The target variable as a LaTeX string, e.g., r'a'
    - variables_list: A list of known variables and their values in string format, e.g., ['F=10', 'm=2'].

    Returns:
    - The solution for the target variable.
    """

    # Clean the LaTeX equation and target variable
    equation_latex = clean_latex(equation_latex)
    target_variable_latex = clean_latex(target_variable_latex)

    # Parse the target variable from LaTeX to SymPy
    try:
        target_var_sympy = parse_latex(target_variable_latex)
    except Exception as e:
        print(f"Error parsing target variable LaTeX '{target_variable_latex}': {e}")
        return None

    # Create a dictionary to hold variable names and values
    known_values = {}

    # Process each variable in the list
    for var in variables_list:
        var = var.strip()
        if '=' in var:
            name, value = var.split('=')
            name = name.strip()
            value = value.strip()

            # Parse the variable name and value from LaTeX to SymPy
            try:
                name_sympy = parse_latex(name)
            except Exception as e:
                print(f"Error parsing variable name LaTeX '{name}': {e}")
                continue

            try:
                # Try converting value to float, otherwise use SymPy parsing
                value_numeric = float(value)
                known_values[name_sympy] = value_numeric
            except ValueError:
                try:
                    # Parse complex LaTeX values like pi, fractions, etc.
                    value_sympy = parse_latex(value)
                    known_values[name_sympy] = value_sympy
                except Exception as e:
                    print(f"Error parsing variable value LaTeX '{value}': {e}")
                    continue

    # Parse the equation from LaTeX to SymPy
    try:
        equation_latex_clean = clean_latex(equation_latex)
        if '=' in equation_latex_clean:
            lhs_latex, rhs_latex = equation_latex_clean.split('=')
            lhs_sympy = parse_latex(lhs_latex.strip())
            rhs_sympy = parse_latex(rhs_latex.strip())
            equation_sympy = Eq(lhs_sympy, rhs_sympy)
        else:
            print("Equation does not contain an equality sign.")
            return None
    except Exception as e:
        print(f"Error parsing equation LaTeX '{equation_latex}': {e}")
        return None

    # Substitute known values into the equation
    equation_substituted = equation_sympy.subs(known_values)

    # Solve for the target variable
    try:
        solution = solve(equation_substituted, target_var_sympy)
    except Exception as e:
        print(f"Error solving the equation: {e}")
        return None

    if not solution:
        print("No solution found.")
        return None

    # Return the first solution
    return solution[0]


def solve_physics_equation_with_latex(equation_latex, target_variable_latex, variables_latex):
    """
    Solves a physics equation given in LaTeX format for a target variable, with known variables also in LaTeX format.

    Parameters:
    - equation_latex: The equation as a LaTeX string, e.g., r'F = m \times a'
    - target_variable_latex: The target variable as a LaTeX string, e.g., r'a'
    - variables_latex: A string of known variables and their values in LaTeX format, separated by commas,
      e.g., r'F=10, m=2'

    Returns:
    - The solution for the target variable.
    """

    # Parse the target variable from LaTeX to SymPy
    try:
        target_var_sympy = parse_latex(target_variable_latex)
    except Exception as e:
        print(f"Error parsing target variable LaTeX '{target_variable_latex}': {e}")
        return None

    # Split the variables string into individual assignments
    variables_list = variables_latex.split(',')

    # Create a dictionary to hold variable names and values
    known_values = {}

    for var in variables_list:
        var = var.strip()
        if '=' in var:
            name_latex, value_latex = var.split('=')
            name_latex = name_latex.strip()
            value_latex = value_latex.strip()

            # Parse the variable name and value from LaTeX to SymPy
            try:
                name_sympy = parse_latex(name_latex)
            except Exception as e:
                print(f"Error parsing variable name LaTeX '{name_latex}': {e}")
                continue

            try:
                value_sympy = parse_latex(value_latex)
                # Try to evaluate the value numerically
                value_numeric = float(value_sympy.evalf())
                known_values[name_sympy] = value_numeric
            except Exception:
                # Keep as symbolic expression
                known_values[name_sympy] = value_sympy
        else:
            # Handle variables without assignments
            known_values[var] = None  # or raise an error as needed

    # Parse the equation from LaTeX to SymPy
    try:
        # Remove any dollar signs from LaTeX
        equation_latex_clean = equation_latex.replace('$', '')
        # Split the equation into lhs and rhs
        if '=' in equation_latex_clean:
            lhs_latex, rhs_latex = equation_latex_clean.split('=')
            lhs_sympy = parse_latex(lhs_latex.strip())
            rhs_sympy = parse_latex(rhs_latex.strip())
            equation_sympy = Eq(lhs_sympy, rhs_sympy)
        else:
            print("Equation does not contain an equality sign.")
            return None
    except Exception as e:
        print(f"Error parsing equation LaTeX '{equation_latex}': {e}")
        return None

    # Substitute known values into the equation
    equation_substituted = equation_sympy.subs(known_values)

    # Solve for the target variable
    solution = solve(equation_substituted, target_var_sympy)

    if not solution:
        print("No solution found.")
        return None

    # Return the first solution
    return solution[0]


def solve_physics_equation_with_string(equation, target_variable, variables_str):
    # Split the string into individual variable assignments
    variables_list = variables_str.split(',')
    
    # Create a dictionary to hold variable names and values
    known_values = {}
    
    for var in variables_list:
        # Remove any leading/trailing whitespace
        var = var.strip()
        if '=' in var:
            # Split the variable assignment into name and value
            name, value = var.split('=')
            name = name.strip()
            value = value.strip()
            try:
                # Try to convert the value to a float
                value = float(value)
            except ValueError:
                try:
                    # If conversion fails, attempt to sympify (e.g., for 'pi', '2*pi')
                    value = sympify(value)
                except SympifyError:
                    # Keep it as a string if sympify fails
                    pass
            known_values[name] = value
        else:
            # Handle cases where the variable is given without an assignment
            # e.g., 'x' instead of 'x=5'
            known_values[var] = None  # or raise an error as needed

    # Call the existing function with the known values
    result = solve_physics_equation(equation, target_variable, **known_values)
    return result


def solve_physics_equation(equation, target_variable, **known_values):
    # Replace known values in the equation
    for var, value in known_values.items():
        equation = equation.replace(var, str(value))

    # Solve for the target variable
    result = extract_and_solve(equation, target_variable)
    return result

def differentiate(equation, variable, order=1):
    variable_symbol = symbols(variable)
    differentiated_eq = sympify(equation)
    
    for _ in range(order):
        differentiated_eq = diff(differentiated_eq, variable_symbol)
    
    return differentiated_eq

def integrate_equation(equation, variable, limits=None):
    variable_symbol = symbols(variable)
    integrated_eq = sympify(equation)
    
    if limits:
        lower_limit, upper_limit = limits
        integrated_eq = integrate(integrated_eq, (variable_symbol, lower_limit, upper_limit))
    else:
        integrated_eq = integrate(integrated_eq, variable_symbol)
    
    return integrated_eq

def solve_system(equations, target_variables):
    symbols_list = symbols(target_variables)
    sympy_eqs = []
    
    for eq in equations:
        lhs, rhs = eq.split('=')
        sympy_eqs.append(Eq(sympify(lhs), sympify(rhs)))
    
    solution = solve(sympy_eqs, symbols_list)
    
    return solution

def derivative_function(equation, variable):
    variable_symbol = symbols(variable)
    return Derivative(sympify(equation), variable_symbol).doit()

def integral_function(equation, variable, limits=None):
    variable_symbol = symbols(variable)
    if limits:
        lower_limit, upper_limit = limits
        return Integral(sympify(equation), (variable_symbol, lower_limit, upper_limit)).doit()
    else:
        return Integral(sympify(equation), variable_symbol).doit()