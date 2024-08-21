import re
from sympy import symbols, Eq, solve, diff, integrate, sympify, Derivative, Integral, solveset, S

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