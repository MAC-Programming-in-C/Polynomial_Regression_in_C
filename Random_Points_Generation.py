import numpy as np
import random
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def generate_points_and_polynomial():
    n = int(input("Enter the polynomial order n: "))
    x_vals = np.array([random.uniform(-10, 10) for _ in range(n)])
    y_vals = np.array([random.uniform(-10, 10) for _ in range(n)])
    coeffs = np.polyfit(x_vals, y_vals, n)
    x = sp.Symbol('x')
    poly_expr = sum(sp.Float(coeffs[i]) * x**(n - i) for i in range(n + 1))

    print("\nGenerated Points (x, y):")
    for xi, yi in zip(x_vals, y_vals):
        print(f"({xi:.4f}, {yi:.4f})")
    
    print("\npoint list:")
    for xi, yi in zip(x_vals, y_vals):
        print(f"({xi}, {yi})")

    print("\nPolynomial coefficients:")
    print(coeffs)

    print("\nPolynomial (Symbolic):")
    print(sp.expand(poly_expr))

    return n, coeffs, poly_expr


def generate_synthetic_data(poly_expr, x_range=(-5, 5), n_points=100, noise_std=1.0):
    x = np.linspace(x_range[0], x_range[1], n_points)
    y_true = np.array([poly_expr.subs('x', val) for val in x], dtype=float)
    y_noisy = y_true + np.random.normal(0, noise_std, size=len(x))
    return x, y_noisy, y_true


def build_curve_fit_model(order):
    def model(x, *params):
        return sum(params[i] * x**(order - i) for i in range(order + 1))
    return model


def main():
    #Generate polynomial from user input
    n, coeffs, poly_expr = generate_points_and_polynomial()

    #Generate synthetic noisy data
    x_data, y_noisy, y_true = generate_synthetic_data(poly_expr, n_points=200, noise_std=2.0)

    model = build_curve_fit_model(n)
    initial_guess = np.ones(n + 1)

    popt, pcov = curve_fit(model, x_data, y_noisy, p0=initial_guess)
    y_fit = model(x_data, *popt)

    print("\nFitted Parameters:")
    print(popt)

    
    np_polyfit_coeffs = np.polyfit(x_data, y_noisy, n)
    print("\nPython np.polyfit coefficients:")
    print(np_polyfit_coeffs)

    filename = "synthetic_polynomial_data.txt"
    with open(filename, "w") as f:
        f.write(f"Polynomial order: {n}\n")
        f.write(f"Number of data points: {len(x_data)}\n")
        f.write("x\ty_noisy\ty_true\n")
        for xi, yi_noisy, yi_true in zip(x_data, y_noisy, y_true):
            f.write(f"{xi:.6f}\t{yi_noisy:.6f}\t{yi_true:.6f}\n")

        # Add results to file
        f.write("\nFitted parameters (curve_fit):\n")
        f.write(", ".join(str(c) for c in popt) + "\n")

        f.write("\nPython np.polyfit coefficients:\n")
        f.write(", ".join(str(c) for c in np_polyfit_coeffs) + "\n")

        f.write("\n# Desmos-ready points (x, y):\n")
        for xi, yi in zip(x_data, y_noisy):
            f.write(f"({xi},{yi})\n")


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_noisy, label="Noisy data", s=15)
    plt.plot(x_data, y_true, label="True polynomial", linewidth=2)
    plt.plot(x_data, y_fit, label="Fitted curve", linestyle='--', linewidth=2)
    plt.title(f"Polynomial Order {n}: True vs Fitted Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run everything
main()
