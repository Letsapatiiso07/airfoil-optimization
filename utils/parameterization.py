import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class PARSECAirfoil:
    """
    PARSEC airfoil parameterization method
    11 parameters define the airfoil geometry:
    - rle: leading edge radius
    - x_up, y_up: upper crest location
    - y_xxup: upper crest curvature
    - x_lo, y_lo: lower crest location
    - y_xxlo: lower crest curvature
    - yte: trailing edge thickness
    - alpha_te: trailing edge direction
    - beta_te: trailing edge wedge angle
    - dz_te: trailing edge offset
    """
    
    def __init__(self, params: np.ndarray):
        """
        Initialize PARSEC airfoil with 11 parameters
        
        Args:
            params: Array of 11 PARSEC parameters
        """
        self.params = params
        self.rle = params[0]
        self.x_up = params[1]
        self.y_up = params[2]
        self.y_xxup = params[3]
        self.x_lo = params[4]
        self.y_lo = params[5]
        self.y_xxlo = params[6]
        self.yte = params[7]
        self.alpha_te = params[8]
        self.beta_te = params[9]
        self.dz_te = params[10]
        
    def generate_coordinates(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate airfoil coordinates from PARSEC parameters
        
        Args:
            n_points: Number of points on each surface
            
        Returns:
            x, y coordinates of airfoil
        """
        # Generate x coordinates (cosine spacing for better resolution at LE/TE)
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        # Calculate upper and lower surface
        y_upper = self._calculate_surface(x, 'upper')
        y_lower = self._calculate_surface(x, 'lower')
        
        # Combine upper and lower surfaces
        x_full = np.concatenate([x[::-1], x[1:]])
        y_full = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_full, y_full
    
    def _calculate_surface(self, x: np.ndarray, surface: str) -> np.ndarray:
        """
        Calculate y coordinates for upper or lower surface
        
        Args:
            x: x coordinates
            surface: 'upper' or 'lower'
            
        Returns:
            y coordinates
        """
        if surface == 'upper':
            x_crest = self.x_up
            y_crest = self.y_up
            y_xx = self.y_xxup
        else:
            x_crest = self.x_lo
            y_crest = self.y_lo
            y_xx = self.y_xxlo
        
        # Calculate coefficients for 6th order polynomial
        C = np.zeros(7)
        
        # Leading edge radius constraint
        C[0] = np.sqrt(2 * self.rle)
        
        # Build constraint matrix
        A = np.array([
            [x_crest**1.5, x_crest**2.5, x_crest**3.5, x_crest**4.5, x_crest**5.5],
            [1.5*np.sqrt(x_crest), 2.5*x_crest**1.5, 3.5*x_crest**2.5, 4.5*x_crest**3.5, 5.5*x_crest**4.5],
            [0.75/np.sqrt(x_crest), 3.75*np.sqrt(x_crest), 8.75*x_crest**1.5, 15.75*x_crest**2.5, 24.75*x_crest**3.5],
            [1.0**1.5, 1.0**2.5, 1.0**3.5, 1.0**4.5, 1.0**5.5],
            [1.5, 2.5, 3.5, 4.5, 5.5]
        ])
        
        # Constraint vector
        sign = 1 if surface == 'upper' else -1
        b = np.array([
            y_crest - C[0]*np.sqrt(x_crest),
            0,  # Zero slope at crest
            y_xx,
            sign * self.yte/2 + sign * self.dz_te - C[0],
            np.tan(self.alpha_te - sign * self.beta_te/2)
        ])
        
        # Solve for remaining coefficients
        try:
            C[1:6] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            C[1:6] = np.zeros(5)
        
        # Calculate y coordinates
        y = (C[0]*np.sqrt(x) + C[1]*x**1.5 + C[2]*x**2.5 + 
             C[3]*x**3.5 + C[4]*x**4.5 + C[5]*x**5.5)
        
        if surface == 'lower':
            y = -y
            
        return y
    
    def save_dat_file(self, filename: str, n_points: int = 150):
        """
        Save airfoil coordinates to .dat file (XFoil format)
        
        Args:
            filename: Output filename
            n_points: Number of points per surface
        """
        x, y = self.generate_coordinates(n_points)
        
        with open(filename, 'w') as f:
            f.write(f"PARSEC Airfoil\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:10.6f}  {yi:10.6f}\n")
    
    def plot(self, ax=None, **kwargs):
        """
        Plot the airfoil shape
        
        Args:
            ax: Matplotlib axis
            **kwargs: Additional plot arguments
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        x, y = self.generate_coordinates()
        ax.plot(x, y, **kwargs)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x/c')
        ax.set_ylabel('y/c')
        ax.set_title('Airfoil Geometry')
        
        return ax


class CST_Airfoil:
    """
    Class-Shape Transformation (CST) parameterization
    Alternative to PARSEC with different mathematical basis
    """
    
    def __init__(self, au: np.ndarray, al: np.ndarray, n1: float = 0.5, n2: float = 1.0, dte: float = 0.0):
        """
        Initialize CST airfoil
        
        Args:
            au: Upper surface coefficients
            al: Lower surface coefficients
            n1: Class function parameter (leading edge)
            n2: Class function parameter (trailing edge)
            dte: Trailing edge thickness
        """
        self.au = au
        self.al = al
        self.n1 = n1
        self.n2 = n2
        self.dte = dte
        
    def _class_function(self, x: np.ndarray) -> np.ndarray:
        """Class function C(x) = x^n1 * (1-x)^n2"""
        return x**self.n1 * (1 - x)**self.n2
    
    def _shape_function(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Bernstein polynomial shape function"""
        n = len(a) - 1
        S = np.zeros_like(x)
        for i, ai in enumerate(a):
            K = np.math.comb(n, i)
            S += ai * K * x**i * (1 - x)**(n - i)
        return S
    
    def generate_coordinates(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate airfoil coordinates"""
        beta = np.linspace(0, np.pi, n_points)
        x = 0.5 * (1 - np.cos(beta))
        
        C = self._class_function(x)
        Su = self._shape_function(x, self.au)
        Sl = self._shape_function(x, self.al)
        
        y_upper = C * Su + x * self.dte
        y_lower = -(C * Sl - x * self.dte)
        
        x_full = np.concatenate([x[::-1], x[1:]])
        y_full = np.concatenate([y_upper[::-1], y_lower[1:]])
        
        return x_full, y_full


def generate_random_parsec_params(n_samples: int = 1) -> np.ndarray:
    """
    Generate random but feasible PARSEC parameters
    
    Args:
        n_samples: Number of parameter sets to generate
        
    Returns:
        Array of shape (n_samples, 11) with PARSEC parameters
    """
    params = np.zeros((n_samples, 11))
    
    # Parameter bounds (reasonable ranges for aerodynamic airfoils)
    bounds = {
        'rle': (0.005, 0.025),      # Leading edge radius
        'x_up': (0.25, 0.45),        # Upper crest x
        'y_up': (0.04, 0.08),        # Upper crest y
        'y_xxup': (-0.5, -0.1),      # Upper curvature
        'x_lo': (0.30, 0.50),        # Lower crest x
        'y_lo': (-0.05, -0.02),      # Lower crest y (negative)
        'y_xxlo': (0.1, 0.5),        # Lower curvature
        'yte': (0.001, 0.005),       # Trailing edge thickness
        'alpha_te': (-0.15, 0.15),   # TE direction
        'beta_te': (0.05, 0.20),     # TE wedge angle
        'dz_te': (-0.002, 0.002)     # TE offset
    }
    
    for i, (pmin, pmax) in enumerate(bounds.values()):
        params[:, i] = np.random.uniform(pmin, pmax, n_samples)
    
    return params


def load_airfoil_database() -> dict:
    """
    Load common airfoil shapes for reference
    Returns dictionary with name: PARSEC_params
    """
    # Pre-computed PARSEC parameters for standard airfoils
    database = {
        'NACA0012': np.array([0.0126, 0.35, 0.06, -0.35, 0.35, -0.06, 0.35, 0.0012, 0.0, 0.10, 0.0]),
        'NACA2412': np.array([0.0150, 0.30, 0.075, -0.40, 0.40, -0.04, 0.30, 0.0015, -0.05, 0.12, 0.0]),
        'NACA4412': np.array([0.0160, 0.28, 0.095, -0.45, 0.42, -0.03, 0.25, 0.0018, -0.08, 0.14, 0.0]),
    }
    
    return database


if __name__ == "__main__":
    # Example usage
    print("Airfoil Parameterization Module")
    print("=" * 50)
    
    # Generate random airfoil
    params = generate_random_parsec_params(1)[0]
    airfoil = PARSECAirfoil(params)
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot random airfoil
    airfoil.plot(ax=axes[0, 0], linewidth=2, color='blue')
    axes[0, 0].set_title('Random PARSEC Airfoil')
    
    # Load and plot standard airfoils
    database = load_airfoil_database()
    for idx, (name, params) in enumerate(database.items()):
        row = (idx + 1) // 2
        col = (idx + 1) % 2
        airfoil_std = PARSECAirfoil(params)
        airfoil_std.plot(ax=axes[row, col], linewidth=2, color='red')
        axes[row, col].set_title(f'{name} Airfoil')
    
    plt.tight_layout()
    plt.savefig('/home/claude/airfoil_optimization/data/example_airfoils.png', dpi=150)
    print("\nExample airfoils saved to data/example_airfoils.png")
    
    # Save random airfoil to .dat file
    airfoil.save_dat_file('/home/claude/airfoil_optimization/data/test_airfoil.dat')
    print("Test airfoil saved to data/test_airfoil.dat")