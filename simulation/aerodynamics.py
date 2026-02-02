import numpy as np
from typing import Dict, Tuple, Optional
import subprocess
import os
from pathlib import Path


class AirfoilSimulator:
    """
    Simplified aerodynamic simulator for airfoil analysis
    Uses potential flow theory with boundary layer corrections
    """
    
    def __init__(self, n_panels: int = 200):
        """
        Initialize simulator
        
        Args:
            n_panels: Number of panels for discretization
        """
        self.n_panels = n_panels
        self.rho = 1.225  # Air density at sea level (kg/m³)
        
    def analyze(self, x: np.ndarray, y: np.ndarray, 
                alpha: float, reynolds: float, mach: float = 0.0) -> Dict:
        """
        Perform aerodynamic analysis on airfoil
        
        Args:
            x, y: Airfoil coordinates
            alpha: Angle of attack (degrees)
            reynolds: Reynolds number
            mach: Mach number
            
        Returns:
            Dictionary with performance metrics
        """
        # Convert alpha to radians
        alpha_rad = np.radians(alpha)
        
        # Panel method for inviscid flow
        cl_inviscid, cp_dist = self._panel_method(x, y, alpha_rad)
        
        # Boundary layer corrections
        cd_viscous = self._estimate_viscous_drag(x, y, reynolds, cl_inviscid)
        cd_wave = self._estimate_wave_drag(mach, cl_inviscid)
        
        # Total drag
        cd = cd_viscous + cd_wave
        
        # Moment coefficient (about quarter chord)
        cm = self._calculate_moment(x, y, cp_dist)
        
        # Lift-to-drag ratio
        ld_ratio = cl_inviscid / cd if cd > 0 else 0
        
        return {
            'cl': cl_inviscid,
            'cd': cd,
            'cm': cm,
            'ld_ratio': ld_ratio,
            'cp_distribution': cp_dist,
            'converged': True
        }
    
    def _panel_method(self, x: np.ndarray, y: np.ndarray, 
                     alpha: float) -> Tuple[float, np.ndarray]:
        """
        Simplified panel method for inviscid flow
        Uses thin airfoil theory approximation
        """
        # Calculate camber line
        n = len(x) // 2
        # Ensure equal lengths
        x_upper = x[:n]
        x_lower = x[n:2*n]
        y_upper = y[:n]
        y_lower = y[n:2*n]
        
        # Make sure arrays are same length
        min_len = min(len(x_upper), len(x_lower))
        x_camber = (x_upper[:min_len] + x_lower[:min_len]) / 2
        y_camber = (y_upper[:min_len] + y_lower[:min_len]) / 2
        
        # Calculate camber slope with error handling
        if len(x_camber) > 2:
            # Filter out duplicate x values
            unique_mask = np.diff(x_camber, prepend=x_camber[0]-1) > 1e-10
            x_camber_clean = x_camber[unique_mask]
            y_camber_clean = y_camber[unique_mask]
            
            if len(x_camber_clean) > 2:
                dy_dx = np.gradient(y_camber_clean, x_camber_clean)
                # Use trapezoid instead of deprecated trapz
                integral = np.trapezoid(dy_dx, x_camber_clean)
            else:
                integral = 0
        else:
            integral = 0
        
        # Thin airfoil theory CL
        cl = 2 * np.pi * (alpha + integral)
        
        # Estimate pressure distribution
        cp = self._estimate_cp_distribution(x, y, alpha, cl)
        
        return cl, cp
    
    def _estimate_cp_distribution(self, x: np.ndarray, y: np.ndarray, 
                                  alpha: float, cl: float) -> np.ndarray:
        """
        Estimate pressure coefficient distribution
        """
        n = len(x)
        cp = np.zeros(n)
        
        # Simple approximation based on surface curvature and angle
        for i in range(n):
            # Local curvature effect
            if i > 0 and i < n - 1:
                dx_sq = (x[i+1] - x[i-1])**2 + 1e-6
                curvature = (y[i+1] - 2*y[i] + y[i-1]) / dx_sq
            else:
                curvature = 0
            
            # Upper vs lower surface
            is_upper = i < n // 2
            
            # Pressure coefficient approximation
            if is_upper:
                cp[i] = -2 * alpha - curvature * 0.5 - cl * (1 - x[i])
            else:
                cp[i] = 2 * alpha + curvature * 0.5 + cl * (1 - x[i]) * 0.5
        
        return cp
    
    def _estimate_viscous_drag(self, x: np.ndarray, y: np.ndarray, 
                              reynolds: float, cl: float) -> float:
        """
        Estimate viscous drag using flat plate approximation
        """
        # Turbulent flat plate skin friction
        cf = 0.074 / (reynolds ** 0.2) if reynolds > 0 else 0.01
        
        # Form factor for airfoil (accounts for thickness)
        thickness = self._calculate_thickness(x, y)
        form_factor = 1 + 2 * thickness + 60 * thickness**4
        
        # Viscous drag coefficient
        cd_viscous = cf * form_factor
        
        # Induced drag
        cd_induced = cl**2 / (np.pi * 8)  # Assuming aspect ratio ~8
        
        return cd_viscous + cd_induced
    
    def _estimate_wave_drag(self, mach: float, cl: float) -> float:
        """
        Estimate wave drag for compressible flow
        """
        if mach < 0.3:
            return 0.0
        
        # Critical Mach number approximation
        mach_crit = 0.8 - cl / 10
        
        if mach > mach_crit:
            # Simple wave drag model
            cd_wave = 20 * (mach - mach_crit)**4
            return cd_wave
        
        return 0.0
    
    def _calculate_thickness(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate maximum thickness ratio
        """
        n = len(x) // 2
        min_len = min(n, len(x) - n)
        if min_len > 0:
            thickness = np.max(np.abs(y[:min_len] - y[n:n+min_len]))
            return thickness
        return 0.0
    
    def _calculate_moment(self, x: np.ndarray, y: np.ndarray, 
                         cp: np.ndarray) -> float:
        """
        Calculate moment coefficient about quarter chord
        """
        # Quarter chord location
        x_ref = 0.25
        
        # Integrate moment contribution
        cm = 0.0
        for i in range(len(x) - 1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            x_mid = (x[i] + x[i+1]) / 2
            cp_mid = (cp[i] + cp[i+1]) / 2
            
            # Moment arm
            r = x_mid - x_ref
            
            # Contribution to moment
            cm += cp_mid * r * np.sqrt(dx**2 + dy**2)
        
        return cm
    
    def sweep_analysis(self, x: np.ndarray, y: np.ndarray,
                      alphas: np.ndarray, reynolds: float,
                      mach: float = 0.0) -> Dict:
        """
        Perform analysis over range of angles of attack
        
        Args:
            x, y: Airfoil coordinates
            alphas: Array of angles of attack
            reynolds: Reynolds number
            mach: Mach number
            
        Returns:
            Dictionary with arrays of performance metrics
        """
        results = {
            'alpha': alphas,
            'cl': [],
            'cd': [],
            'cm': [],
            'ld_ratio': []
        }
        
        for alpha in alphas:
            res = self.analyze(x, y, alpha, reynolds, mach)
            results['cl'].append(res['cl'])
            results['cd'].append(res['cd'])
            results['cm'].append(res['cm'])
            results['ld_ratio'].append(res['ld_ratio'])
        
        # Convert to arrays
        for key in ['cl', 'cd', 'cm', 'ld_ratio']:
            results[key] = np.array(results[key])
        
        return results


class FlightConditions:
    """
    Define flight conditions for optimization
    """
    
    def __init__(self, altitude: float = 0, velocity: float = 50,
                 temperature: float = 288.15):
        """
        Initialize flight conditions
        
        Args:
            altitude: Altitude in meters
            velocity: Velocity in m/s
            temperature: Temperature in Kelvin
        """
        self.altitude = altitude
        self.velocity = velocity
        self.temperature = temperature
        
        # Atmospheric properties
        self.pressure = self._calculate_pressure()
        self.density = self._calculate_density()
        self.viscosity = self._calculate_viscosity()
        
        # Derived quantities
        self.reynolds = self._calculate_reynolds()
        self.mach = self._calculate_mach()
        
    def _calculate_pressure(self) -> float:
        """Calculate atmospheric pressure using standard atmosphere"""
        p0 = 101325  # Sea level pressure (Pa)
        if self.altitude < 11000:
            T0 = 288.15
            L = 0.0065  # Temperature lapse rate
            p = p0 * (1 - L * self.altitude / T0) ** 5.2561
        else:
            # Stratosphere
            p = 22632 * np.exp(-0.0001577 * (self.altitude - 11000))
        return p
    
    def _calculate_density(self) -> float:
        """Calculate air density"""
        R = 287.05  # Gas constant for air
        return self.pressure / (R * self.temperature)
    
    def _calculate_viscosity(self) -> float:
        """Calculate dynamic viscosity using Sutherland's law"""
        mu0 = 1.716e-5  # Reference viscosity at T0
        T0 = 273.15
        S = 110.4  # Sutherland constant
        
        mu = mu0 * (self.temperature / T0)**1.5 * (T0 + S) / (self.temperature + S)
        return mu
    
    def _calculate_reynolds(self, chord: float = 1.0) -> float:
        """Calculate Reynolds number"""
        return self.density * self.velocity * chord / self.viscosity
    
    def _calculate_mach(self) -> float:
        """Calculate Mach number"""
        gamma = 1.4  # Specific heat ratio for air
        R = 287.05
        a = np.sqrt(gamma * R * self.temperature)  # Speed of sound
        return self.velocity / a


def create_performance_database(n_samples: int = 1000,
                               output_file: str = 'airfoil_database.npz'):
    """
    Create database of airfoil geometries and performance
    
    Args:
        n_samples: Number of airfoils to simulate
        output_file: Output file path
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.parameterization import generate_random_parsec_params, PARSECAirfoil
    
    print(f"Generating database with {n_samples} airfoils...")
    
    # Generate random airfoils
    params_list = generate_random_parsec_params(n_samples)
    
    # Storage arrays
    all_params = []
    all_cl = []
    all_cd = []
    all_cm = []
    all_ld = []
    
    # Simulation setup
    simulator = AirfoilSimulator()
    conditions = FlightConditions(altitude=0, velocity=50)
    alphas = np.array([0, 2, 4, 6])  # Test angles
    
    for i, params in enumerate(params_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_samples} airfoils...")
        
        # Create airfoil
        airfoil = PARSECAirfoil(params)
        x, y = airfoil.generate_coordinates(100)
        
        # Run sweep
        results = simulator.sweep_analysis(x, y, alphas, 
                                          conditions.reynolds, 
                                          conditions.mach)
        
        # Store results
        all_params.append(params)
        all_cl.append(results['cl'])
        all_cd.append(results['cd'])
        all_cm.append(results['cm'])
        all_ld.append(results['ld_ratio'])
    
    # Save to file
    np.savez_compressed(output_file,
                       params=np.array(all_params),
                       cl=np.array(all_cl),
                       cd=np.array(all_cd),
                       cm=np.array(all_cm),
                       ld_ratio=np.array(all_ld),
                       alphas=alphas)
    
    print(f"\nDatabase saved to {output_file}")
    print(f"  Shape: {len(all_params)} airfoils × {len(alphas)} angles")