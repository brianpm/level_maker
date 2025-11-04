"""
Convert hybrid-sigma pressure levels to reference heights for atmosphere models.

Uses isothermal atmosphere assumption with 7 km scale height.
"""

import numpy as np
import netCDF4 as nc
from scipy.ndimage import uniform_filter1d


def compute_pressure_levels(hyam, hybm, hyai, hybi, PS, P0):
    """
    Compute pressure at layer midpoints and interfaces.
    
    Parameters:
    -----------
    hyam : array_like
        Hybrid A coefficient at layer midpoints
    hybm : array_like
        Hybrid B coefficient at layer midpoints
    hyai : array_like
        Hybrid A coefficient at layer interfaces
    hybi : array_like
        Hybrid B coefficient at layer interfaces
    PS : float
        Surface pressure (Pa)
    P0 : float
        Reference pressure (Pa)
    
    Returns:
    --------
    p_mid : ndarray
        Pressure at layer midpoints (Pa)
    p_int : ndarray
        Pressure at layer interfaces (Pa)
    """
    # Pressure at layer midpoints
    p_mid = hyam * P0 + hybm * PS
    
    # Pressure at layer interfaces
    p_int = hyai * P0 + hybi * PS
    
    return p_mid, p_int


def pressure_to_height(pressure, scale_height=7000.0, p_surface=101325.0):
    """
    Convert pressure to height using isothermal atmosphere.
    
    Parameters:
    -----------
    pressure : array_like
        Pressure values (Pa)
    scale_height : float, optional
        Atmospheric scale height (m), default 7000 m (7 km)
    p_surface : float, optional
        Surface pressure (Pa), default 101325 Pa
    
    Returns:
    --------
    height : ndarray
        Height values (m)
    """
    # Isothermal atmosphere: z = H * ln(p_surface / p)
    # where H is the scale height
    height = scale_height * np.log(p_surface / pressure)
    
    return height


def smooth_heights(heights, window_size=3):
    """
    Apply smoothing to remove abrupt changes in height profile.
    
    Parameters:
    -----------
    heights : array_like
        Height values to smooth (m)
    window_size : int, optional
        Size of smoothing window, default 3
    
    Returns:
    --------
    smoothed : ndarray
        Smoothed height values (m)
    """
    # Use uniform filter for simple moving average
    # mode='nearest' to handle boundaries
    smoothed = uniform_filter1d(heights, size=window_size, mode='nearest')
    
    return smoothed


def convert_sigma_to_height(hyam, hybm, hyai, hybi, PS, P0, 
                            scale_height=7000.0, 
                            smooth_window=3,
                            apply_smoothing=True):
    """
    Convert hybrid-sigma levels to reference heights.
    
    Parameters:
    -----------
    hyam : array_like
        Hybrid A coefficient at layer midpoints (nlev,)
    hybm : array_like
        Hybrid B coefficient at layer midpoints (nlev,)
    hyai : array_like
        Hybrid A coefficient at layer interfaces (nlev+1,)
    hybi : array_like
        Hybrid B coefficient at layer interfaces (nlev+1,)
    PS : float
        Surface pressure (Pa)
    P0 : float
        Reference pressure (Pa)
    scale_height : float, optional
        Atmospheric scale height (m), default 7000 m
    smooth_window : int, optional
        Smoothing window size, default 3
    apply_smoothing : bool, optional
        Whether to apply smoothing, default True
    
    Returns:
    --------
    z_mid : ndarray
        Heights at layer midpoints (m)
    z_int : ndarray
        Heights at layer interfaces (m)
    """
    # Compute pressure levels
    p_mid, p_int = compute_pressure_levels(hyam, hybm, hyai, hybi, PS, P0)
    
    # Convert to heights
    z_mid = pressure_to_height(p_mid, scale_height, PS)
    z_int = pressure_to_height(p_int, scale_height, PS)
    
    # Apply smoothing if requested
    if apply_smoothing and smooth_window > 1:
        z_mid = smooth_heights(z_mid, smooth_window)
        z_int = smooth_heights(z_int, smooth_window)
    
    return z_mid, z_int


def write_height_netcdf(output_file, z_mid, z_int, 
                        hyam=None, hybm=None, hyai=None, hybi=None,
                        PS=None, P0=None, scale_height=7000.0):
    """
    Write reference heights to NetCDF file.
    
    Parameters:
    -----------
    output_file : str
        Path to output NetCDF file
    z_mid : array_like
        Heights at layer midpoints (m)
    z_int : array_like
        Heights at layer interfaces (m)
    hyam, hybm, hyai, hybi : array_like, optional
        Original hybrid coefficients (for reference)
    PS : float, optional
        Surface pressure (Pa)
    P0 : float, optional
        Reference pressure (Pa)
    scale_height : float, optional
        Scale height used (m)
    """
    nlev = len(z_mid)
    nlevp = len(z_int)
    
    with nc.Dataset(output_file, 'w', format='NETCDF4') as ds:
        # Create dimensions
        ds.createDimension('lev', nlev)
        ds.createDimension('ilev', nlevp)
        
        # Create variables for heights
        var_zmid = ds.createVariable('z_mid', 'f8', ('lev',))
        var_zmid.long_name = 'Reference height at layer midpoints'
        var_zmid.units = 'm'
        var_zmid[:] = z_mid
        
        var_zint = ds.createVariable('z_int', 'f8', ('ilev',))
        var_zint.long_name = 'Reference height at layer interfaces'
        var_zint.units = 'm'
        var_zint[:] = z_int
        
        # Optionally include original hybrid coefficients
        if hyam is not None:
            var = ds.createVariable('hyam', 'f8', ('lev',))
            var.long_name = 'Hybrid A coefficient at layer midpoints'
            var[:] = hyam
        
        if hybm is not None:
            var = ds.createVariable('hybm', 'f8', ('lev',))
            var.long_name = 'Hybrid B coefficient at layer midpoints'
            var[:] = hybm
        
        if hyai is not None:
            var = ds.createVariable('hyai', 'f8', ('ilev',))
            var.long_name = 'Hybrid A coefficient at layer interfaces'
            var[:] = hyai
        
        if hybi is not None:
            var = ds.createVariable('hybi', 'f8', ('ilev',))
            var.long_name = 'Hybrid B coefficient at layer interfaces'
            var[:] = hybi
        
        # Global attributes
        ds.title = 'Reference heights from hybrid-sigma levels'
        ds.scale_height = f'{scale_height} m'
        ds.atmosphere_model = 'Isothermal'
        
        if PS is not None:
            ds.surface_pressure = f'{PS} Pa'
        if P0 is not None:
            ds.reference_pressure = f'{P0} Pa'
        
        ds.history = 'Converted from hybrid-sigma to height coordinates'


def read_sigma_file_and_convert(input_file, output_file, 
                                 ps_var='PS', p0_var='P0',
                                 scale_height=7000.0,
                                 smooth_window=3):
    """
    Read hybrid-sigma data from NetCDF and convert to heights.
    
    Parameters:
    -----------
    input_file : str
        Path to input NetCDF file with hybrid coefficients
    output_file : str
        Path to output NetCDF file
    ps_var : str, optional
        Name of surface pressure variable
    p0_var : str, optional
        Name of reference pressure variable
    scale_height : float, optional
        Atmospheric scale height (m)
    smooth_window : int, optional
        Smoothing window size
    """
    with nc.Dataset(input_file, 'r') as ds:
        hyam = ds.variables['hyam'][:]
        hybm = ds.variables['hybm'][:]
        hyai = ds.variables['hyai'][:]
        hybi = ds.variables['hybi'][:]
        
        # Get PS and P0
        if ps_var in ds.variables:
            PS = ds.variables[ps_var][:]
            if PS.ndim > 0:
                PS = PS[0]  # Take first value if array
        else:
            PS = 101325.0  # Default surface pressure
            print(f"Warning: {ps_var} not found, using default {PS} Pa")
        
        if p0_var in ds.variables:
            P0 = ds.variables[p0_var][:]
            if P0.ndim > 0:
                P0 = P0[0]
        else:
            P0 = 100000.0  # Default reference pressure
            print(f"Warning: {p0_var} not found, using default {P0} Pa")
    
    # Convert to heights
    z_mid, z_int = convert_sigma_to_height(
        hyam, hybm, hyai, hybi, PS, P0,
        scale_height=scale_height,
        smooth_window=smooth_window
    )
    
    # Write output
    write_height_netcdf(
        output_file, z_mid, z_int,
        hyam=hyam, hybm=hybm, hyai=hybi, hybi=hybi,
        PS=PS, P0=P0, scale_height=scale_height
    )
    
    print(f"Conversion complete. Output written to {output_file}")
    print(f"Height range: {z_int[0]:.2f} m to {z_int[-1]:.2f} m")
    print(f"Number of layers: {len(z_mid)}")
    
    return z_mid, z_int


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python sigma_to_height.py <input_file> <output_file>")
        print("\nExample with synthetic data:")
        
        # Create example data
        nlev = 30
        hyam = np.linspace(0, 90000, nlev)
        hybm = np.linspace(1.0, 0.0, nlev)
        hyai = np.linspace(0, 95000, nlev+1)
        hybi = np.linspace(1.0, 0.0, nlev+1)
        PS = 101325.0
        P0 = 100000.0
        
        z_mid, z_int = convert_sigma_to_height(hyam, hybm, hyai, hybi, PS, P0)
        
        print(f"\nExample conversion (nlev={nlev}):")
        print(f"Height at surface: {z_int[0]:.2f} m")
        print(f"Height at top: {z_int[-1]:.2f} m")
        print(f"First few interface heights: {z_int[:5]}")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        read_sigma_file_and_convert(input_file, output_file)