import argparse
import json
import numpy as np

def make_levels(dps, purmax, regions, print_out=False):
    # infer model top from the first entry in the last tuple:
    ptop_model = regions[-1][0]
    # start with surface layer
    outint = [1000.0]   
    outdln = [dps]
    outint.append(np.exp(np.log(outint[0]) - outdln[0]))
    outdp = [outint[0] - outint[1]]
    # we now start building the levels using the region spec
    # hold current values of the increment and region max:
    region_current = 0
    dpinc_current = regions[region_current][1]
    dpmax_current = regions[region_current][2]
    while outint[-1] > ptop_model:
        outdln.append(min([outdln[-1] + dpinc_current, dpmax_current]))
        outint.append(np.exp(np.log(outint[-1]) - outdln[-1]))
        outdp.append(outint[-2] - outint[-1])
        # check if we need to move to next region
        if outint[-1] <= regions[region_current][0]:
            if region_current == len(regions)-1:
                print(f"Last region finished... break out of level building loop.")
                break
            else:
                region_current += 1
                dpinc_current = regions[region_current][1]
                dpmax_current = regions[region_current][2]
                print(f"Moving into region: {region_current} with increment {dpinc_current} and top {dpmax_current}")

    ktop = len(outint)-1  # subtract 1 b/c zero-based python indexing
    # convert lists to Numpy ndarray for easier calculations:
    # NOTE: Be cautious of views versus copies. 
    #       Throughout this code, we want assignment to produce COPIES!
    # NOTE: Caution #2 is that when we slice numpy arrays, it is exclusive on the upper limit:
    #       Example: 
    #       >>> arr = np.array([10., 20., 30., 40.])
    #       >>> arr[0:2]
    #           array([10., 20.])
    outint = np.array(outint)
    outdln = np.array(outdln)
    outdp = np.array(outdp)
    # midpoint values
    outmid = 0.5 * (outint[:-1] + outint[1:])
    outz = 8. * np.log(1000. / outmid)
    # informational output:
    # interface outint[k]
    #                    outmid[k] outdp[k]
    # interface outint[k+1]
    print("int, mid, dp, dlnp, z")
    for k in np.arange(ktop, -1, -1):
        if k < len(outmid):
            print(f"        {outmid[k]:11.5f} , {outdp[k]:11.5f} , {outdln[k]:11.5f}, {outz[k]:5.3f}")
        print(f"[k={k}] {ktop-k+1} {outint[k]}")
    # INFO: kmax is the number of midpoint levels
    #       ktop is the number of interface levels
    #       kpur is the lowest (in height) interface level that is pure pressure
    #       xmin is the ai+bi value at the kpur level
    #       xmax is the ai+bi value at the penultimate interface
    kmax = ktop - 1 
    ampbm = outmid[::-1].copy()  # am plus bm, top to bottom, same as p levels
    am = ampbm.copy()            # assume all levels are pure pressure initially
    bm = np.zeros_like(am)
    aipbi = outint[::-1].copy()  # set interfaces in same way as midpoints
    ai = aipbi.copy()
    bi = np.zeros_like(ai)
    # find bottom most pure pressure interface
    candidates = np.argwhere(aipbi > purmax)
    if candidates.size == 0:
        kpur = -1  # ?? If all pressures are below where it's pure pressure
    else:
        kpur = np.min(candidates) - 1  # last interface in pure pressure region
    xmin = aipbi[kpur]
    xmax = aipbi[kmax]
    # interface values // set values from next past kpur until last TWO interfaces
    # print(f"Set interface values at levels k = {kpur+1} to {kmax-1}")
    ai[kpur+1:kmax] = xmin * (1.0 - (aipbi[kpur+1:kmax]-xmin) / (xmax - xmin))
    bi[kpur+1:kmax] = aipbi[kpur+1:kmax] - ai[kpur+1:kmax]
    # last two interfaces (k = kmax and k = ktop) get set explicitly:
    ai[kmax:] = 0.0
    bi[kmax:] = aipbi[kmax:].copy()
    # midpoint values ; from the level below kpur all the way to the surface
    # note: we could go to kpur+1:kmax to exclude last one, but it just gets overwritten anyway
    # print(f"Set midpoint values at levels k = {kpur+1} to {kmax}")
    am[kpur+1:] = xmin * (1.0 - (ampbm[kpur+1:] - xmin) / (xmax - xmin))
    bm[kpur+1:] = ampbm[kpur+1:] - am[kpur+1:]
    am[kmax] = 0.0
    bm[kmax] = ampbm[-1].copy()
    aisig = ai/1000.0
    bisig = bi/1000.0
    amsig = am/1000.0
    bmsig = bm/1000.0

    if print_out:
        print("lev , a , b, am+bm")
        for i, a in enumerate(am):
            print(f"{i:02d} , {a:1.5E} , {bm[i]:1.5E} , {ampbm[i]:7.4f}")
        print("ilev , aisig , bisig, ai+bi")
        for i, ais in enumerate(aisig):
            print(f"{i:02d} , {ais:1.5E} , {bisig[i]:1.5E} , {aipbi[i]:7.4f}")
    return amsig, bmsig, aisig, bisig, outmid, outint


def send_to_output(lev, am, bm, ilev, ai, bi, output_file):
    try:
        import xarray as xr
    except:
        raise ImportError("Sorry, looks like you don't have xarray available.")
    lev_xr = xr.DataArray(lev, dims=["lev"], coords={"lev":lev}, name='lev')
    lev_xr.attrs['long_name'] = "hybrid level at midpoints (1000*(A+B))"
    lev_xr.attrs['units'] = 'hPa'
    lev_xr.attrs['positive'] = 'down'
    lev_xr.attrs['standard_name'] = "atmosphere_hybrid_sigma_pressure_coordinate"
    lev_xr.attrs['formula_terms'] = "a: hyam b: hybm p0: P0 ps: PS"
    lev_xr.attrs['_FillValue'] = 9.9692099683868690e+36

    ilev_xr = xr.DataArray(ilev, dims=["ilev"], coords={"ilev":ilev}, name='ilev')
    ilev_xr.attrs['long_name'] = "hybrid level at interfaces (1000*(A+B))"
    ilev_xr.attrs['units'] = 'hPa'
    ilev_xr.attrs['positive'] = 'down'
    ilev_xr.attrs['standard_name'] = "atmosphere_hybrid_sigma_pressure_coordinate"
    ilev_xr.attrs['formula_terms'] = "a: hyai b: hybi p0: P0 ps: PS"
    ilev_xr.attrs['_FillValue'] = 9.9692099683868690e+36

    hyam = xr.DataArray(am, dims=["lev"], coords={"lev":lev_xr}, name="hyam")
    hyam.attrs['long_name'] = "hybrid A coefficient at layer midpoints"
    hyam.attrs['_FillValue'] = 9.9692099683868690e+36

    hybm = xr.DataArray(bm, dims=["lev"], coords={"lev":lev_xr}, name="hybm")
    hybm.attrs['long_name'] = "hybrid B coefficient at layer midpoints"
    hybm.attrs['_FillValue'] = 9.9692099683868690e+36
    
    hyai = xr.DataArray(ai, dims=["ilev"], coords={"ilev":ilev_xr}, name="hyai")
    hyai.attrs['long_name'] = "hybrid A coefficient at layer interfaces"
    hyai.attrs['_FillValue'] = 9.9692099683868690e+36

    hybi = xr.DataArray(bi, dims=["ilev"], coords={"ilev":ilev_xr}, name="hybi")
    hybi.attrs['long_name'] = "hybrid B coefficient at layer interfaces"
    hybi.attrs['_FillValue'] = 9.9692099683868690e+36

    ds = xr.merge([hyam, hybm, hyai, hybi])
    ds.to_netcdf(output_file)

if __name__ == "__main__":
    """
    NCAR/CGD/AMP Generator for atmosphere hybrid-sigma coefficients.

    Parameters
    ----------
    infile : A string specifying the path to a JSON file.

            The JSON file is expected to contain the following required input:
            - dps : surface layer thickness in dlnp
            - purmax : maximum (bottom) pure pressure interface
            - regions: An array of 3-element arrays, specified as 
            (top-pressure(mb), increment (dlnp), max thickness (dlnp))
            The array regions can have arbitary number of regions, but must be
            ordered bottom (surface) to top (of atmosphere).
            - [optional] outfile : a string specifying the output netCDF file
                         The outfile can also be specified as a second command-line argument,
                         with that one taking precedence if both are present.
    Returns
    -------
            Returns hybrid-sigma coefficients: am, bm, ai, bi
            where m means mid-point and i means interface.

            If either a second command line argument or an outfile parameter in
            the JSON file are specified, the coefficients are written to a netCDF
            file. That requires xarray be available.
            Otherwise, the output is directed to standard out when run as command line utility.

            If the `make_levels` function is imported and used directly,
            it returns a tuple of (am, bm, ai, bi, lev, ilev).

    
    Reference
    ---------
    Uses the algorithm described in:
    David L. Williamson, Jerry G. Olson, and Byron A. Boville. 
        A comparison of semi-Lagrangian and Eulerian tropical climate simulations. 
        Monthly Weather Review, 126(4):1001–1012, 1998. 
        doi: 10.1175/1520-0493(1998)126<1001:ACOSLA>2.0.CO;2



    """
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="JSON file containing input specficiation.", type=str)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args()
    #open the file
    with open(args.infile) as f:
        data = json.load(f)
    # DEFAULT INPUT TO REPRODUCE 26-LEVEL GRID:
    # dps_in = .015 # surface layer thickness in dlnp
    # purmax_in = 90.0 # maximum (bottom) pure pressure interface
    # regions are specified by tuple: (top-pressure(mb), increment (dlnp), max thickness (dlnp))
    # supplied within in an interable, e.g., list:
    # example: [(bl), (tropo), (strato), (meso)]
    #regions_in = [(970.0, 0.015, 1.0),
    #(90.0, 0.0267, .1625),
    #(40.0, 0.08, 1.0),
    #(3.0, 0.1, 1.0)]
    assert 'dps' in data
    assert 'purmax' in data
    assert 'regions' in data
    if args.outfile is not None:
        savefile = True
        output_file = args.outfile    
    am, bm, ai, bi, lev, ilev = make_levels(data['dps'], data['purmax'], data['regions'])
    if savefile:
        send_to_output(lev, am, bm, ilev, ai, bi, output_file)