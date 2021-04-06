import argparse
import json
import numpy as np
import xarray as xr
from makelev import  get_coefs, send_to_output
import logging
logging.basicConfig(level=logging.DEBUG)

def get_approx_z(p, H=8.0):
    """Return z in kilometers from p in hectopascal using scale height H."""
    return H * np.log(1000. / p)


def get_target_pressures(z, H=8000):
    """Return p in hPa from z in meters.
       Parameter H is scale height in meters.
    """
    if isinstance(z, list):
        z = np.array(z)
    return 1000. * np.exp((-1*z)/H)


def eriksson(a, b, beta=1, npoints=25):
    """Apply stretching transformation due to Eriksson (1982),
       clustering of points is put at end of interval [a, b]:
       a : starting value in physical space
       b : ending value in physical space
       beta : parameter controlling how quickly the transition happens
       npoints : the number of points to use
       
       RETURNS
       xe : the stretched grid points in physical space (numpy ndarray)
    """
    logging.debug(f"eriksson args: {(a, b, beta, npoints)}")
    eta = np.linspace(0, 1, npoints) # transformed coordinate -- equal spacing in computation space
    # cluster near beginning
    # xescale = (np.exp(beta*eta) - 1) / (np.exp(beta) - 1)
    # xe = (1 - xescale)*(a-b) + b
    # cluster near end
    xescale = (np.exp(beta) - np.exp(beta*(1 - eta)))/(np.exp(beta) - 1)
    xe = (1 - xescale)*(a-b) + b
    return xe


def tanh_stretch(a, b, beta=1, npoints=25):
    """Apply hyperbolic tangent transformation
       a : starting value in physical space
       b : ending value in physical space
       beta : parameter to control how quickly transition happens (tanh(beta*x))
       npoints : the number of points to use
       
       RETURNS
       xe : the stretched grid points in physical space (numpy ndarray)
       
       NOTES
       The computational space is [0, π]. The tanh values do not strictly go
       from 0 to 1, so we apply a second scaling to ensure that the end points
       are at a and b in physical space.
    """
    eta = np.linspace(0, np.pi, npoints)
    xescale = np.tanh(beta*eta)
    # but xescale won't actually go to 1, so have to rescale the scale
    xescale = (xescale - xescale.min())/(xescale.max() - xescale.min())
    xe = (1 - xescale)*(a - b) + b
    return xe


def no_stretch(a, b, npoints=25):
    """Just put equally spaced points from a to b."""
    return np.linspace(a, b, npoints)



def modify_lowertrop(inlev_Pa, pPa_bottom, cut_Pa, tot_points, method, **kwargs):
    """
    inlev_Pa : input levels in Pa
    p_bottom : the pressure at the bottom of the grid
    cut_Pa   : value in Pa where to start modifying (levels p < cut_Pa are unchanged)
    tot_points : integer value defining the total number of output levels
    method : provide the function (actual function, not string) to use for grid generation
             (current supported methods: eriksson, tanh_stretch)
    *args : contains the parameters that will go into method
    
    RETURNS
    outlev : final grid of length tot_points, units Pa, top-to-bottom ordering
    
    NOTES
    I use `get_target_pressures` to convert target bottom height to p_bottom
    """
    dp = np.gradient(inlev_Pa)
    lev_trop = inlev_Pa[inlev_Pa <= cut_Pa]
    dp_trop = dp[inlev_Pa <= cut_Pa]
#     print(f"Actual cut point at {lev_trop.max()}; dp = {dp_trop[-1]}")
    M = tot_points - len(lev_trop)  # number of points to use in grid generation
    M += 1 # get one back because we won't use the duplicate at lev_trop[-1]
    # Note: we are assuming that any method will follow simple API
    upper_value = lev_trop.max()
    lower_value = pPa_bottom
    # expecting kwargs to contain beta parameter
    mpts = method(upper_value, lower_value, npoints=M, **kwargs)
    return np.append(lev_trop[:-1], mpts) # outlev, still should be top-to-bottom

#
# implement 1-d mesh generation from Qarteroni book
#  --> not currently used, but I'm leaving in case useful later. 
#
def mesh_1d(a, b, func, *args):
    # note: func must be strictly positive (including being non-zero at a)
    coord = [a]
    while coord[-1] < b:
        x = coord[-1]
        xnew = x + func(x, *args)
        coord.append(xnew)
    
    if (coord[-1] - b) > (b - coord[-2]):
        coord = coord[:-2]
    coord_old = coord.copy()
    print(f"first pass: {coord_old}")
    kappa = (b - coord[-2]) / (coord[-1] - coord[-2])
    coord = [a]
    for i in coord_old[:-1]:
        coord.append(i + kappa*func(i, *args))
    return coord


if __name__ == "__main__":
    """Provided JSON file with an input file path, use `lev` in that file
       and splice a new set of levels on to the bottom, as specified by
       the rest of the JSON entries:
       infile : the input file, which has a `lev` variable
       bottom_z OR bottom_p : either specify bottom of grid in z (meters) or p (hPa) (p is given precedence)
       cut_pressure : where in `lev` to start changing levels, specify in Pa
       out_grid_size : the total number of levels in the final output grid. (optional to use add_n_levels instead)
       add_n_levels : use `len(lev)` from infile and add this number of levels for final grid (out_grid_size take precedence, one of them is needed) 
       method: currently specify either 'eriksson' or 'tanh'
       method_params: provide a dictionary to pass to method, currently just ["beta":VALUE]
       scale_height: assumed scale height in conversions between p & z, defaults to 7 km.
       pure_p: pressure where levels become pure pressure, defaults to 90 hPa
       outfile: output file path, secondary to command line option `-o`
    """ 
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="JSON file containing input specficiation.", type=str)
    parser.add_argument("-o", "--outfile", type=str)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    #open the file
    with open(args.infile) as f:
        data = json.load(f)

    # get input lev (and ilev) from file:
    ds_input = xr.open_dataset(data['infile'])
    lev_input = ds_input['lev'].values
    logging.info(f"Input levels: number of levels {len(lev_input)}, min: {lev_input.min()}, max: {lev_input.max()}")

    # if 'ilev' in ds_input:
    #     ilev_input = ds_input['ilev']
    #     has_ilev = True

    # get bottom value (of lowest mid-point)
    if 'scale_height' in data:
        h = data['scale_height']
    else:
        h = 7.0

    if 'bottom_p' in data:
        p_bottom = data['bottom_p']
    elif 'bottom_z' in data:    
        p_bottom = 100.0 * get_target_pressures(data['bottom_z'], H=h*1000)
        logging.debug(f"p_bottom derived as {p_bottom}")
    
    # specify the reference pressure:
    if 'p_reference' in data:
        p0 = data['p_reference']
    else:
        p0 = 1000.

    # specify the function for method:
    if data['method'].casefold() == 'eriksson':
        mth = eriksson
    elif data['method'].casefold() == 'tanh':
        mth = tanh_stretch
    elif data['method'].casefold() == 'linear':
        mth = no_stretch
    else:
        raise IOError("Method needs to be one of [eriksson, tanh, linear]")
    if 'method_params' in data:
        opt = data['method_params']
    else:
        opt = dict()

    if 'pure_p' in data:
        purp = data['pure_p']
    else:
        purp = 90.0  # in hPa

    if args.outfile is not None:
        savefile = True
        output_file = args.outfile 
    elif 'outfile' in data:
        savefile = True
        output_file = data['outfile']
    else:
        savefile = False

    if 'out_grid_size' in data:
        nlevels = data['out_grid_size']
    elif 'add_n_levels' in data:
        nlevels = len(lev_input) + data['add_n_levels']
        logging.info(f"The number of levels is {nlevels} (type: {type(nlevels)}.")
    else:
        raise IOError("Need to specify either out_grid_size or add_n_levels to determine final grid size.")

    logging.info(f"Inputs to modify_lowertrop: lev (size {lev_input.size}), p_bottom: {p_bottom}, cut_pressure: {data['cut_pressure']}, n: {nlevels}, method: {mth.__name__}, opt: {opt}")
    logging.info(f"It is useful to know which direction lev goes: lev[0] = {lev_input[0]}, lev[-1] = {lev_input[-1]}.")
    lev = modify_lowertrop(lev_input, p_bottom, data['cut_pressure'], nlevels, mth, **opt)
    print(lev)
    lev *= 0.01  # to hPa
    #
    # derive the interface levels
    #
    # ilev = [1000.]
    rlev = lev[::-1]  # need to go bottom to top ("reverse lev")
    print(f"rlev = {rlev}")
    # for k, p in enumerate(rlev):
    #     ilev.append(2*rlev[k] - ilev[k])
    ilev = [-999999]  # initialize with a bogus lowest interface
    for k, f in enumerate(rlev[:-1]):
        print(f"in loop - k = {k}, f = {f}")
        ilev.append(0.5*(f+rlev[k+1]))

    # end points of ilev:
    # top interface: 
    # if ilev is provided in input, use first value
    if 'ilev' in ds_input:
        logging.info(f"Use the highest interface level from the input data: {ds_input['ilev'][0]*0.01}")
        ilev.append(0.01*ds_input['ilev'][0])
    else:
        # extrapolate ... take highest midpoint, rlev[-1]
        #                 and put the next interface same distance as the lower interface from there
        logging.info(f"Extrapolate to highest interface {rlev[-1]} - ({ilev[-1]} - {rlev[-1]})")
        ilev.append(rlev[-1] - (ilev[-1] - rlev[-1]))

    ilev = np.array(ilev[::-1])
    # do not allow interface levels to go beyond reference pressure:
    ilev = np.where(ilev > p0, p0, ilev)
    # actually, lets assume that the last ilev should be equivalent to p0
    ilev[-1] = p0
    logging.info(f"Preliminary values of ilev: {ilev}")
    #
    # get the coefficients -- note that we need to provide bottom-to-top ordering
    #
    am, bm, ai, bi = get_coefs(ilev[::-1], lev[::-1], purp)
    logging.debug(f"The ai values: {ai}")
    #
    # save to a simple netCDF
    #
    if savefile:
        send_to_output(lev, am, bm, ilev, ai, bi, output_file)