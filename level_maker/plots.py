import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import argparse


def main():
    """
    Parse command-line arguments and generate plots.
    """
    parser = argparse.ArgumentParser(description="""Given two files that contain vertical level information, 
    create a 3-panel plot showing pressures and δz (full and lower-troposphere).""")
    parser.add_argument('f1', type=str, help='path to input file 1')
    parser.add_argument('f2', type=str, help='path to input file 2')
    parser.add_argument('nam1', type=str, help='short name for f1')
    parser.add_argument('nam2', type=str, help='short name for f2')
    parser.add_argument('pout', type=str, help='path to output plot file')
    args = parser.parse_args()

    # load the grids to plot
    ds = {
        args.nam1: xr.open_dataset(args.f1),
        args.nam2: xr.open_dataset(args.f2)
    }
    lev = {c: ds[c]['lev'] for c in ds}

    # units check
    for x in lev:
        if lev[x].max() > 2000.:
            lev[x] = lev[x] / 100

    ilev = {c: ds[c]['ilev'] for c in ds}
    for x in ilev:
        if ilev[x].max() > 2000.:
            ilev[x] = ilev[x] / 100

    z = {c: get_approx_z(lev[c], H=7.0) for c in lev}
    dz = {c: np.gradient(z[c]) for c in z}

    make_plot(z, dz, lev, args.pout)


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


def make_plot(z, dz, lev, ofil):
    # consistent colors:
    palette = ["navy", "lightskyblue", "firebrick","tomato"]
    colors = {c:palette[i] for i,c in enumerate(dz)}
    #
    # Plot the original levels along with the constructed points
    #
    fig, ax = plt.subplots(figsize=(12, 3), ncols=3, constrained_layout=True)
    for x in lev:
        ax[0].plot(
            lev[x],
            np.arange(len(lev[x])),
            color=colors[x],
            linestyle="none",
            marker=".",
            label=x,
        )
        ax[1].plot(dz[x], lev[x], marker=".", color=colors[x], label=x)
        ax[2].plot(dz[x], lev[x], color=colors[x], marker=".", linestyle=':', label=x)

    ax[0].set_ylabel("LEV INDEX")
    ax[0].set_xlabel("NOMINAL PRESSURE")
    ax[0].invert_yaxis()
    ax[0].legend()
    #
    # PANEL 2 Plot dz vs lev for the new grids
    #
    ax[1].invert_yaxis()
    ax[1].legend()
    ax[1].set_xlabel("δz (km)")
    ax[1].set_ylabel("lev (hPa)")
    #
    # PANEL 3 Plot the dz vs lev just in the lower troposphere; compare with original grid
    #
    ax[2].invert_yaxis()
    ax[2].axvline(-0.2, linestyle=":", color="gray")
    ax[2].axhline(800, linestyle=":", color="gray")
    ax[2].set_ylim([1013, 500])
    ax[2].set_xlim([-1, 0.001])
    ax[2].legend()
    ax[2].set_xlabel("δz (km)")
    ax[2].set_ylabel("lev (hPa)")
    fig.savefig(f'{ofil}.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
    print("FINISHED")