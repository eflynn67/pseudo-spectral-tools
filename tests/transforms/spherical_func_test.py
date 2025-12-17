import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser(f'~/pseudo-spectral-tools/src/pseudo-spectral-tools'))
import transforms

def plot_3d_slices_contour(f_xyz, cPnts, num_slices=3):
    """
    Plots cross-sectional (x, y) slices of the 3D array f_xyz using 
    plt.contourf (filled contours) and plt.contour (level lines).

    Args:
        f_xyz (np.ndarray): The 3D NumPy array of the function f(x, y, z).
        r_max (float): The maximum coordinate value (extent) of the grid.
        num_slices (int): The number of equally spaced slices to plot.
    """
    cPnts = np.arange(-cPnts[-1],cPnts[-1],0.05)
    N = f_xyz.shape[0]  # Grid size
    
    # 1. Define the 2D coordinate grid (X_2D, Y_2D) for plotting
    # These arrays are needed by contour/contourf
    X_2D, Y_2D = np.meshgrid(cPnts,cPnts, indexing='ij')

    # Calculate the indices for the slices
    slice_indices = np.linspace(0, N - 1, num_slices, dtype=int)
    z_coords = cPnts
    
    # Create the figure and subplots
    #fig, axes = plt.subplots(1, , figsize=(4.5 * num_slices, 4.5))
    
    if num_slices == 1:
        axes = [axes] # Ensure axes is iterable

    # Loop through the desired slice indices
    for i, z_index in enumerate(slice_indices):
        # 2. Extract the (x, y) slice at a fixed z_index
        slice_2d = f_xyz[:, :, z_index] 
        z_value = z_coords[z_index]

        # 3. Plot the filled contours (plt.contourf)
        # We specify 20 levels for a smooth color map
        contour_filled = plt.contourf(
            X_2D, Y_2D, slice_2d, levels=20, cmap='Spectral_r'
        )
        
        # 4. Plot the level lines (plt.contour) on top
        # We use a dark color (k=black) for the lines to distinguish them
        plt.contour(
            X_2D, Y_2D, slice_2d, levels=20, colors='k', linewidths=0.5
        )
        
        # 5. Set titles and labels
        #plt.set_title(f'Contour Plot at $z = {z_value:.2f}$', fontsize=12)
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        #plt.set_aspect('equal', adjustable='box') # Keep circles looking circular
        
        # 6. Add a colorbar for reference (using the filled contour object)
        plt.colorbar(contour_filled, orientation='vertical', fraction=0.046, pad=0.04)
        plt.show()


def test_func(r):
    return np.exp(-r)


cPnts = np.arange(0,10,.05)
fArr = test_func(cPnts)

plt.plot(cPnts,fArr)
plt.show()


f_xyz = transforms.radial_to_3d(fArr, cPnts, len(cPnts))

plot_3d_slices_contour(f_xyz, cPnts, num_slices=30)
