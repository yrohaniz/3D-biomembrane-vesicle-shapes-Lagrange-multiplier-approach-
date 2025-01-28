import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset


class PhaseFieldNet(torch.nn.Module):
    """
    A neural network for estimating the phase field associated
    with a specific vesicle
    """

    def __init__(self):
        super(PhaseFieldNet, self).__init__()

        self.n_features = 3
        self.n_out = 1
        self.n_h_1 = 20
        self.n_h_2 = 10

        self.hidden0 = nn.Sequential(
            nn.Linear(self.n_features, self.n_h_1),
            nn.Sigmoid()
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(self.n_h_1, self.n_h_2),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(self.n_h_2, self.n_out),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.hidden0(z)
        z = self.hidden1(z)
        z = self.out(z)
        return z


def integrate_func(y, pr, intgrtd_pr):
    """
    This function calculates the MC integral of functions of the
    phase field estimated by the neural network (y). The weight here
    is given by parameter 'p'. The vol (volume) is set to the value
    determined by the spatial integral of the weight 'p' ,i.e. P(r)
    """
    integral = torch.sum(y / pr)
    integral = integral * (intgrtd_pr / np.float64(y.size()[0]))
    return integral


def auto_diff(y, x):
    """
    This function uses the autograd method of Pytorch for spatial derivative
    (gradient) calculations. The first parameter is the output of the neural net
    (estimated function) and the second parameter is the variable with respect
    to which the derivative is calculated (here it is the position variable)
    """

    dy_dx = torch.autograd.grad(outputs=y, inputs=x,
                                grad_outputs=torch.ones_like(y),
                                create_graph=True)[0]
    return dy_dx


def phi_dphi_d2phi(r, first=False, second=False):
    """
    This function returns the value of the phase field as estimated by
    the neural net as well as the modulus squared of the first spatial derivative
    and the spatial Laplacian of the phase field. The first parameter is the 3D
    spatial point, and the second and third boolean parameters determine
    whether the derivative calculations are required or not.
    """

    r.requires_grad_(True)
    phi = n_net(r)
    dphi_dr = torch.empty(0)
    mod_dphidr_sq = torch.empty(0)
    d2phi_dr2 = torch.empty(0)

    if first:
        dphi_dr = auto_diff(phi, r)
        mod_dphidr_sq = torch.sum(torch.square(dphi_dr), dim=1, keepdim=True)

    if second:
        first_col = auto_diff(dphi_dr[:, 0], r)[:, 0].view(phi.size())
        second_col = auto_diff(dphi_dr[:, 1], r)[:, 1].view(phi.size())
        third_col = auto_diff(dphi_dr[:, 2], r)[:, 2].view(phi.size())
        d2phi_dr2 = torch.cat([first_col, second_col, third_col], dim=1)
        d2phi_dr2 = torch.sum(d2phi_dr2, dim=1, keepdim=True)

    # After carrying out the spatial differentiations remove the requires_grad flag
    r.requires_grad_(False)

    return phi, mod_dphidr_sq, d2phi_dr2


def bending_energy_density(phi, del2phi, epsilon, c_0):
    """
    This function returns the value of the bending energy
    density for the given values of the phase filed and phase field
    Laplacian at different spatial points, and also in terms of
    epsilon (membrane thickness) and c_0 (spontaneous curvature)
    """

    return (3.0 / (8.0 * torch.sqrt(torch.tensor(2.0)) * epsilon)) * (
            (epsilon * del2phi) + ((phi / epsilon) + c_0 * torch.sqrt(torch.tensor(2.0)))
            * (1.0 - phi ** 2.0)) ** 2.0


def surface_density(phi, mod_delphi_sq, epsilon):
    """
    This function returns the value of the surface density for
    the given values of the phase filed and the magnitude of the
    phase field gradient at different spatial points, and also in
    terms of the parameter epsilon (membrane thickness)
    """

    return (3.0 / (4.0 * torch.sqrt(torch.tensor(2.0)))) * (
            (epsilon * mod_delphi_sq) + ((1.0 / (2.0 * epsilon)) * (1.0 - phi ** 2.0) ** 2.0))


def volume_density(phi):
    """
    This function returns the value of the volume density for
    the given values of the phase filed at different spatial points
    """

    return 0.5 * (1.0 + phi)


def train_phasefield_net(optimizer, r, lmbdas):
    """
    This function contains some Pytorch library subroutines
    that perform autodiff and use a specified optimizer (e.g. SGD, Adam, etc.)
    to execute back propagation through the neural net and calculate the
    gradients wrt the weights, biases, input data, etc. It also updates the
    weights and biases.
    """

    # Reset gradients
    optimizer.zero_grad()

    # Compute the phase field and its first and second order spatial derivatives
    phase_field, mod_sq_first_derv, second_derv = phi_dphi_d2phi(r, True, True)

    # Compute the bending energy density using the value of the phase field at different positions
    bending_e_dns = bending_energy_density(phase_field, second_derv, epsilon=eps, c_0=c_zero)

    # Compute the surface density as a function of the calculated phase field for different positions
    surface_dns = surface_density(phase_field, mod_sq_first_derv, epsilon=eps)

    # Compute the volume density as a function of the calculated phase field for different positions
    volume_dns = volume_density(phase_field)

    p_r = 1.0
    integrated_p_r = (r_max - r_min) ** dim  # Volume of the 3D domain for MC weight p_r = 1.0
    # Integrate the bending energy density using the Monte Carlo method
    bending_energy = integrate_func(bending_e_dns, p_r, integrated_p_r)

    # Integrate the surface density using the Monte Carlo method
    surface_integral = integrate_func(surface_dns, p_r, integrated_p_r)

    # Using the target value for the surface area of the vesicle calculate the loss for the surface term
    surf_loss = (surface_integral - torch.tensor(A_0))

    # Integrate the volume density using the Monte Carlo method
    volume_integral = integrate_func(volume_dns, p_r, integrated_p_r)

    # Using the target value for the volume of the vesicle calculate the loss for the volume term
    vol_loss = (volume_integral - torch.tensor(V_0))

    v_star = volume_integral
    a_star = surface_integral
    w_star = bending_energy

    # Calculate the center of mass term for minimization towards the coordinate origin
    mass_vol_dns_x = torch.mul(volume_dns, r[:, 0].view(volume_dns.size()))
    center_mass_x = integrate_func(mass_vol_dns_x, p_r, integrated_p_r)
    mass_vol_dns_y = torch.mul(volume_dns, r[:, 1].view(volume_dns.size()))
    center_mass_y = integrate_func(mass_vol_dns_y, p_r, integrated_p_r)
    mass_vol_dns_z = torch.mul(volume_dns, r[:, 2].view(volume_dns.size()))
    center_mass_z = integrate_func(mass_vol_dns_z, p_r, integrated_p_r)
    center_mass_2 = center_mass_x ** 2.0 + center_mass_y ** 2.0 + center_mass_z ** 2.0

    # Initialize the Lagrange multipliers
    lmbda_a = lmbdas[0, 0]  # surface tension
    lmbda_v = lmbdas[0, 1]  # pressure
    lmbda_cm = lmbdas[0, 2]  # force/length

    # Calculate cost. The constraints on surface area, volume and center of mass are imposed by Lagrange multipliers
    cost_val = bending_energy + lmbda_a * surf_loss + lmbda_v * vol_loss + lmbda_cm * center_mass_2
    # backpropagate
    cost_val.backward()

    # Update weights with gradients
    optimizer.step()

    # Update the Lagrange multipliers (steepest ascent)
    lmbda_a += eta * surf_loss.detach()
    lmbda_v += eta * vol_loss.detach()
    lmbda_cm += eta * center_mass_2.detach()

    return cost_val, phase_field, mod_sq_first_derv, a_star, v_star, w_star, lmbda_a, lmbda_v, lmbda_cm


def fixed_pt_training(num_epochs, pts):
    """This is the training function for the case where we have an
    evenly spaced grid of spatial points in the 3D space. Inputs are
    the number of epochs of optimization and the generated tensor of
    3D points"""

    # Train the phase field using evenly spaced spatial data
    training_cost_list = []
    volume = []
    area = []
    energy = []
    reduced_vol = []
    epoch: int = 0
    # Initialize the Lagrange multipliers
    lagrng_multips = torch.zeros([1, 3], device=device)
    lagrng_multips[0, 0] = -1.0
    lagrng_multips[0, 1] = 2.0
    lagrng_multips[0, 2] = 10.0
    for epoch in range(num_epochs):
        pts = pts.to(device)
        permute = torch.randperm(pts.size()[0], device=device)
        pts = pts[permute]
        training_cost, _, _, vesicle_surf, vesicle_vol, bend_eng, l_a, l_v, l_cm \
            = train_phasefield_net(func_optimizer, pts, lagrng_multips)
        lagrng_multips[0, 0] = l_a
        lagrng_multips[0, 1] = l_v
        lagrng_multips[0, 2] = l_cm

        training_cost_list.append(training_cost.cpu().detach().numpy())
        volume.append(vesicle_vol.cpu().detach().numpy())
        area.append(vesicle_surf.cpu().detach().numpy())
        energy.append((bend_eng.cpu().detach().numpy()) / (4.0 * np.pi))
        reduced_vol.append(vesicle_vol.cpu().detach().numpy()
                           / ((4.0 * np.pi / 3.0) *
                              ((vesicle_surf.cpu().detach().numpy() / (4.0 * np.pi)) ** 1.5)))

        print("Epoch %s training complete" % epoch)
        print("Training cost: %s " % training_cost_list[-1])
        print("area= %s" % area[-1])
        print("volume= %s" % volume[-1])
        print("reduced_volume= %s" % reduced_vol[-1])
        print("bending_energy= %s" % energy[-1])
        print(lagrng_multips)

    plot_training_cost(plot_dir, training_cost_list, epoch)
    plot_integrals(plot_dir, volume, epoch, 'volume')
    plot_integrals(plot_dir, area, epoch, 'area')
    plot_integrals(plot_dir, energy, epoch, 'bending energy')
    plot_integrals(plot_dir, reduced_vol, epoch, 'reduced volume')

    # Save the primary model trained by evenly-spaced spatial points using GPU
    torch.save(n_net.state_dict(), "{}/prime_PhaseField_{}Epoch_gpu.pt".format(saved_model_dir, epoch))

    # Save the primary model trained by evenly-spaced spatial points using CPU
    n_net.to(torch.device('cpu'))
    torch.save(n_net.state_dict(), "{}/prime_PhaseField_{}Epoch_cpu.pt".format(saved_model_dir, epoch))


def plot_training_cost(directory, tr_cost, n_epochs, training_cost_xmin=0):
    """
    This function plots the cost(loss) value against the corresponding
    epoch number
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, n_epochs),
            tr_cost[training_cost_xmin:n_epochs],
            color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, n_epochs])
    ax.grid(True)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.savefig('{}/cost.png'.format(directory), bbox_inches='tight')
    plt.close()


def plot_integrals(directory, integ_val, n_iters, label, integ_val_xmin=0):
    """
    This function plots the value of the volume or area integrals
    at different iterations of the importance sampling
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(integ_val_xmin, n_iters),
            integ_val[integ_val_xmin:n_iters],
            color='#2A6EA6')
    ax.set_xlim([integ_val_xmin, n_iters])
    ax.grid(True)
    ax.set_ylabel('{}_integral'.format(label))
    ax.set_xlabel('epoch')
    ax.set_title('{}_integral vs. epochs'.format(label))
    plt.savefig('{}/{}_integral_sampled_pts.png'.format(directory, label), bbox_inches='tight')
    plt.close()


def plot_phase_field(directory, y_data, x_data, label):
    """
    This function plots the phase field vs. one of the
    spatial coordinates e.g. phi(r) vs. x, phi(r) vs. y
    and phi(r) vs. z
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_data, y_data, 'bo', markersize=3)  # ,color='#2A6EA6'
    # ax.plot(x_data, auxil.numpy()[:, 14, 14], 'ro', markersize=3)
    ax.set_xticks(np.arange(-5.0, 5.1, 1))
    ax.set_yticks(np.arange(-1.2, 1.3, 0.2))
    ax.grid(True)
    ax.set_xlabel('{}'.format(label))
    ax.set_ylabel('phi')
    ax.set_title('phase field along {}'.format(label))
    plt.savefig('{}/{} phase field.png'.format(directory, label), bbox_inches='tight')
    plt.close()


def color_map_2d(directory, grid, axis1, axis2, field_vals):
    c = plt.pcolormesh(grid, grid, field_vals, cmap='Greens', shading='auto')
    plt.colorbar(c)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.savefig("{}/{}{}_colormap.png".format(directory, axis1, axis2))
    plt.close()


def evenly_spaced_r():
    """
    This function returns a tensor of size (number of points, number of dimensions).
    It basically produces evenly spaced points in 3 dimensions. For instance, if one
    wants to produce 4x4x4=64 points between coordinates -2 and 2, one can use
    a grid size (mesh size) of 4 along each axis and get
    [[-2.0000, -2.0000, -2.0000],
     [-2.0000, -2.0000, -0.6667],
     [-2.0000, -2.0000,  0.6667],
     [-2.0000, -2.0000,  2.0000],
     [-2.0000, -0.6667, -2.0000],
     [-2.0000, -0.6667, -0.6667],
     [-2.0000, -0.6667,  0.6667],
     [-2.0000, -0.6667,  2.0000],
     ...
     [ 2.0000,  0.6667, -2.0000],
     [ 2.0000,  0.6667, -0.6667],
     [ 2.0000,  0.6667,  0.6667],
     [ 2.0000,  0.6667,  2.0000],
     [ 2.0000,  2.0000, -2.0000],
     [ 2.0000,  2.0000, -0.6667],
     [ 2.0000,  2.0000,  0.6667],
     [ 2.0000,  2.0000,  2.0000]]
    """
    # Generate evenly spaced numbers along each spatial axis
    uni_spaced_nums = torch.linspace(r_min, r_max, mesh_size)
    # Generate spatial data based on the grid size for each coordinate (x, y, z)
    r_dt = torch.zeros(num_spatial_pts, dim)  # Initialize the spatial data
    # Loop through the dimensions (here 3)
    for r in range(dim):
        if r == 0:  # Condition for the first dimension (say x)
            for s in range(mesh_size):  # Fill the x coords up to mesh size squared with the current value
                r_dt[s * mesh_size ** 2:(s + 1) * mesh_size ** 2, r] = uni_spaced_nums[s]
        elif r == 1:  # Condition for the second dimension (say y)
            for s in range(mesh_size):  # Fill the y coords with the current value up to mesh size
                for t in range(0, num_spatial_pts, mesh_size ** 2):
                    r_dt[t + s * mesh_size:t + s * mesh_size + mesh_size, r] = uni_spaced_nums[s]
        elif r == 2:  # Condition for the third dimension (say z)
            for t in range(0, num_spatial_pts, mesh_size):
                for s in range(mesh_size):  # Fill the z coords with the current value and keep cycling
                    r_dt[t + s][r] = uni_spaced_nums[s]
    return r_dt


def evenly_spaced_integration():
    """This function can be used to check the volume integrals of bending energy,
    volume and surface densities for the given parameters of a trained model
    """

    r = evenly_spaced_r()
    test_phi, test_dphi, test_d2phi = phi_dphi_d2phi(r, True, True)
    v_dns = volume_density(test_phi)
    a_dns = surface_density(test_phi, test_dphi, eps)
    w_dns = bending_energy_density(test_phi, test_d2phi, eps, c_zero)
    v_intg = integrate_func(v_dns, 1.0, (r_max - r_min) ** dim)
    a_intg = integrate_func(a_dns, 1.0, (r_max - r_min) ** dim)
    w_intg = integrate_func(w_dns, 1.0, (r_max - r_min) ** dim)
    reduced_vol = v_intg / (torch.tensor(4.0 * np.pi / 3.0) * ((a_intg / torch.tensor(4.0 * np.pi)) ** 1.5))
    print('volume = %s' % v_intg.detach().numpy())
    print('area = %s' % a_intg.detach().numpy())
    print('bending_en = %s' % w_intg.detach().numpy())
    print('reduced_vol = %s' % reduced_vol.detach().numpy())
    return 0


# Make a Dataset out of the sampled spatial data
class NumbersDataset(Dataset):
    def __init__(self):
        self.samples = spatial_pts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def init_weights(wb):
    if isinstance(wb, nn.Linear):
        torch.nn.init.kaiming_normal_(wb.weight, mode='fan_in')
        if wb.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(wb.weight)
            std = 1.0 / np.sqrt(fan_in)
            torch.nn.init.normal_(wb.bias, 0.0, std)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    param_arr_range = 19
    mesh_size: int = 184
    dim: int = 3
    num_spatial_pts: int = mesh_size ** dim
    area_constraints = [(1.0 + k / 20) * 4.0 * np.pi for k in range(param_arr_range)]
    volume_constraints = [(1.0 - k / 20) * 4.0 * np.pi / 3.0 for k in range(param_arr_range)]
    epsilon_list = [(1.0 - k / 20) * 0.25 for k in range(param_arr_range)]

    # Determine which type of training toggled and what types of plots produced
    load_previous_model = True
    pts_fixed_training_mode = True
    pts_MC_training_mode = False
    test_mode = True
    one_D_plots = True
    color_map_plot = True

    # Read in the value of the environment variable associated with the SLURM_ARRAY_TASK_ID
    index = int(sys.argv[1])  # Use this number to pick the same index in area or volume constraints or epsilon
    # Create directories to save the plots and NN model parameters
    if not os.path.exists('plots_' + str(index)):
        os.mkdir('plots_' + str(index))
    if not os.path.exists('saved_model_' + str(index)):
        os.mkdir('saved_model_' + str(index))

    # Save the directory names so that the plotting functions can use them
    plot_dir = 'plots_' + str(index)
    saved_model_dir = 'saved_model_' + str(index)

    # Print the values of the picked constraints for the area or volume or the value of epsilon
    print('Final A_0 is = %s' % area_constraints[0])
    print('Final V_0 is = %s' % volume_constraints[index])
    print('Final eps is = %s' % epsilon_list[17])
    print(plot_dir)
    print(saved_model_dir)

    """Phase field theory parameters (use the right parameters for MC initialization"""
    # Constraints on surface area and volume of the vesicle
    A_0 = area_constraints[0]
    V_0 = volume_constraints[index]
    # Vesicle thickness parameter epsilon and spontaneous curvature c_0
    eps = epsilon_list[17]
    c_zero = 0.0
    print('Theoretical_reduced_vol = %s' % (V_0 / ((4.0 * np.pi / 3.0) * ((A_0 / (4.0 * np.pi)) ** 1.5))))

    # Instantiate the neural network object using the PhaseFieldNet() class
    n_net = PhaseFieldNet()
    n_net.apply(init_weights)
    n_net.to(device)

    # load saved models
    if load_previous_model:
        n_net.load_state_dict(torch.load("saved_model_0/prime_PhaseField_150000Epoch_gpu.pt"))

    # Optimizer used for optimizing the network parameters
    eta = torch.tensor(1.0e-2, device=device)
    func_optimizer = optim.Adam(n_net.parameters(), lr=1.0e-3, weight_decay=0.0, amsgrad=True)

    # Compute the integrals of a trained model using evenly spaced points (Riemann sum)
    """r_min = -3.0
    r_max = 3.0
    evenly_spaced_integration()"""

    # Toggle the training using fixed points in the space
    if pts_fixed_training_mode:

        # Lower bound and higher bound of coordinates along each axis
        r_min = -2.7
        r_max = 2.7

        # Create evenly spaced points with point number equal to mesh size cubed (for 3D case)
        spatial_pts = evenly_spaced_r()

        training_epochs = 80001
        fixed_pt_training(training_epochs, spatial_pts)

    # Test the trained model
    if test_mode:
        device = torch.device('cpu')
        n_net.to(device)
        test_mesh_size = 120
        grid_pts = torch.linspace(-3, 3, test_mesh_size)
        vesicle_phase_field = torch.zeros([test_mesh_size, test_mesh_size, test_mesh_size])

        n_net.eval()
        with torch.no_grad():
            for m in range(test_mesh_size):
                for n in range(test_mesh_size):
                    for w in range(test_mesh_size):
                        sp = torch.tensor([grid_pts[m], grid_pts[n], grid_pts[w]], requires_grad=False)
                        vesicle_phase_field[m, n, w] = n_net(sp.view(1, dim))

    else:
        test_mesh_size = 0
        grid_pts = torch.linspace(-3, 3, test_mesh_size)
        vesicle_phase_field = torch.zeros([test_mesh_size, test_mesh_size, test_mesh_size])

    if one_D_plots and test_mode:
        middle_idx = int(test_mesh_size / 2 - 1)
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[:, 0, 0], grid_pts, 'left_x')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[:, middle_idx, middle_idx], grid_pts, 'middle_x')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[:, -1, -1], grid_pts, 'right_x')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[0, :, 0], grid_pts, 'left_y')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[middle_idx, :, middle_idx], grid_pts, 'middle_y')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[-1, :, -1], grid_pts, 'right_y')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[0, 0, :], grid_pts, 'left_z')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[middle_idx, middle_idx, :], grid_pts, 'middle_z')
        plot_phase_field(plot_dir, vesicle_phase_field.numpy()[-1, -1, :], grid_pts, 'right_z')

    if color_map_plot and test_mode:
        middle_idx = int(test_mesh_size / 2 - 1)
        color_map_2d(plot_dir, grid_pts, 'x', 'y', vesicle_phase_field.numpy()[:, :, middle_idx])
