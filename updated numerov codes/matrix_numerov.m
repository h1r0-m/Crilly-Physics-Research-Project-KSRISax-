%% initialization

% infinite potential, hard boundary condition at r = r_box
r_box = 30;

% avoiding r_points(1) to be 0, because can blow up
r_start = 1e-5;

% # of points, distance interval, and orbital quantum number designation
N_points = 4000;
d = (r_box-r_start) / (N_points-1);
l = 0;

% choosing which energy level to investigate up to
n_lim = 10;

r_points = linspace(r_start, r_box, N_points)';

%% forming the matrices for implementation of the numerov matrix method

A_lower = ones(N_points-2,1);
A_mid = -2*ones(N_points-2, 1);
A_upper = ones(N_points-2, 1);

A = spdiags([A_lower, A_mid, A_upper], [-1,0,1], N_points-2, N_points-2) ./ d^2;

B_lower = ones(N_points-2, 1);
B_mid = 10*ones(N_points-2, 1);
B_upper = ones(N_points-2, 1);

B = spdiags([B_lower, B_mid, B_upper], [-1,0,1], N_points -2, N_points-2) ./ 12;

% V_eff = V_coulomb + centrifugal component
V_eff_vec = -1 ./ r_points + l .* (l+1) ./ (2 .* r_points .^2);

V_eff = spdiags(V_eff_vec(2:end-1), 0, N_points - 2, N_points -2);

% calculating eigenvals and eigenvecs using the eigs function
[eig_vec, D] = eigs(-1/2 .* A + B * V_eff, B, n_lim, "smallestabs");

% extracting eigenvals from the diagonalized eigenval matrix and ordering
eig_val = diag(D);
[energies, idx] = sort(eig_val);
psi = eig_vec(:, idx);

% checking ground state which should be E = -0.5 Ha
fprintf('Ground State Energy: %.5f Hartree\n', energies(1));

%% plotting

figure;
hold on;

% Plot numerical energies with markers
plot(1:n_lim, energies(1:n_lim), 'x', "MarkerSize", 15, "LineWidth", 2, 'DisplayName', 'Numerical');

% Plot analytical hydrogen energies as a smooth curve
fplot(@(x) -1./(2*x.^2), [1 n_lim], 'r', 'LineWidth', 1.5, 'DisplayName', 'Analytical (E = -1/(2n^2))');

xlabel('n', 'FontSize', 15);
ylabel('E (Ha)', 'FontSize', 15);
title(sprintf('Hydrogen Energy Levels (l = %d, N_{points} = %d, r_{box} = %.2f, E_1 = %.5f Ha)', l, N_points, r_box, energies(1)), 'FontSize', 14);
legend('Location', 'northwest', "FontSize", 15);
grid on;
xlim([1 n_lim]);

% saving
filename = sprintf('hydrogen_energies_l%d_N%d_r%.2f.png', l, N_points, r_box);

exportgraphics(gcf, filename, 'Resolution', 300);
