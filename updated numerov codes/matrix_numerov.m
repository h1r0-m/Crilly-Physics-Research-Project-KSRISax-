%% initialization

% infinite potential, hard boundary condition at r = r_box
r_box = 30;

% avoiding r_points(1) to be 0, because can blow up
r_start = 1e-5;

% # of points, distance interval, and orbital quantum number designation
N_points = 2000;
d = (r_box-r_start) / (N_points-1);
l = 0;

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

% finding eigenvecs and eignevals, using full() because A and B are sparse
% matrices
[eig_vec, D] = eig(full(-1/2 .* A + B * V_eff), full(B));

% extracting eigenvals from the diagonalized eigenval matrix and ordering
eig_val = diag(D);
[energies, idx] = sort(eig_val);
psi = eig_vec(:, idx);

% checking ground state which should be E = -0.5 Ha
fprintf('Ground State Energy: %.5f Hartree\n', energies(1));

%% plotting

figure
hold on;
n_lim = 20;
plot(1:n_lim, energies(1:n_lim))