clear; clc; close all;

N = 2000;
R_box = 30;
R_c = 15;

% syntax: schrodingerEval(l, N, R_box, R_c, potential_index, end_condition_index, n_wall, lambda_yukawa)

n_1 = 1;
l_1 = 0;
m_1 = 0;

% syntax: Visual3D(l, r, u, n, m, R_box)

[r_1, u_1, energies_1] = schrodingerEval(l_1, N, R_box, R_c, 0, 0);
figure
Visual3D(l_1,r_1,u_1,n_1,m_1,R_box)

%% makinng gif

gif_filename = 'Orbital_Compression.gif';

% Define the sequence of compression: Shrink R_c from 30 down to 5
Rc_values = linspace(3, 1e-10, 20);

% Create a fixed figure window
hFig = figure('Color', 'w', 'Position', [100, 100, 800, 600]);

for k = 1:length(Rc_values)
    current_Rc = Rc_values(k);
    
    % A. Solve Schrödinger Eq for current compression
    % Note: We use current_Rc for the physics
    [r, u, energies] = schrodingerEval(l_1, N, R_box, current_Rc, 0, 0);
    
    % B. Visualization
    clf(hFig); % Clear previous frame
    
    % We pass R_box to keep the camera zoom constant
    Visual3D(l_1, r, u, n_1, m_1, R_box);
    
    % Add a title showing the current wall position
    title(sprintf('Compression R_c = %.1f a.u.\n(n=%d, l=%d, m=%d)', ...
        current_Rc, n_1, l_1, m_1), 'FontSize', 14);
    
    % Force the axes to stay fixed so we see the wall moving in
    xlim([-5, 5]);
    ylim([-5, 5]);
    zlim([-5, 5]);
    
    % Update the drawing immediately
    drawnow;
    
    % C. Capture Frame for GIF
    frame = getframe(hFig);
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);
    
    % Write to GIF file
    if k == 1
        % First frame: create file
        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
    else
        % Subsequent frames: append
        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
    end
    
    fprintf('Frame %d/%d processed (Rc = %.1f)\n', k, length(Rc_values), current_Rc);
end

fprintf('Done! Saved as %s\n', gif_filename);
%% functions

function [r, u,energies] = schrodingerEval(l, N, R_box, R_c, potential_index, end_condition_index, n_wall, lambda_yukawa)
    
    % l (orbital quantum number): 0 - s-orbital (sphere), 1 - p-orbital (dumbell) etc..
    % potential index: 0 - hard wall, 1 - soft wall, 2 - yukawa potential
    % n_wall: n (steepness) for soft wall
    % lambda_yukawa: lambda parameter for yukawa potential
    % end_condition_index: 0 - dirichlet (u = 0), 1 - neumann (u' = 0)

    if (potential_index == 0) % hard wall, no extra terms - just hard coding wave function to be 0 at R_c
        dr = R_c / (N+1);
        r = linspace(dr, R_c-dr, N)';
    else % soft wall or yukawa
        dr = R_box / (N+1);
        r = linspace(dr, R_box - dr, N)';
    end

    % potentials (atomic units)
    V_coulomb = -1.0 ./ r;

    V_centrifugal = (l*(l+1)) ./ (2 * r.^2);
    
    switch potential_index
        case 0 % hard wall
            V_eff = V_coulomb + V_centrifugal;
        case 1 % soft wall
            V_extra = (r/R_c).^n_wall;
            V_eff = V_coulomb + V_centrifugal + V_extra;
        case 2 % yukawa potential
            V_yukawa = -1/r * exp(-r/lambda_yukawa) ;
            V_eff = V_yukawa + V_centrifugal;
    end
    
    % constructing hamiltonian matrix
    % building the central difference B[1,-2,1] matrix for the second
    % derivative
    onesVec = ones(N, 1);
    Laplacian = spdiags([onesVec, -2*onesVec, onesVec], -1:1, N, N) / dr^2;
    
    % end condition: if u want u = 0 (dirichlet condition) at the end (better
    % for hard wall apparently), then keep it as it is, if you want u' = 0
    % (neumann condition), then:
    
    switch end_condition_index
        case 0
    
        case 1
            Laplacian(end,end) = -1 / dr^2;
    end
    
    % Laplacian(end, end) = Laplacian(end, end-1); 
    T = -0.5 * Laplacian;
    
    % diagonal matrix for potentials
    V_matrix = spdiags(V_eff, 0, N, N);
    
    H = T + V_matrix;
    
    % finding eigenvalues (energy) and corresponding eignevector (u vectors)
    [Vectors, D_full] = eig(full(H)); 
    
    % extracting diagonal (energies)
    all_energies = diag(D_full);
    
    % sorting the energies 
    [energies, idx] = sort(all_energies);
    
    % sorting the wave functions
    u = Vectors(:, idx);
end

function Visual3D(l, r, u, n, m, R_box)
    % some initial configurations

    % this is necessary because we're deciding l first, and n corresonds to
    % the nth lowest energy level for that particular l, e.g. if l = 2,
    % then n = 3 以上しかありえない, and therefore (n=3,l=2) ペアはit becomes
    % associated to the lowest energy for that l, therefore n = n-l works.

    k = n - l;
    if k < 1 || k > size(u, 2)
        error('The quantum number n=%d does not exist for l=%d.', n, l);
    end
    Box_View = R_box;
    Grid_Res = 200; 

    % creating 3d grid
    vals = linspace(-Box_View, Box_View, Grid_Res);
    [X, Y, Z] = meshgrid(vals, vals, vals);

    % converting to spherical coordinates, so calculating r,theta, and phi
    % for each point on the 3d grid
    R_3D = sqrt(X.^2 + Y.^2 + Z.^2);
    Theta = acos(Z ./ R_3D);
    Phi = atan2(Y, X);
    Theta(isnan(Theta)) = 0; % at the center, theta becomes NaN because division by 0 so fixing that

    % computing radial part, so basically interpolating between the two
    % closest points of r to find the relevant u value, and then changing
    % into R(r) using R(r) = u(r) / r (reduced radial function --> actual
    % radial wave function)
    u_interp = interp1(r, u(:, k), R_3D, 'linear', 0);
    Radial_Part = u_interp ./ R_3D;
    Radial_Part(isnan(Radial_Part)) = 0; 

    % angular part, using legendre polynomials
    P_all = legendre(l, cos(Theta)); 
    m_abs = abs(m);

    % robust extraction of Legendre polynomial P_l^m
    % flatten the spatial dimensions to handle l=0 and l>0 uniformly
    P_flat = reshape(P_all, l+1, []);  % Shape: [l+1, Total_Points]
    P_lm_flat = P_flat(m_abs + 1, :);  % Shape: [1, Total_Points]
    P_lm = reshape(P_lm_flat, size(Theta)); % Shape: [Grid_Res, Grid_Res, Grid_Res]

    % combining with phi
    if m == 0
        Angular_Part = P_lm;
    elseif m > 0
        Angular_Part = P_lm .* cos(m * Phi);
    else
        Angular_Part = P_lm .* sin(abs(m) * Phi);
    end

    % actual wave function is the combination of the radial and angular
    % part
    Psi = Radial_Part .* Angular_Part;
    Density = Psi.^2;

    % dynamic slicing (for the visualization of probability density)
    if m == 0
        h = slice(X, Y, Z, Density, [], 0, []); % Cut Y=0 (Vertical)
    elseif m > 0
        h = slice(X, Y, Z, Density, [], [], 0); % Cut Z=0 (Horizontal)
    else
        h = slice(X, Y, Z, Density, 0, [], []); % Cut X=0 (Vertical)
    end
    set(h, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
    colormap(hot);

    iso_val = max(Density(:)) * 0.02; 
    p = patch(isosurface(X, Y, Z, Density, iso_val));
    p.FaceColor = 'cyan'; p.EdgeColor = 'none'; p.FaceAlpha = 0.3;

    axis equal; 
    xlim([-Box_View Box_View]); ylim([-Box_View Box_View]); zlim([-Box_View Box_View]);
    title(sprintf('Orbital: n=%d, l=%d, m=%d', n, l, m));
    camlight; lighting phong; view(3);
    xlabel("x")
    ylabel("y")
    zlabel("z")
end