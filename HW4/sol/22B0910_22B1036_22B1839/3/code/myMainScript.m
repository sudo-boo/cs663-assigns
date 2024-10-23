clc;
tic;

k = 5;
fprintf("Running it for %d times....\n", k);
for i = 1:k
    fprintf("Error in Iteration %d:\t", i);
    runSimSVD();
end

toc;


%-------------------------------------------------------------------------------
%                           FUNCTIONS
%-------------------------------------------------------------------------------


function runSimSVD()
    m = 3;
    n = 3;
    A = rand(m, n);

    % Compute A^T * A and A * A^T
    AtA = A' * A;
    AAt = A * A';
    
    % Compute eigenvalues and eigenvectors
    [V, D_v] = eig(AtA);                % Right singular vectors
    [U, ~] = eig(AAt);                  % Left singular vectors
    
    % Extract eigenvalues (diagonal entries of D_u or D_v)
    singular_values = sqrt(diag(D_v));  % Eigenvalues correspond to SSV
    
    % Sort singular values and rearrange eigenvectors accordingly
    [sorted_singular_values, idx] = sort(singular_values, 'descend');
    V = V(:, idx);                      % Sort the right singular vectors
    U = U(:, idx);                      % Sort the left singular vectors
    
    % Construct the diagonal matrix of singular values
    S = diag(sorted_singular_values);
    A_reconstructed = U * S * V';       % Reconstructed matrix
    
    % Check the difference between the reconstructed matrix and A
    error = norm(A - A_reconstructed);
    disp(error);
end

%-------------------------------------------------------------------------------