function R = projectJacobi2Rotation(A, sol)

if nargin<2, sol = 'AVX'; end

d = size(A, 2);

if strcmpi(sol, 'AVX')
    R = project3DRotations(A);
elseif strcmpi(sol, 'sse')
    R = project3DRotations(A, 'sse');
elseif strcmpi(sol, 'scalar')
    R = project3DRotations(A, 'scalar'); % avx
elseif strcmpi(sol, 'loop')
    R = A;
    assert( mod(size(A, 1), d) == 0 );
    for i=1:size(A,1)/d
        [u, s, v] = svd(A((i-1)*d+(1:d),:));
        if det(u*v')<0
            s(d, d) = -s(d, d);
            v(:, d) = -v(:, d);
        end
        R((i-1)*d+(1:d),:) = u*v';
    end
elseif strcmpi(sol, 'eigen') % Eigen Version
    R = projectBlockRotation(A.').';
else
    error( 'unknown rotation projection solver: %s', sol );
end