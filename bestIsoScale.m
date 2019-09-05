function [scale, z] = bestIsoScale(x, y, t)

nv = size(x, 1);
nt = size(t, 1);

V2Em = sparse( [1:3 1:3] + (0:nt-1)'*3, t(:, [1 1 1 2:4]), repmat([-1 -1 -1 1 1 1], nt, 1), 3*nt, nv);
M1 = blockblas('inverse', V2Em*x);
toff = repmat( reshape( repmat( 0:nt-1, 3, 1 )*3, [], 1 ), 1, 3 );
X2J = sparse(repmat((1:3)', nt, 3)+toff, repmat(1:3, 3*nt, 1)+toff, M1)*V2Em;

vol = reshape(repmat( signedVolume(x, t), 1, 3 )', [], 1);

J = X2J*y;
invJ = blockblas('inverse', J);

scale = sqrt( norm( sqrt(vol).*invJ, 'fro' )/norm( sqrt(vol).*J, 'fro' ) );

if nargout>1
    z = y*scale;
end

J2 = J*scale;
Emin = sum(vol)*2;
E0 = norm( sqrt(vol).*J, 'fro' )^2 + norm( sqrt(vol).*invJ, 'fro' )^2;
E1 = norm( sqrt(vol).*J2, 'fro' )^2 + norm( sqrt(vol).*blockblas('inverse', J2), 'fro' )^2;

fprintf('Eiso before/after global scaling: %e, %e', E0-Emin, E1-Emin);