function [L, fwts] = laplacian3d(x, t, type)

%% vectorized laplacian for tet meshes
nv = size(x, 1);

esi = [1 1 1 2 2 3];
eei = [2 3 4 3 4 4];

if nargin>2 && strcmp(type, 'uniform')
%     VV = sparse(t(:,esi), t(:,eei), true, nv, nv);
%     L = VV | VV';

    VV = sparse(t(:,esi), t(:,eei), 1, nv, nv);
    L = boolean(VV + VV');
    L = spdiags(-sum(L,2), 0, double(L));
else
    tfaces = [t(:, 2:4); t(:, [1 4 3]); t(:, [1 2 4]); t(:, [1 3 2])];
    tnormals = cross3( x(tfaces(:,1), :) - x(tfaces(:,2), :), x(tfaces(:,1), :) - x(tfaces(:,3), :) );
    tnormals2 = permute( reshape(tnormals', 3, [], 4), [3 1 2] );

    vols = signedVolume(x, t)';
    ndot = squeeze( sum( tnormals2(esi, :, :).*tnormals2(eei, :, :), 2 ) );
    fwts = ndot./vols;
    L = sparse(t(:, [esi eei]), t(:, [eei esi]), [fwts; fwts]', nv, nv);
    L = spdiags(-sum(L,2), 0, L);
end

function c = cross3(a, b)
c(:,1) = a(:,2).*b(:,3)-a(:,3).*b(:,2);
c(:,2) = a(:,3).*b(:,1)-a(:,1).*b(:,3);
c(:,3) = a(:,1).*b(:,2)-a(:,2).*b(:,1);