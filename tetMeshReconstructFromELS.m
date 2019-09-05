function z = tetMeshReconstructFromELS(els, edges, t, x0, reconMethod)

if nargin<5
    reconMethod = 'eigen'; %'greedy'; 'eigen'; % leastsquare
end

% edges = tr.edges;
nedge = size(edges, 1);
nv = size(x0, 1);
vv2EdgeIdx = sparse(edges, edges(:,[2 1]), repmat( (1:nedge)', 1, 2 ), nv, nv);
edgeIdxPerTet = at_sparse(vv2EdgeIdx, uint64(t(:, [1 2 3 2 1 1])), uint64(t(:, [4 4 4 3 3 2])));

%% source/target dihendral angles
da = dihedralAnglesFromTetELS( els(edgeIdxPerTet) );
da = da(:, [3 2 4 1 5 6]);  % reorder to ABF input

%% get neighbor information for abfflaten
tr = triangulation(t, x0);
tetneighbors = tr.neighbors;
tetneighbors(isnan(tetneighbors)) = 0;
tetneighbors = int32( tetneighbors-1 );

%%
% the outer iteration changes the regularizer: difference to the target interpolated dihedral angles, 
% we do not want that, so set nOuterIter = 1
z = abfflatten(da, x0, int32(t-1), tetneighbors, struct('reconstruction', reconMethod, 'regularization', 1e-8, 'nOuterIter', 0, 'nInnerIter', 0, 'min_angle', 0.01));

fSqrLen = @(x) sum(x.^2, 2);
els2 = fSqrLen(z(edges(:,1), :) - z(edges(:,2), :));
ss = sqrt( mean(els2./els) );
z = z/ss;

% fprintf('reconstruction error = %.3e\n', norm(fSqrLen(z(edges(:,1), :) - z(edges(:,2), :)) - els, inf ));