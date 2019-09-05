function z = ABF_interp(x1, x2, t, wt, reconMethod)

if nargin<5 || (reconMethod ~= 'greedy' && reconMethod ~= 'eigen' && reconMethod ~= 'leastsquare')
    reconMethod = 'eigen'; %'greedy'; 'eigen'; % leastsquare
end

%% abf interp
tr = triangulation(t, x1);
edges = tr.edges;
nedge = size(edges, 1);
nv = size(x1, 1);
vv2EdgeIdx = sparse(edges, edges(:,[2 1]), repmat( (1:nedge)', 1, 2 ), nv, nv);
edgeIdxPerTet = at_sparse(vv2EdgeIdx, uint64(t(:, [1 2 3 2 1 1])), uint64(t(:, [4 4 4 3 3 2])));


fLen2s = @(x) sum(x.^2, 2);
els_source = fLen2s(x1(edges(:,1),:) - x1(edges(:,2),:)); %all els
els_target = fLen2s(x2(edges(:,1),:) - x2(edges(:,2),:)); %all els


%% source/target dihendral angles
da_src = dihedralAnglesFromTetELS( els_source(edgeIdxPerTet) );
da_tgt = dihedralAnglesFromTetELS( els_target(edgeIdxPerTet) );
da_src = da_src(:, [3 2 4 1 5 6]); da_tgt = da_tgt(:, [3 2 4 1 5 6]); % convert to ABF order


%% get neighbor information for abfflaten
tetneighbors = tr.neighbors;
tetneighbors(isnan(tetneighbors)) = 0;
tetneighbors = int32( tetneighbors-1 );


%%
% the outer iteration changes the regularizer: difference to the target interpolated dihedral angles, 
% we do not want that, so set nOuterIter = 1
da_blend = (1-wt)*da_src+wt*da_tgt;
z = abfflatten(da_blend, x1, int32(t-1), tetneighbors, struct('reconstruction', reconMethod, 'regularization', 1e-8, 'nOuterIter', 1, 'min_angle', 0.01));

%% fix global scaling DOF
% [~, z] = bestIsoScale(x1, z, t);
% interp volumne
v_interp = (1-wt)*sum(signedVolume(x1, t)) + wt*sum(signedVolume(x2, t));
s = v_interp/sum(signedVolume(z, t));
z = z*nthroot(s, 3);