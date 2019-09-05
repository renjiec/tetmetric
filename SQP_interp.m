classdef SQP_interp < handle   %InterpolartorSQP
    properties(SetAccess = protected)
        x1
        x2
        t
        optimization_method
        reconstructionMethod = 'greedy_graph'  %options from ABF3D: eigen, leastsquare, greedy
        path
        els_source
        els_target
        edges
        E2M0
        E2Mnzidx
        H_SD0
        Hnzidx
        J0
        Jnzidx
        JtJ0
        JtJnzidx
        A0
        Anzidx
        MdaSum
        MSumPerTet
        edgeIdxsPerTet
        interiorEdgeIdxs
        eIdPerTets
        solver
        tolCurvature = 1e-6;
    end
    
    methods
        function I = SQP_interp(x1, x2, t, method, reconstruction)
            assert( all(size(x1) == size(x2)), 'x_source and x_target must have matching dimension' );

            if nargin<4, method = 1; end %1-Newton(soft constraints), 3-SDP
            if nargin<5, reconstruction = 'greedy_graph'; end

            I.x1 = x1; I.x2 = x2; I.t = t; I.optimization_method = method; I.reconstructionMethod = reconstruction;
            
            tic
            preprocess(I);
            fprintf('preprocessed in %fs\n', toc);
        end

        function setReconstructionMethod(I, reconstruction)
            I.reconstructionMethod = reconstruction;
        end
        
        function preprocess(I)
            source_vertices = I.x1;
            target_vertices = I.x2;

            %% preprocessing - part 1
            S = triangulation(I.t, I.x1);
            I.edges = S.edges;

            numEdges = size(I.edges, 1);
            I.els_source = squarednorm(source_vertices(I.edges(:,1), :) - source_vertices(I.edges(:,2), :));
            I.els_target = squarednorm(target_vertices(I.edges(:,1), :) - target_vertices(I.edges(:,2), :));

            nv = size(I.x1, 1);
            nt = size(I.t, 1);
            vv2EdgeIdx = sparse(I.edges, I.edges(:,[2 1]), repmat( (1:numEdges)', 1, 2 ), nv, nv);

            tIdx = uint64(I.t);
            tetEdgeIdxs = at_sparse(vv2EdgeIdx, tIdx(:, [1 2 3 2 1 1]), tIdx(:, [4 4 4 3 3 2]));
            % eIdPerTets  = at_sparse(vv2EdgeIdx, tIdx(:, [1 1 1 2 2 3]), tIdx(:, [2 3 4 3 4 4]));
            I.eIdPerTets = tetEdgeIdxs(:,[6,5,1,4,2,3]);

            boundary_faces = S.freeBoundary;
            boundary_half_edges = reshape( boundary_faces(:,[1 1 2 2 3 3, 2 3 3 1 1 2]), [], 2 );
            [~, I.interiorEdgeIdxs]= setdiff(I.edges,boundary_half_edges,'rows');

            %%
            E2Mi = kron((1:nt*6)', ones(1, 6));
            E2Mj = kron(I.eIdPerTets, ones(6,1));
            I.E2M0 = sparse(E2Mi, E2Mj, 1, nt*6, numEdges );
            I.E2Mnzidx = ij2nzIdxs(I.E2M0, uint64(E2Mi), uint64(E2Mj));

            Mi = kron(I.eIdPerTets, ones(6,1));
            Mj = blockblas('transpose', Mi);
            I.H_SD0 = sparse(Mi, Mj, 1, numEdges, numEdges);
            I.Hnzidx = ij2nzIdxs(I.H_SD0, uint64(Mj), uint64(Mi))';

            colIdx = kron(tetEdgeIdxs, ones(6, 1));
            rowIdx = blockblas('transpose', colIdx);
            I.J0 = sparse(rowIdx, colIdx, 1, numEdges, numEdges);
            I.Jnzidx = ij2nzIdxs(I.J0, uint64(rowIdx), uint64(colIdx))';
            
            I.JtJ0 = I.J0(I.interiorEdgeIdxs,:)'*I.J0(I.interiorEdgeIdxs,:);
            [JtJ0i, JtJ0j] = find(I.JtJ0);
            JtJ0ij = sort([JtJ0i JtJ0j], 2);

            I.A0 = I.JtJ0 + I.H_SD0;
            I.solver = pardisosolver(I.A0, 'ldlt');
        
            % nonzero indices of H_SD and JtJ in matrix A
            I.Anzidx{1} = ij2nzIdxs(I.A0, uint64(Mj), uint64(Mi))';
            I.Anzidx{2} = ij2nzIdxs(I.A0, uint64(JtJ0i), uint64(JtJ0j))';

            % 
            I.JtJ0 = triu(I.JtJ0);  % keep only upper triangle
            I.JtJnzidx = ij2nzIdxs(I.JtJ0, uint64(JtJ0ij(:,1)), uint64(JtJ0ij(:,2)))';
            
            %% for greedy reconstruction 
            root = 1;
            I.path = bfs(S,root);

            I.edgeIdxsPerTet = tetEdgeIdxs;
        end
        
        function [E2M, invAs] = E2MFromELS(I, els)
            fELSi = @(i) els( I.edgeIdxsPerTet(:, i) );
            xref = tetEmbedFromEdgeLength2( fELSi(3), fELSi(2), fELSi(4), fELSi(1), fELSi(5), fELSi(6) ); % see metric_interp.m

            nt = size(I.t, 1);
            fEdgeIndexPerTet = @(idx) reshape(idx' + (0:nt-1)*6, [], 1);
            fVtxIndexPerTet = @(idx) reshape(idx' + (0:nt-1)*3, [], 1);
            evs = zeros(nt, 3);
            evs(fEdgeIndexPerTet(1:3),:) = xref;
            evs(fEdgeIndexPerTet(4:6),:) = xref( fVtxIndexPerTet([2 3 3]), : ) - xref( fVtxIndexPerTet([1 1 2]), : );

            matAs(:, [1 4 6]) = evs.^2;
            matAs(:, [2 3 5]) = evs(:, [1 1 2]).*evs(:, [2 3 3])*2;

            invAs = blockblas('inverse', matAs);
            
            E2M = replaceNonzeros(I.E2M0, myaccumarray(I.E2Mnzidx, invAs));
        end
        
        function z = interp(I, t_interpolation)
            MAX_ITER = 1e3;
            maxBisectionIterations = 60;
            line_serach_factor = 0.5;

            numEdges = size(I.edges, 1);

            method = I.optimization_method;
            dihedral_input_indices = I.edgeIdxsPerTet;
            interior_els_indices = I.interiorEdgeIdxs;

            fSubVec = @(x, i) x(i);
            fGaussCurvature = @(els) accumarray(dihedral_input_indices(:), reshape(dihedralAnglesFromTetELS(els(dihedral_input_indices)), [], 1), [numEdges 1]);
            fInteriorAngleDefect = @(els) 2*pi-fSubVec(fGaussCurvature(els), interior_els_indices);

            tolerance_curvature = I.tolCurvature;
            
            nt = size(I.t, 1);
            els_blend = (1-t_interpolation) * I.els_source + t_interpolation * I.els_target;

            %% preprocessing - part 2, based on blend edge length    
            [E2Metric_mat, invAs] = I.E2MFromELS(els_blend);

            volumes = tetVolumnFromEdgeLen2(els_blend(dihedral_input_indices(:,[6 5 1 4 2 3])'))';
            weights = volumes/sum(volumes);


            fSymmDirichEn = @(el) isometricEnergyFromMetric3Dc(reshape(E2Metric_mat*el, 6, [])', invAs, weights);
            
            da_diff_prev = fInteriorAngleDefect(els_blend);

            %% initialization before SQP
            els_prev = els_blend;

            %% SQP
            iter = 1;
            lineSearchSuccessFlag = 1;
            lambda = min( 1/norm(da_diff_prev, 'inf'), 1e10 );
            while iter <= MAX_ITER  &&  tolerance_curvature < norm(da_diff_prev,'inf') && lineSearchSuccessFlag
                %% objective energy function for line search
                mySDEmbedEnergy = @(els) fSymmDirichEn(els) + lambda*norm( fInteriorAngleDefect(els) )^2;
                
                %% compute J (at prev posiotin)
                Jda = dihedralAnglesJacobian(els_prev(dihedral_input_indices)');
                J = replaceNonzeros(I.J0, myaccumarray(I.Jnzidx, Jda));
                J = J(interior_els_indices,:); %extract only rows of inner edges

                if method == 3 % SDP
                    tsdp = tic;
                    Midx = sub2ind([6 6 nt], repmat([1 1 1 2 2 3], 1, nt), repmat([1 2 3 2 3 3], 1, nt), reshape(repmat(1:nt, 6, 1), 1, []))';
                    diagQidx = sub2ind([6 6 nt], repmat(1:6, nt, 1), repmat(1:6, nt, 1), repmat(1:nt, 6, 1)');
                    cvx_begin sdp quiet
                    cvx_solver Mosek
                        variables els(numEdges,1)
                        variable Q(6, 6, nt) semidefinite

%                         minimize( weights'*sum(Q(diagQidx), 2) +  sum_square_abs(J*els-da_diff_prev)*lambda ) % \sum SD
                        minimize( max( sum(Q(diagQidx), 2) ) +  sum_square_abs(J*els-da_diff_prev)*lambda )  % max (SD)

                        subject to
                            Q(1:3, 4:6, :) == repmat(eye(3,3),1,1,nt); %#ok<*EQEFF>
                            Q(Midx) == E2Metric_mat*els;
                    cvx_end

                    if contains(cvx_status, 'Solved')
                        els_candidate = els;
                    else
                        error('something went wrong! cvx_status: %s', cvx_status);
                    end
                    sdPerTet = sum(Q(diagQidx), 2) - 6;
                    maxk = norm(fInteriorAngleDefect(els_candidate), inf);
                    fprintf('cvx iter %2d: time = %.2f, |K|_inf = %.1e, |SD|_inf = %.1e, |SD|_avg = %.1e, lambda = %1.0e\n', iter, toc(tsdp), maxk, max(sdPerTet), mean(sdPerTet), lambda);
                else
                    %% compute H_SD and G_SD (at prev posiotin)
                    [~, gSD, hSD] = fSymmDirichEn(els_blend);
                    G_SD = myaccumarray(uint64(I.eIdPerTets)', gSD, zeros(numEdges, 1));

                    %% build linear system (A*delta_x = rhs)
                    if method==1 % Newton
                        uJtJnzs = spAtA_nonzeros(J, I.JtJ0); % upper triangle JtJ
                        Anzvals0 = myaccumarray(I.Anzidx{1}, hSD);
                        Anzvals = myaccumarray(I.Anzidx{2}, uJtJnzs(I.JtJnzidx)*2*lambda, Anzvals0);
                        A = replaceNonzeros(I.A0, Anzvals);

                        rhs = 2*lambda*J'*da_diff_prev - G_SD;
                    end

                    delta_x = I.solver.refactor_solve(A, rhs);

                    %% compute next els
                    delta_x = delta_x(1:numEdges);
                    els_candidate = els_prev + delta_x;
                end

                %% line-search
                injectiveStep = maxInjectiveStepSizeForMetrics(reshape(E2Metric_mat*els_prev, [6 nt])', reshape(E2Metric_mat*els_candidate, [6 nt])');
                [els_next, lineSearchSuccessFlag] = quicklineSearchInterpolation(mySDEmbedEnergy, maxBisectionIterations, els_candidate, els_prev, injectiveStep, line_serach_factor);

                da_diff_after_lineSearch = fInteriorAngleDefect(els_next);

                %% preparations for next iteration
                if lineSearchSuccessFlag
                    els_prev = els_next;
                    da_diff_prev = da_diff_after_lineSearch;
                    iter = iter+1;
                    lambda = 1/norm(da_diff_prev, 'inf'); % update penalty weight
                end
            end % END OF SQP

            if ~exist('G_SD', 'var'), G_SD = 0; end
            if ~exist('els_next', 'var'), els_next = els_blend; end
            sdPerTet = symmetricDirichletEnergyFromMetric(reshape(E2Metric_mat*els_next, [6 nt])');
            fprintf("t = %3.3f, E = %.1e, nIt = %d, |K|_inf = %.1e, |K|_avg = %.1e, |SD|_inf = %.1e, |G_SD| = %.1e, lambda = %.0e\n", t_interpolation, fSymmDirichEn(els_next), iter-1, norm(da_diff_prev,'inf'), mean(abs(da_diff_prev)), max(sdPerTet), norm(G_SD), lambda);

            %% greedy reconstruction 
            if lineSearchSuccessFlag
                if iter == 1, els_next = els_blend; end
            else
                els_next = els_prev;
            end

            if strcmp(I.reconstructionMethod, 'greedy_graph')
                z = tetMeshReconstructionGreedy(I.t, dihedral_input_indices, els_next, I.path, size(I.x1, 1));
            else
                z = tetMeshReconstructFromELS(els_next, I.edges, I.t, I.x1, I.reconstructionMethod);
                v_interp = (1-t_interpolation)*sum(signedVolume(I.x1, I.t)) + t_interpolation*sum(signedVolume(I.x2, I.t));
                s = v_interp/sum(signedVolume(z, I.t));
                z = z*s^(1/3);
            end
%             fprintf('reconstruction error = %.3e\n', norm( squarednorm(z(I.edges(:,1), :) - z(I.edges(:,2), :))-els_next, 'inf'));
        end
    end
end

function e = symmetricDirichletEnergyFromMetric(M, weights)
    adf = M(:, [1 4 6]);
    ecb = M(:, [5 3 2]);
    minsd = 6;
    e = sum(adf, 2) + sum( ecb.^2-adf(:, [1 1 2]).*adf(:,[2 3 3]), 2)./(sum(adf.*ecb.^2,2) - prod(adf,2) - 2*prod(ecb,2)) - minsd;
    if nargin>1, e = dot(e, weights); end
end
    
function [els_out, successFlag, t, num_steps] = quicklineSearchInterpolation(mySDEmbedEnergy, maxBisectionIterations, els_candidate, els_prev, injectiveStep, line_serach_factor)
                               
tolerance_delta_els = 1e-15;
            
successFlag = false;

t = min(0.95*injectiveStep, 1);


els_diff = els_candidate - els_prev;
norm_els_diff = norm(els_diff);
els_t = els_prev + t*els_diff;

E_tot_prev = mySDEmbedEnergy(els_prev);

for j = 1:maxBisectionIterations
    E_tot_current = mySDEmbedEnergy(els_t);
    
    if norm_els_diff*t < tolerance_delta_els
        successFlag = false;
        break;
    end

    if E_tot_current < E_tot_prev
        successFlag = true;
        break;
    else
        t = line_serach_factor*t; %keep cutting 't'
        els_t = els_prev + t*els_diff;
    end
    
end

if successFlag
    els_out = els_t;
else
    els_out = els_prev;
end

num_steps = j;

end

function s = squarednorm(x)
    s = sum(x.^2, 2);
end