classdef ShapeSpace_interp < handle
    properties(SetAccess = protected)
        x1
        x2
        t
        anchorId
        M1
        M2
        X2Js1
        X2Js2
        vol1
        vol2
        gIdx
        h0
        Hnzidx
        solver
        zlast
        energy_type   % 1: symmDirichlet, 5: arap
        initFromLast
        p2p_weight = 1e5
        maxIteration = 500
    end

    methods
        function I = ShapeSpace_interp(x1, x2, t, energy_type, initFromLast)
            assert( all(size(x1) == size(x2)), 'x_source and x_target must have matching dimension' );
            
            if nargin<4, energy_type = 5; end % default to type 5: arap
            if nargin<5, initFromLast = true; end
            I.initFromLast = initFromLast;

            I.x1 = x1; I.x2 = x2; I.t = t; I.energy_type = energy_type;
            I.anchorId = reshape(t(1, 1:3), [], 1);
            I.zlast = x1;
            
            I.preprocess();
        end
        
        function preprocess(I)
            nv = size(I.x1, 1);
            nt = size(I.t, 1);

            %% preprocess            
            V2Em = sparse( [1:3 1:3] + (0:nt-1)'*3, I.t(:, [1 1 1 2:4]), repmat([-1 -1 -1 1 1 1], nt, 1), 3*nt, nv);
            invA1 = blockblas('inverse', V2Em*I.x1);
            invA2 = blockblas('inverse', V2Em*I.x2);
            toff = repmat( reshape( repmat( 0:nt-1, 3, 1 )*3, [], 1 ), 1, 3 );
            I.M1 = sparse(repmat((1:3)', nt, 3)+toff, repmat(1:3, 3*nt, 1)+toff, invA1)*V2Em;
            I.M2 = sparse(repmat((1:3)', nt, 3)+toff, repmat(1:3, 3*nt, 1)+toff, invA2)*V2Em;
            
            I.X2Js1 = invA1*[-1 1 0 0; -1 0 1 0; -1 0 0 1];  % per tet x->Jacobian, local dense version of M1
            I.X2Js2 = invA2*[-1 1 0 0; -1 0 1 0; -1 0 0 1];  % per tet x->Jacobian

            
            I.vol1 = signedVolume(I.x1, I.t);
            I.vol2 = signedVolume(I.x2, I.t);

            I.gIdx = uint64( I.t(:,[1 1 1 2 2 2 3 3 3 4 4 4])'*3 + [-2:0 -2:0 -2:0 -2:0]' );
            
            %% get sparse matrix pattern
            viPerTet = [1 1 1 2 2 2 3 3 3 4 4 4];
            [rowi, coli] = meshgrid(viPerTet, viPerTet);
            rowi = rowi(:); coli = coli(:);

            Mi = I.t(:, rowi)*3 + repmat(reshape( repmat(-2:0,12,4) , [], 1 )', nt, 1);
            Mj = I.t(:, coli)*3 + repmat(reshape( repmat(-2:0,12,4)', [], 1 )', nt, 1);

            L = laplacian3d(zeros(nv,3), I.t, 'uniform');
            I.h0 = kron(L, ones(3)); % only pattern is needed

            I.Hnzidx = ij2nzIdxs(I.h0, uint64(Mj), uint64(Mi));
            I.Hnzidx = reshape(I.Hnzidx', 12, []);
            
            %% init solver
            I.solver = pardisosolver(I.h0, 'llt');
        end
        
        function z = interp(I, wt)
            fInterp = @(x, y) x + wt*(y-x);
            tt = tic;
            z = I.zlast;
            
            P2Pxyz = I.anchorId*3 - [2 1 0];
            % zAnchors = fInterp(wt, I.x1(I.anchorId,:), I.x2(I.anchorId,:));
            zAnchors = z(I.anchorId, :);
            fEnergyP2P = @(y) norm(y(I.anchorId,:)-zAnchors,'fro')^2*I.p2p_weight;
            
            nv = size(I.x1, 1);
            z(I.anchorId,:) = zAnchors;
            
            
            %% initial grad / hessian with given p2p constraints
            g0 = zeros(nv*3, 1);
            g0(P2Pxyz) = (z(I.anchorId,:)-zAnchors)*I.p2p_weight*2;
            
            Hnonzeros0 = zeros(nnz(I.h0), 1);
            Hnonzeros0( ij2nzIdxs(I.h0, uint64(P2Pxyz), uint64(P2Pxyz)) ) = 2*I.p2p_weight;

            keepMapOrientation = (I.energy_type~=5);  % no orientation preserving (locally injective) for arap, energy_type=5

            ls_beta = 0.5;
            ls_alpha = 0.2;
            
            fMyEnergy = @(y) fInterp( isometricEnergyFromJ3Dc((I.M1*y)', [], I.vol1, I.energy_type), ...
                                      isometricEnergyFromJ3Dc((I.M2*y)', [], I.vol2, I.energy_type) ) + fEnergyP2P(y);

            for it=1:I.maxIteration
                Js1 = I.M1*z;
                Js2 = I.M2*z;
                [Eiso1, Eg1, Eh1] = isometricEnergyFromJ3Dc(Js1', I.X2Js1, I.vol1, I.energy_type);
                [Eiso2, Eg2, Eh2] = isometricEnergyFromJ3Dc(Js2', I.X2Js2, I.vol2, I.energy_type);
                E = fInterp(Eiso1, Eiso2) + fEnergyP2P(z);

                %% global g/h
                g = myaccumarray(I.gIdx, fInterp(Eg1, Eg2), g0);
                Hnonzeros = myaccumarray(I.Hnzidx, fInterp(Eh1, Eh2), Hnonzeros0);

                deltax = I.solver.refactor_solve(replaceNonzeros(I.h0, Hnonzeros), g);
                deltax = reshape(deltax, 3, [])';
                normdeltax = norm(deltax, inf);
                
                %% initial step size for line search
                ls_t = 1;
                if keepMapOrientation
                    ls_t = min(maxInjectiveStepSizeForMesh(I.t, z, z-deltax)*0.95, 1);
                    assert( all( signedVolume(z-deltax*ls_t, I.t)>=0 ) );
                end

                %% wolf condition
                dxdotg = dot( deltax(:), g );
                fQPEstim = @(t) E-ls_alpha*t*dxdotg;

                e_new = fMyEnergy( z-deltax*ls_t );
                while normdeltax*ls_t > 1e-6 && e_new > fQPEstim(ls_t)
                    ls_t = ls_t*ls_beta;
                    e_new = fMyEnergy( z-deltax*ls_t );
                end

                if normdeltax*ls_t < 1e-6
                    break;
                end

                z = z-deltax*ls_t;
            end

            fprintf('it %3d, time = %.2f, step = %.2e, energy = %.3e\n', it, toc(tt), ls_t*normdeltax, e_new);
            I.zlast = z;
        end
    end % methods
end
