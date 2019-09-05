classdef FFMP_interp < handle   %InterpolartorMetric
    properties(SetAccess = protected)
        x1
        x2
        t
        anchorId
        G
        GtG
        tfi
        vfi
        Qti
        QQs
        QQt
        Rs
        Ss
        Rt
        St
        H0
        useHardConstraints = false;
    end
    
    methods
        function I = FFMP_interp(x1, x2, t)
            assert( all(size(x1) == size(x2)), 'x_source and x_target must have matching dimension' );
            
            I.x1 = x1; I.x2 = x2; I.t = t; 
            I.anchorId = t(1,1);

            I.preprocess();
        end
        
        function preprocess(I)
            nv = size(I.x1, 1);
            nt = size(I.t, 1);

            % neighboring tet indices
            tr = triangulation(I.t, I.x1);
            I.Qti = reshape( [ repmat(1:nt, 4, 1)'  tr.neighbors ], [], 2 );
            I.Qti = unique( sort( I.Qti( isfinite(I.Qti(:,2)), : ), 2 ), 'rows' );

            %%
            I.G = sparse( repmat(1:nt,6,1)'*3+[-2 -2 -1 -1 0 0], I.t(:, [2 1 3 1 4 1]), repmat([1 -1], nt, 3), nt*3, nv);
            I.GtG = I.G'*I.G;
            Dit = @(x, i) I.G( reshape(reshape(i,1,[])*3+(-2:0)',[],1), : )*x;

            % by defintion from the paper, the analysis, connection are all about *transposed* frames, i.e. affs and afft
            affs = blockblas('transpose', Dit(I.x1, 1:nt));
            afft = blockblas('transpose', Dit(I.x2, 1:nt));

            I.Rs = projectJacobi2Rotation(affs);
            I.Ss = blockblas('solve', I.Rs, affs);
            I.Rt = projectJacobi2Rotation(afft);
            I.St = blockblas('solve', I.Rt, afft);


            I.QQs = rotation2quaternion(  blockblas('transpose_A_and_multiply', fBlocks( I.Rs, I.Qti(:,2) ), fBlocks( I.Rs, I.Qti(:,1) ) ) );
            I.QQt = rotation2quaternion(  blockblas('transpose_A_and_multiply', fBlocks( I.Rt, I.Qti(:,2) ), fBlocks( I.Rt, I.Qti(:,1) ) ) );

            I.vfi = I.anchorId;
            [I.tfi, ~] = find(I.t==I.anchorId(1), 1);

            I.GtG(I.vfi, :) = sparse(1:numel(I.vfi), I.vfi, 1, numel(I.vfi), nv);
            nq = size(I.Qti, 1);
            I.H0 = sparse( (1:nq)'*3+(-2:0), I.Qti(:,1)*3+(-2:0), -1, nq*3, nt*3 );
        end
        
        function z = interp(I, wt)
            lambda = 1e5;

            fInterp = @(wt, x, y) (1-wt)*x+wt*y;
            fInterpQuat2Rot = @(wt, Q1, Q2) quaternion2rotation( quaternion_slerp(Q1, Q2, wt), 1 );
            fInterpRotation = @(wt, R1, R2) fInterpQuat2Rot( wt, rotation2quaternion(R1), rotation2quaternion(R2) );
            
            Rint = fInterpQuat2Rot(wt, I.QQs, I.QQt);

            Sint = fInterp(wt, I.Ss, I.St);
            S1int = fBlocks(Sint, I.Qti(:,1));
            S2int = fBlocks(Sint, I.Qti(:,2));

            Q12 = blockblas('multiply', blockblas('solve', S2int, Rint), S1int);

            nq = size(I.Qti, 1);
            nt = size(I.t, 1);

            H = I.H0 + sparse( (1:nq)'*3+repmat(-2:0, 1, 3), I.Qti(:,2)*3+[-2 -2 -2 -1 -1 -1 0 0 0], reshape(Q12',9,[])', nq*3, nt*3 );

        %     norm( H*fBlockFun(affs, @transpose), 'fro' )

            H = H'*H;

            Dfix = fInterpRotation(wt, fBlocks(I.Rs, I.tfi), fBlocks(I.Rt, I.tfi))*fInterp(wt, fBlocks(I.Ss, I.tfi), fBlocks(I.St, I.tfi));


            rhs = zeros(nt*3, 3);
            if I.useHardConstraints
                H(I.tfi*3+(-2:0),:) = sparse(1:3, I.tfi*3+(-2:0), 1, 3, nt*3);
                rhs(I.tfi*3+(-2:0),:) = Dfix';
            else % soft constraints
                H(I.tfi*3+(-2:0), I.tfi*3+(-2:0)) = H(I.tfi*3+(-2:0), I.tfi*3+(-2:0)) + lambda*speye(3);
                rhs(I.tfi*3+(-2:0),:) = lambda * Dfix';
            end

            Dt = H\rhs;

            %%
            Dt = I.G'*Dt;
            Dt(I.vfi, :) = fInterp(wt, I.x1(I.vfi,:), I.x2(I.vfi,:));
            z = I.GtG \ Dt;
        end
    end % methods
end

function a = fBlocks(m, ib) 
    a = m( reshape(ib,1,[])*3+(-2:0)', : );
end

