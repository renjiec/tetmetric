classdef arap_interp < handle
    properties(SetAccess = protected)
        x1
        x2
        t
        anchorId
        S
        H
        Qrot
        W
    end
    
    methods
        function I = arap_interp(x1, x2, t)
            assert( all(size(x1) == size(x2)), 'x_source and x_target must have matching dimension' );
            
            I.x1 = x1; I.x2 = x2; I.t = t; 
            I.anchorId = t(1,1);

            I.preprocess();
        end
        
        function preprocess(I)
            nv = size(I.x1, 1);
            nt = size(I.t, 1);

            V2Em = sparse( [1:3 1:3] + (0:nt-1)'*3, I.t(:, [1 1 1 2:4]), repmat([-1 -1 -1 1 1 1], nt, 1), 3*nt, nv);
            invM = blockblas('inverse', V2Em*I.x1);
            J = blockblas('multiply', invM, V2Em*I.x2);
            Rot = projectJacobi2Rotation( J );
            I.S = blockblas('transpose_A_and_multiply', Rot, J);

            vol = signedVolume(I.x1, I.t);
            I.W = sparse(1:nt*3, 1:nt*3, reshape(repmat(vol,1,3)',1,[]), nt*3, nt*3);

            toff = repmat( reshape( repmat( 0:nt-1, 3, 1 )*3, [], 1 ), 1, 3 );
            I.H = I.W*sparse(repmat((1:3)', nt, 3)+toff, repmat(1:3, 3*nt, 1)+toff, invM)*V2Em;

            
            I.Qrot = rotation2quaternion(Rot);
        end
        
        function z = interp(I, wt)
            nv = size(I.x1, 1);
            nt = size(I.t, 1);

            Rint = quaternion2rotation( quaternion_slerp([1 0 0 0], I.Qrot, wt), 1 );
            Sint = repmat(eye(3), nt, 1)*(1-wt) + I.S*wt;
            Aint = blockblas('multiply', Rint, Sint);

            rhs = I.W*Aint;
            z(I.anchorId,:) = (1-wt)*I.x1(I.anchorId,:) + wt*I.x2(I.anchorId,:);
            
            Iset = setdiff(1:nv, I.anchorId);
            z(Iset,:) = I.H(:, Iset)\(rhs-I.H(:,I.anchorId)*z(I.anchorId,:));
        end
    end % methods
end