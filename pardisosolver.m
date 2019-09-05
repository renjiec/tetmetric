classdef pardisosolver < handle
    properties(SetAccess = protected)
        info = [];
        A = [];  % matrix, nonzero pattern
        verbose = false;
    end
    
    methods(Static)
        function i = mtype(solvername)
            switch lower(solvername)
                case {'llt'}
                    i = 2;
                case {'ldlt'}
                    i = -2;
                case {'lu'}
                    i = 11;
                otherwise
                    error('unknown matrix type %s', solvername);
            end
        end
        
        function print(varargin)
            verbose = false;
            if verbose, fprintf(varargin{:}); end
        end
    end
    
    
    methods
        function s = pardisosolver(A, solvername)
            tic
            info = pardiso_imp('init', pardisosolver.mtype(solvername), 0);
            s.info = pardiso_imp('reorder', tril(A), info, s.verbose);
            s.A = A;
            pardisosolver.print('\npardiso    symf   = %fs', toc);
        end

        function refactorize(s, A)
            tic
            s.A = A;
            s.info = pardiso_imp('factor', tril(A), s.info, s.verbose);
%             assert(f, 'linear solver failed with factorization');
            pardisosolver.print('\npardiso    numf   = %fs', toc);
        end
        
        function x = solve(s, b)
            tic
            [x, s.info] = pardiso_imp('solve', tril(s.A), b, s.info, s.verbose);
            pardisosolver.print('\npardiso    solve  = %fs', toc);
        end
        
        function x = mldivide(s, b)
            tic
            [x, s.info] = pardiso_imp('solve', tril(s.A), b, s.info, s.verbose);
            pardisosolver.print('\npardiso    solve  = %fs', toc);
        end

        function x = refactor_solve(s, A, b)
            s.refactorize(A);
            x = s.solve(b);
        end

        function delete(s)
            if ~isempty(s.info)
                tic
                pardiso_imp('free', s.info);
                pardisosolver.print('\npardiso    clean up   = %fs\n', toc);
            end
        end
    end
end
