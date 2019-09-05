function angles = dihedralAnglesFromTetELS( mat_in )
% input: els (all edges).
% output: dihedral angles around all edges
% note: outside the function, we should extract only rows which belong to interior edges

A2 = [0,  1,  1, -2,  1,  1; 1,  0,  1,  1, -2,  1; 1,  1,  0,  1,  1, -2; -2,  1,  1,  0,  1,  1; 1, -2,  1,  1,  0,  1; 1,  1, -2,  1,  1, 0];
A3 = [0,  0,  0,  0,  0,  0; 0,  0, -1,  0,  0,  0; -1, -1,  0,  0,  0,  0; 0,  1,  1,  0, -1,  1; 1,  0,  0,  1,  0, -1; 0,  0,  0, -1,  1, 0];
A4 = [0,  1,  1,  0, -1,  1; 1,  0,  0,  1,  0, -1; 0,  0,  0, -1,  1,  0; 0,  0,  0,  0,  0,  0; 0,  0, -1,  0,  0,  0; -1, -1,  0,  0,  0, 0];
A5 = [1,  0,  0,  0,  0,  0; 0,  1,  1,  0,  0,  0; 0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  1,  0; 0,  0,  0,  1,  0,  1];   
A6 = [0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  0,  0; 1,  0,  0,  0,  0,  0; 0,  1,  1,  1,  0,  1; 0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  1,  0];
A7 = [1,  0,  0,  0,  0,  0; 0,  1,  1,  0,  0,  0; 1, -1, -1,  0,  0,  0; 0,  1,  1,  1, -1,  1; -1,  0,  0, -1,  1, -1; 0,  0,  0,  1,  1, 1];
A8 = [0,  0,  1,  0,  0,  0; 0,  0,  0,  1,  0,  0; 0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  0,  0; 0,  0,  0,  0,  1,  0; 1,  1,  0,  0,  0,  1];
A9 = [0,  0,  0,  0,  1,  0; 1,  1,  0,  0,  0,  1; 0,  0,  1,  0,  0,  0; 0,  0,  0,  1,  0,  0; 0,  0,  0,  0,  0,  0;0,  0,  0,  0,  0,  0];
A10= [-1, -1,  1,  0,  1, -1; 1,  1,  0,  1,  0,  1; 0,  0,  1, -1, -1,  0; 0,  0,  0,  1,  0,  0; 0,  0, -1,  0,  1,  0; 1,  1,  0,  0,  0,  1];

% ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
%           ******************
%           a1 b1 c1 d1 e1 f1
%           a2 b2 c2 d2 e2 f2
%                   .
% mat_in =          .
%                   .
%                   .
%           an bn cn dn en fn        
%           ******************
%          dimension: (numTetras X 6)
% ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~

% ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
% calculate the following matrix: 
%             ******************
%             dihedral angle around a1 only in tet 1,  dihedral angle around b1 only in tet 1, dihedral angle around c1 only in tet 1 .....
%             dihedral angle around a2 only in tet 2,  dihedral angle around b2 only in tet 2, dihedral angle around c2 only in tet 2 .....
%                    .
%                    .
%  mat_out =         .
%                    .
%                    .
%             dihedral angle around an only in tet n,  dihedral angle around bn only in tet n, dihedral angle around cn only in tet n
%             ******************
%          dimension: (numTetras X 6)
% ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
nominator = -(mat_in).^2 + (mat_in*A2).*(mat_in) + (mat_in*A3).*(mat_in*A4); 
denominator = ((4*(mat_in*A5).*(mat_in*A6)-(mat_in*A7).^2).*(4*(mat_in*A8).*(mat_in*A9)-(mat_in*A10).^2)).^0.5;
angles = acos(nominator./denominator);   


