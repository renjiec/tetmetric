function [V, T] = readMESH( filename )
% readMESH reads an MESH file with vertex/face/tet information
%
% [V,T] = readMESH( filename )
%
% Input:
%  filename  path to .mesh file
% Outputs:
%  V  #V by 3 list of vertices
%  T  #T by 4 list of tet indices

[fp, errmsg] = fopen(filename, 'r');
if fp == -1, error(errmsg); end
cleanupObj = onCleanup(@()fclose(fp));

% First line is mandatory header
MESHheader = eat_comments(fp,'#');
assert( strcmp(MESHheader(1:end-1),'MeshVersionFormatted '), sprintf('First line should be "MeshVersionFormatted " not ("%s")...',MESHheader) ); %todo: works for version<=2, may need update for version >2

% force read line feed
fscanf(fp,'\n');
Dimension3 = eat_comments(fp,'#');

% second line is mandatory Dimension 3
if ~strcmp(Dimension3,'Dimension 3')
    % tetgen likes to put the 3 on the next line, try to append next word hoping its a 3 force read line feed
    Dimension3 = [Dimension3 ' ' eat_comments(fp,'#')];
    assert( strcmp(Dimension3,'Dimension 3'), 'Second line should be "Dimension 3"...' );
end

%%
Vertices = eat_comments(fp,'#');
% thrid line is mandatory Vertices
assert( contains(Vertices,'Vertices'), 'Third line should be "Vertices"...' );

% read vertex count
num_vertices = fscanf(fp,'%d\n',1);

% figureout if there is extra dimention of 0
fpos = ftell(fp);
edim = numel( sscanf(fgetl(fp), '%f') );
fseek(fp, fpos, 'bof');
assert(edim>=3);

% read num_vertices many sets of vertex coordinates (x,y,z,ref)
V = fscanf(fp,'%g',num_vertices*edim);
V = reshape(V,edim,num_vertices)';
V = V(:, 1:3);

%%
Triangles = eat_comments(fp,'#');
% forth non numeric is mandatory Triangles
if strcmp(Triangles, 'Triangles')
    % read triangle count
    num_triangles = fscanf(fp,'%d\n',1);
    % read num_triangles many sets of face indices (a,b,c,ref)
    
    if num_triangles>0
        fpos = ftell(fp);
        edim = numel( sscanf(fgetl(fp), '%f') );
        fseek(fp, fpos, 'bof');
        assert(edim>=3);
        F = fscanf(fp,'%d',edim*num_triangles);
        F = reshape(F, edim, num_triangles)';
        F = F(:,1:3);
    end
end

if ~strcmp(Triangles,'Tetrahedra')
    Tetrahedra = eat_comments(fp, '#');
else
    Tetrahedra = Triangles;
end

%%
% forth non numeric is mandatory Tetrahedra
assert( contains(Tetrahedra,'Tetrahedra'), 'Fifth (non-number) line should be "Tetrahedra"...');

% read tetrahedra count
num_tetrahedra = fscanf(fp,'%d\n',1);
fpos = ftell(fp);
edim = numel( sscanf(fgetl(fp), '%f') );
fseek(fp, fpos, 'bof');
assert(edim>=4);

% read num_tetrahedra many sets of tet indices (a,b,c,d,ref)
T = fscanf(fp,'%d',edim*num_tetrahedra);
T = reshape(T,edim,num_tetrahedra)';
T = T(:, 1:4);

if min(T(:)) == 0, T = T+1; end
end


function line = eat_comments(file_pointer,comment_symbol)
% EAT_COMMENTS use fscanf to eat lines starting with a comment_symbol
% 
% line = eat_comments(file_pointer,comment_symbol)
%
% Inputs:
%   file_pointer  returned from fopen
%   comment_symbol  symbol of comment, e.g. '#'
% Output:
%   line  next line that does not start with a comment_symbol 
%     (file_pointer is correspondingly adjusted)
%
assert(size(comment_symbol,2) == 1);

while(true)
    % read next whole line
    line = fscanf(file_pointer,' %[^\r\n]s');
    if(size(line,2) == 0)
        break;
    elseif(line(1,1) ~= comment_symbol)
        break;
    end
end
end
