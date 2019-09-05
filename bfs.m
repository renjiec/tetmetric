function [path] = bfs(M, root)

numTets = size(M.ConnectivityList,1);
N = M.neighbors();

ii = repmat(1:numTets, 1, 4); %ii = repmat([1:numTets], 1, 4);
jj = reshape(N, 1,4*numTets);

kk = ~isnan(jj);

%a list of dual edges. Each dual edge is composed of two indices
%representing 2 tetrahedras

adjacencyGraph = sparse(ii(kk), jj(kk), 1, numTets, numTets);


[disc, pred] = graphtraverse(adjacencyGraph, root, 'Directed', false, 'Method', 'BFS');
next = uint32(disc(2:end));
prev = uint32(pred(next));
path = [prev; next];

end
