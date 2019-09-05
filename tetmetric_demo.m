%% input: specify which model and which time to compare
dataname = "bar_twisted"; % folder that contains the input meshes and results
interpmethod = "SQP";  %["ARAP" "FFMP" "GE" "ABF" "SQP"];

basedir = '.';
%% load input
basedir = sprintf('%s/%s/', basedir, dataname);
[x1, tets] = readMESH( basedir + "source.mesh" );
x2 = load( basedir + "target.txt" );

%% global alignment based on anchors
anchorId = tets(1,1:3)';
fAlign = @(z) alignToAnchors(z, [anchorId x2(anchorId,:)]);

%% show iterpolation results
switch interpmethod
    case 'SQP'
        I = SQP_interp(x1, x2, tets, 1);
    case 'ARAP'
        I = arap_interp(x1, x2, tets);
    case 'GE'
        I = ShapeSpace_interp(x1, x2, tets);
    case 'FFMP'
        I = FFMP_interp(x1, x2, tets);
    otherwise
        clear I;
end

%% visualize the results
fDrawMesh = @(x, t) trimesh([t(:, 1:3); t(:, 2:4); t(:, [1 3 4]); t(:, [1 2 4])], x(:,1), x(:,2), x(:,3), 'edgecolor', 'k', 'facecolor', 'b', 'edgealpha', 0.1);
genCData = @(y) repmat(symmetricDirichlet(x1, y, tets)-6,4,1);
figure('Units','normalized','Position',[0 0 1 1]); % fullscreen figure
subplot(2,2,1); h1=fDrawMesh(x1, tets); title('source'); axis off; axis equal; set(h1, 'cdata', genCData(x1), 'facecolor', 'flat'); caxis([0 3]); 
subplot(2,2,3); h0=fDrawMesh(fAlign(x2), tets); title('target'); axis off; axis equal; set(h0, 'cdata', genCData(x2), 'facecolor', 'flat'); colormap jet; caxis([0 3]);

%% show continuous morph for t=0...1
% subplot(2,2,2); h =fDrawMesh(x1, tets); axis off; axis equal; set(h, 'cdata', genCData(x1), 'facecolor', 'flat'); colormap jet; caxis([0 3]);
% for t=0:0.05:1
%     if strcmp(interpmethod, 'ABF')
%         xInterp = ABF_interp(x1, x2, tets, t);
%     else
%         xInterp = I.interp(t);
%     end
% 
%     set(h, 'vertices', fAlign(xInterp), 'cdata', genCData(xInterp), 'facecolor', 'flat'); title(sprintf('t=%g', t)); pause(0.1);
% end


%% show 2 interpolation results
for i=1:2
    t=i/3;
    if strcmp(interpmethod, 'ABF')
        xInterp = ABF_interp(x1, x2, tets, t);
    else
        xInterp = I.interp(t);
    end

    subplot(2,2,i*2); h = fDrawMesh(x1, tets); axis off; axis equal; set(h, 'cdata', genCData(x1), 'facecolor', 'flat'); colormap jet; caxis([0 3]);
    set(h, 'vertices', fAlign(xInterp), 'cdata', genCData(xInterp), 'facecolor', 'flat'); title(sprintf('t=%g', t));
end