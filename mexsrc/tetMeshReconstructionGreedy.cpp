#include "mex.h"
#include <vector>
#include <Eigen/Dense>

/* The computational routine */
void tetMeshReconstruction(const double* pTets, const double* pEdgeIdxsPerTet, const double* els, const unsigned* path, size_t nt, size_t nv, double* new_points, size_t size_path)
{
	Eigen::Map<Eigen::MatrixXd> verticesXd(new_points, nv, 3);
	Eigen::Map<const Eigen::MatrixXd> tets(pTets, nt, 4);
	Eigen::Map<const Eigen::MatrixXd> edgesPerTet(pEdgeIdxsPerTet, nt, 6);
	
	int root = (int)path[0] - 1; //due to matlab indexing
    // reconstruct first tet
    int ind_edge_a = (int)edgesPerTet(root, 0)-1;
    int ind_edge_b = (int)edgesPerTet(root, 1)-1;
    int ind_edge_c = (int)edgesPerTet(root, 2)-1;
    int ind_edge_d = (int)edgesPerTet(root, 3)-1;
    int ind_edge_e = (int)edgesPerTet(root, 4)-1;
	int ind_edge_f = (int)edgesPerTet(root, 5)-1;

    double e1 = els[ind_edge_f];
    double e2 = els[ind_edge_e];
    double e3 = els[ind_edge_a];
    double e4 = els[ind_edge_d];
    double e5 = els[ind_edge_b];
    double e6 = els[ind_edge_c];
    
    double p2x = sqrt(e1);
    double p3x = (e1 + e2 - e4)/(2*sqrt(e1));
    double p3y = sqrt(e2 - p3x*p3x);
    double p4x = (e1 + e3- e5)/(2*sqrt(e1));
    double p4y = (e3 - e6 - 2*p3x*p4x + p3x*p3x + p3y*p3y)/(2*p3y);
    double p4z = sqrt(e3 - p4x*p4x - p4y*p4y);
    
    int ind_p1 = (int)tets(root, 0)-1;
    int ind_p2 = (int)tets(root, 1)-1;
    int ind_p3 = (int)tets(root, 2)-1;
    int ind_p4 = (int)tets(root, 3)-1;
    
	verticesXd(ind_p2,0) = p2x;
	verticesXd(ind_p3,0) = p3x;
	verticesXd(ind_p3,1) = p3y;
	verticesXd(ind_p4,0) = p4x;
	verticesXd(ind_p4,1) = p4y;
	verticesXd(ind_p4,2) = p4z;

    std::vector<bool> vertex_embeded(nv, false);
    vertex_embeded[ind_p1] = true;
    vertex_embeded[ind_p2] = true;
    vertex_embeded[ind_p3] = true;
    vertex_embeded[ind_p4] = true;
    
	/* reconstruct other tets */
	for (int i = 0; i<size_path; i++)
    {
		int ct_ind = (int)path[1+2*i] - 1; //current tet index
		int pt_ind = (int)path[2*i] - 1; //previous tet index
        
		//Check if current tet vertices already computed
		ind_p1 = (int)tets(ct_ind, 0) - 1;
		ind_p2 = (int)tets(ct_ind, 1) - 1;
		ind_p3 = (int)tets(ct_ind, 2) - 1;
		ind_p4 = (int)tets(ct_ind, 3) - 1;
		if (vertex_embeded[ind_p1] && vertex_embeded[ind_p2] && vertex_embeded[ind_p3] && vertex_embeded[ind_p4])
			continue; //all vertices already computed. Skip to next tet
			
		ind_edge_f = (int)edgesPerTet(ct_ind, 5) - 1;
		ind_edge_e = (int)edgesPerTet(ct_ind, 4) - 1;
		ind_edge_a = (int)edgesPerTet(ct_ind, 0) - 1;
		ind_edge_d = (int)edgesPerTet(ct_ind, 3) - 1;
		ind_edge_b = (int)edgesPerTet(ct_ind, 1) - 1;
		ind_edge_c = (int)edgesPerTet(ct_ind, 2) - 1;

		e1 = els[ind_edge_f];
		e2 = els[ind_edge_e];
		e3 = els[ind_edge_a];
		e4 = els[ind_edge_d];
		e5 = els[ind_edge_b];
		e6 = els[ind_edge_c];

        // calc the location of the 'fourth' vertex
		int location = -1; 
        int j=0;
        while (location == -1)//run over current tet
        {
            int k=0; 
            while (k<4) //run over previous tet
            {
                if (tets(ct_ind, j) == tets(pt_ind, k))
                    break;

                if (k==3)
                    location = j;

                k++;
            }
            j++;
        }

		int order_tet[4] = { 0 };
		switch (location)
		{
			case 0:
				order_tet[0] = ind_p4;
				order_tet[1] = ind_p3;
				order_tet[2] = ind_p2;
				order_tet[3] = ind_p1;

				p2x = sqrt(e6);
				p3x = (e6 + e5 - e4) / (2 * sqrt(e6));
				p3y = sqrt(e5 - p3x*p3x);
				p4x = (e6 + e3 - e2) / (2 * sqrt(e6));
				p4y = (e3 - e1 - 2 * p3x*p4x + p3x*p3x + p3y*p3y) / (2 * p3y);
				p4z = sqrt(e3 - p4x*p4x - p4y*p4y);

				break;

			case 1:
				order_tet[0] = ind_p3;
				order_tet[1] = ind_p4;
				order_tet[2] = ind_p1;
				order_tet[3] = ind_p2;

				p2x = sqrt(e6);
				p3x = (e6 + e2 - e3) / (2 * sqrt(e6));
				p3y = sqrt(e2 - p3x*p3x);
				p4x = (e6 + e4 - e5) / (2 * sqrt(e6));
				p4y = (e4 - e1 - 2 * p3x*p4x + p3x*p3x + p3y*p3y) / (2 * p3y);
				p4z = sqrt(e4 - p4x*p4x - p4y*p4y);

				break;

			case 2:
				order_tet[0] = ind_p2;
				order_tet[1] = ind_p1;
				order_tet[2] = ind_p4;
				order_tet[3] = ind_p3;

				p2x = sqrt(e1);
				p3x = (e1 + e5 - e3) / (2 * sqrt(e1));
				p3y = sqrt(e5 - p3x*p3x);
				p4x = (e1 + e4 - e2) / (2 * sqrt(e1));
				p4y = (e4 - e6 - 2 * p3x*p4x + p3x*p3x + p3y*p3y) / (2 * p3y);
				p4z = sqrt(e4 - p4x*p4x - p4y*p4y);

				break;

			case 3:
				order_tet[0] = ind_p1;
				order_tet[1] = ind_p2;
				order_tet[2] = ind_p3;
				order_tet[3] = ind_p4;

				p2x = sqrt(e1);
				p3x = (e1 + e2 - e4) / (2 * sqrt(e1));
				p3y = sqrt(e2 - p3x*p3x);
				p4x = (e1 + e3 - e5) / (2 * sqrt(e1));
				p4y = (e3 - e6 - 2 * p3x*p4x + p3x*p3x + p3y*p3y) / (2 * p3y);
				p4z = sqrt(e3 - p4x*p4x - p4y*p4y);

				break;
		} //end construct the new tet raound the origin
		
		ind_p1 = order_tet[0];
		ind_p2 = order_tet[1];
		ind_p3 = order_tet[2];
		ind_p4 = order_tet[3];

		//rotate the new tet
		Eigen::Vector3d u = verticesXd.row(ind_p2) - verticesXd.row(ind_p1);
		Eigen::Vector3d v = verticesXd.row(ind_p3) - verticesXd.row(ind_p1);

		u.normalize();
		Eigen::Vector3d n = u.cross(v).normalized();
		v = n.cross(u);

		Eigen::Matrix3d Rotation;
		Rotation << u, v, n;

		auto translation = verticesXd.row(ind_p1);
		verticesXd.row(ind_p4) = Rotation*Eigen::Vector3d(p4x, p4y, p4z) + translation.transpose();
		vertex_embeded[ind_p4] = true;
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs<5)
        mexErrMsgTxt("Invalid input: not enough input, x = tetMeshReconstructionGreedy(tets, edgeIdxsPerTet, els, path, numVertices);");
    
    const double* tets = mxGetPr(prhs[0]);
    const double* edgeIdxsPerTet = mxGetPr(prhs[1]);
    const double* els = mxGetPr(prhs[2]);
    const unsigned* path = (unsigned*)mxGetPr(prhs[3]);
	
    size_t nv = (size_t)mxGetScalar(prhs[4]);
    size_t nt = mxGetM(prhs[0]);
	size_t size_path = mxGetN(prhs[3]);
    
    /* create output matrices */
    plhs[0] = mxCreateDoubleMatrix(nv, 3, mxREAL);
    
    /* call the computational routine */
    tetMeshReconstruction(tets, edgeIdxsPerTet, els, path, nt, nv, mxGetPr(plhs[0]), size_path);
}