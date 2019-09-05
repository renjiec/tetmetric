#undef free
#include <OGF/cells/cgraph/cgraph_builder.h>
#include <OGF/VolumeABF/commands/whet_grid_volume_abf_commands.h>
#include <OGF/VolumeABF/algos/shape_interpolator_ffmp.h>
#include <OGF/VolumeABF/algos/flattener_angles.h>
#include <OGF/VolumeABF/algos/reconstructor_angles_greedy.h>
#include <OGF/VolumeABF/algos/reconstructor_angles_least_squares.h>
#include <OGF/VolumeABF/algos/reconstructor_angles_eigen.h>
#include <OGF/VolumeABF/algos/shape_interpolator_ffmp.h>
#include <algorithm>
#include <Eigen/Core>


void get_tetrahedron_dihedral_angles( const OGF::vec3& v0, const OGF::vec3& v1, const OGF::vec3& v2, const OGF::vec3& v3,
                                      double& a01, double& a02, double& a03, double& a12, double& a13, double& a23 )
{
    OGF::vec3 n0 = OGF::cross( v3 - v1, v2 - v1 );
    OGF::vec3 n1 = OGF::cross( v2 - v0, v3 - v0 );
    OGF::vec3 n2 = OGF::cross( v3 - v0, v1 - v0 );
    OGF::vec3 n3 = OGF::cross( v1 - v0, v2 - v0 );
    double V6 = OGF::dot( v1 - v0, OGF::cross( v2 - v0, v3 - v0 ) );

    a01 = ::atan2( V6 * OGF::length( v2 - v3 ), -OGF::dot( n0, n1 ) );
    a02 = ::atan2( V6 * OGF::length( v1 - v3 ), -OGF::dot( n0, n2 ) );
    a03 = ::atan2( V6 * OGF::length( v1 - v2 ), -OGF::dot( n0, n3 ) );
    a12 = ::atan2( V6 * OGF::length( v0 - v3 ), -OGF::dot( n1, n2 ) );
    a13 = ::atan2( V6 * OGF::length( v0 - v2 ), -OGF::dot( n1, n3 ) );
    a23 = ::atan2( V6 * OGF::length( v0 - v1 ), -OGF::dot( n2, n3 ) );

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923   // pi/2
#endif
    if ( a01 < -M_PI_2 )
        a01 += 2.0 * M_PI;

    if ( a02 < -M_PI_2 )
        a02 += 2.0 * M_PI;

    if ( a03 < -M_PI_2 )
        a03 += 2.0 * M_PI;

    if ( a12 < -M_PI_2 )
        a12 += 2.0 * M_PI;

    if ( a13 < -M_PI_2 )
        a13 += 2.0 * M_PI;

    if ( a23 < -M_PI_2 )
        a23 += 2.0 * M_PI;
}

class graphite_tetmesh :public OGF::CGraphMutator
{
public:
    graphite_tetmesh() :OGF::CGraphMutator(nullptr) {}

    //CGraph* build_graphite_tetmesh(const double *x, int nv, const int *tets, int nt, const int *tetneighbors)
    template<class mat, class imat>
    OGF::CGraph* build_graphite_tetmesh(mat&& x, imat &&tets, imat &&tetneighbors)
    {
        assert(tets.rows() == tetneighbors.rows());
        size_t nv = x.rows();
        size_t nt = tets.rows();

        OGF::CGraphMutator::set_target(new OGF::CGraph);
        if (target()->size_of_meta_cells() == 0) {
            OGF::CGraphBuilder builder;
            builder.set_target(target());
            builder.begin_volume();
            builder.build_meta_tetrahedron();
            builder.end_volume();
        }

        // Create vertices
        std::vector<OGF::CGraph::Vertex*> vertices;
        for (int i = 0; i < nv; i++) 
            vertices.push_back(new_vertex(OGF::vec3(x(i, 0), x(i, 1), x(i, 2))));

        // Create cells
        std::vector<OGF::CGraph::Cell*> cells;
        OGF::CGraph::MetaCell* meta_tet = target()->meta_cell(0);
        for (int i = 0; i < nt; i++) {
            OGF::CGraph::Cell* c = new_cell(meta_tet);
            for (int j = 0; j < 4; j++) {
                set_cell_vertex(c, j, vertices[tets(i, j)]);
            }
            cells.push_back(c);
        }

        // Create adjacency
        for (int i = 0; i < nt; i++) {
            for (int j = 0; j < 4; j++) {
                const int offset = 0;
                int adjacent = tetneighbors(i, j)-offset;
                if (adjacent >= 0) {
                    set_cell_adjacent(cells[i], j, cells[adjacent]);
                }
            }
        }
        return target();
    }
};

template<class matc, class imat, class mat>
void tetInterpFFMP(matc &&x, matc &&y, imat &&tets, imat &&tetneighbors, mat &&z, double t)
{
    graphite_tetmesh m1;
    auto mesh1 = m1.build_graphite_tetmesh(x, tets, tetneighbors);
    auto mesh2 = m1.build_graphite_tetmesh(y, tets, tetneighbors);

    auto result = mesh1;
    OGF::ShapeInterpolatorFFMP(mesh1, mesh2, result, t).interpolate();

    int i = 0;
    for (auto it = result->vertices_begin(); it != result->vertices_end(); it++, i++) {
        z.row(i) << it->point().x, it->point().y, it->point().z;
    }

    //fprintf(stdout, "### after interpolation: \n");
    //i = 0;
    //for (auto it = result->vertices_begin(); it != result->vertices_end(); it++, i++)
    //    fprintf(stdout, "v %d: %f, %f, %f\n", i, it->point().x, it->point().y, it->point().z);

    //i = 0;
    //std::map<OGF::CGraph::Vertex*, int> v2i;
    //for (auto it = result->vertices_begin(); it != result->vertices_end(); it++, i++)
    //    v2i[it] = i;

    //i = 0;
    //for (auto it = result->cells_begin(); it != result->cells_end(); it++, i++)
    //    fprintf(stdout, "f %d: %d, %d, %d, %d\n", i, v2i[it->vertex(0)], v2i[it->vertex(1)], v2i[it->vertex(2)], v2i[it->vertex(3)]);

    delete mesh1, mesh2;
}

template<class mat6c, class matc, class imat, class mat>
void abfflatten(mat6c &&diangles, matc &&x, imat &&tets, imat &&tetneighbors, mat &&z, 
    double min_angle, double regularization, int nb_outer_iterations,
    int nb_inner_iterations, double threshold, OGF::ReconstructionMethod reconstruction_method)
{
    assert(z.rows() == x.rows());
    graphite_tetmesh m1;
    auto result = m1.build_graphite_tetmesh(x, tets, tetneighbors);
    OGF::CGraphCellAttribute<std::vector<double> > dihedral_angles(result);

    int j = 0;
    for (auto itr = result->cells_begin() ; itr != result->cells_end() ; itr++, j++ ) {
        dihedral_angles[itr].resize(6);
        for (int i = 0; i < 6; ++i)
            dihedral_angles[itr][i] = diangles(j , i);
    }

    auto edges = new OGF::CGraphEdges( result );
    std::vector<double> target_curvatures( edges->nb_edges(), 0 );

    for (size_t i = 0; i < edges->nb_edges(); ++i)
        if ( !edges->is_boundary( i ) )
            target_curvatures[i] = 2 * M_PI;

    OGF::ProgressLogger progress_logger;
    OGF::FlattenerAngles( result, *edges, target_curvatures, dihedral_angles, min_angle, regularization,
                     nb_outer_iterations, nb_inner_iterations, threshold, progress_logger ).flatten();

    switch( reconstruction_method ) {
        case OGF::Greedy: OGF::ReconstructorAnglesGreedy( result, dihedral_angles ).reconstruct(); break;
        case OGF::LeastSquares: OGF::ReconstructorAnglesLeastSquares( result, dihedral_angles, false ).reconstruct(); break;
        case OGF::Eigen: OGF::ReconstructorAnglesEigen( result, dihedral_angles ).reconstruct(); break;
    }

    dihedral_angles.unbind();

    int i = 0;
    for (auto it = result->vertices_begin(); it != result->vertices_end(); it++, i++) {
        z.row(i) << it->point().x, it->point().y, it->point().z;
    }

    delete edges;
    delete result;
}

template<class matc, class imat, class mat>
void tetInterpABF(matc &&x, matc &&y, imat &&tets, imat &&tetneighbors, mat &&z, double tt,
    double min_angle, double regularization, int nb_outer_iterations,
    int nb_inner_iterations, double threshold, OGF::ReconstructionMethod reconstruction_method)
{
    assert(z.rows() == x.rows());
    graphite_tetmesh m1;
    auto mesh1 = m1.build_graphite_tetmesh(x, tets, tetneighbors);
    auto mesh2 = m1.build_graphite_tetmesh(y, tets, tetneighbors);
    
    OGF::CGraph *sources[] = { mesh1, mesh2 };

    OGF::CGraph* result = sources[0];
    OGF::CGraphCellAttribute<std::vector<double> > dihedral_angles(result);

    int nb_sources = 2;
    std::vector<OGF::CGraph::Cell_iterator> its(nb_sources);

    for (int i = 0; i < nb_sources; ++i)
        its[i] = sources[i]->cells_begin();

    OGF::vec3 t(1 - tt, tt, 0);
    for (auto itr = result->cells_begin() ; itr != result->cells_end() ; itr++ ) {
        std::vector<double> angle_source[6];

        for (int i = 0; i < nb_sources; ++i) {
            angle_source[i].resize( 6 );
            get_tetrahedron_dihedral_angles( its[i]->vertex(0)->point(), its[i]->vertex(1)->point(),
                its[i]->vertex(2)->point(), its[i]->vertex(3)->point(),
                angle_source[i][0], angle_source[i][1], angle_source[i][2],
                angle_source[i][3], angle_source[i][4], angle_source[i][5]);
        }

        dihedral_angles[itr].resize( 6, 0 );

        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < nb_sources; ++j)
                dihedral_angles[itr][i] += t[j] * angle_source[j][i];

        for (int i = 0; i < nb_sources; ++i)
            its[i]++;
    }

    //int j = 0;
    //for (auto itr = result->cells_begin(); itr != result->cells_end(); itr++, j++) {
    //    printf("\nRow %d: ", j);
    //    for (int i = 0; i < 6; i++)
    //        printf("%f, ", dihedral_angles[itr][i]);
    //}

    auto edges = new OGF::CGraphEdges( result );
    std::vector<double> target_curvatures( edges->nb_edges(), 0 );

    for (unsigned i = 0; i < edges->nb_edges(); ++i)
        if ( !edges->is_boundary( i ) )
            target_curvatures[i] = 2 * M_PI;

    OGF::ProgressLogger progress_logger;
    OGF::FlattenerAngles( result, *edges, target_curvatures, dihedral_angles, min_angle, regularization,
                     nb_outer_iterations, nb_inner_iterations, threshold, progress_logger ).flatten();

    switch( reconstruction_method )
    {
        case OGF::Greedy: OGF::ReconstructorAnglesGreedy( result, dihedral_angles ).reconstruct(); break;
        case OGF::LeastSquares: OGF::ReconstructorAnglesLeastSquares( result, dihedral_angles, false ).reconstruct(); break;
        case OGF::Eigen: OGF::ReconstructorAnglesEigen( result, dihedral_angles ).reconstruct(); break;
    }

    dihedral_angles.unbind();

    int i = 0;
    for (auto it = result->vertices_begin(); it != result->vertices_end(); it++, i++) {
        z.row(i) << it->point().x, it->point().y, it->point().z;
    }

    delete edges;
    delete mesh1, mesh2;
}

