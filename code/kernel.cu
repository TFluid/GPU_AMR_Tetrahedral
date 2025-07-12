#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <device_functions.h>// Utilities and system includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

const int threadnum = 128;

__global__ void check_AMR_face(int *d_if_Face_refinement, const int *d_if_Cell_refinement, const int *d_owner, const int *d_neighbor, int N_face, int N_inner_face)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_face)
    {
        int owner_ID = d_owner[i];
        if (i < N_inner_face)
        {
            int neighbor_ID = d_neighbor[i];
            d_if_Face_refinement[i] = (d_if_Cell_refinement[owner_ID] == 1 || d_if_Cell_refinement[neighbor_ID] == 1) ? 1 : 0;
        }
        else
        {
            d_if_Face_refinement[i] = (d_if_Cell_refinement[owner_ID] == 1) ? 1 : 0;
        }
    }
}

__device__ double3 d_add(double3 a, double3 b)
{
    return double3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ double3 d_divide(double3 a, double b)
{
    return double3{ a.x / b, a.y / b, a.z / b };
}

__global__ void update_AMR_point_vertex(int4* AMR_vertex, double3* AMR_point, const double3* point, int *d_if_cell_refinement, const int4 *vertex, int N_cell, int N_point)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_cell)
    {
        if (d_if_cell_refinement[i] > 0)
        {
            AMR_point[N_point] = d_divide(d_add(point[vertex[i].x], point[vertex[i].y]), 2.0);
            AMR_point[N_point + 1] = d_divide(d_add(point[vertex[i].x], point[vertex[i].z]), 2.0);
            AMR_point[N_point + 2] = d_divide(d_add(point[vertex[i].x], point[vertex[i].w]), 2.0);
            AMR_point[N_point + 3] = d_divide(d_add(point[vertex[i].y], point[vertex[i].z]), 2.0);
            AMR_point[N_point + 4] = d_divide(d_add(point[vertex[i].y], point[vertex[i].w]), 2.0);
            AMR_point[N_point + 5] = d_divide(d_add(point[vertex[i].z], point[vertex[i].w]), 2.0);
            AMR_vertex[i] = int4{ vertex[i].x, N_point, N_point + 1, N_point + 2 };
            AMR_vertex[N_cell] = int4{ N_point, vertex[i].y, N_point + 3, N_point + 4 };
            AMR_vertex[N_cell + 1] = int4{ N_point + 1, N_point + 3, vertex[i].z, N_point + 5 };
            AMR_vertex[N_cell + 2] = int4{ N_point + 2, N_point + 4, N_point + 5, vertex[i].w };
            AMR_vertex[N_cell + 3] = int4{ N_point, N_point + 3, N_point + 1, N_point + 4 };
            AMR_vertex[N_cell + 4] = int4{ N_point, N_point + 4, N_point + 1, N_point + 2 };
            AMR_vertex[N_cell + 5] = int4{ N_point + 1, N_point + 4, N_point + 5, N_point + 2 };
            AMR_vertex[N_cell + 6] = int4{ N_point + 1, N_point + 3, N_point + 5, N_point + 4 };
        }
    }
}

__device__ int d_face_type(const int3 face_index, const int4 cell_index)
{
    int f_sum = face_index.x + face_index.y + face_index.z;
    int c_sum_0 = cell_index.x + cell_index.y + cell_index.z;
    int c_sum_1 = cell_index.x + cell_index.y + cell_index.w;
    int c_sum_2 = cell_index.x + cell_index.z + cell_index.w;
    int c_sum_3 = cell_index.y + cell_index.z + cell_index.w;
    if (f_sum == c_sum_0) { return 0; }
    if (f_sum == c_sum_1) { return 1; }
    if (f_sum == c_sum_2) { return 2; }
    if (f_sum == c_sum_3) { return 3; }
}

__device__ int4 d_cell_index(const int face_type, const int cell_last, const int cell_ID)
{
    int4 cell_index;
    if (face_type == 0)
    {
        cell_index.x = cell_last;
        cell_index.y = cell_last + 1;
        cell_index.z = cell_last + 3;
        cell_index.w = cell_ID;
    }
    if (face_type == 1)
    {
        cell_index.x = cell_last;
        cell_index.y = cell_last + 2;
        cell_index.z = cell_last + 4;
        cell_index.w = cell_ID;
    }
    if (face_type == 2)
    {
        cell_index.x = cell_last + 1;
        cell_index.y = cell_last + 2;
        cell_index.z = cell_last + 5;
        cell_index.w = cell_ID;
    }
    if (face_type == 3)
    {
        cell_index.x = cell_last + 1;
        cell_index.y = cell_last + 2;
        cell_index.z = cell_last + 6;
        cell_index.w = cell_last;
    }
    return cell_index;
}

__device__ int4 d_map_face_nei_to_own(const int face_type_own, const int face_type_nei, const int4 vertex_own, const int4 vertex_nei, const int N_AMR_face_inner_last, const int i)
{
    int3 point_index_own;
    int3 point_index_nei;
    int4 face_nei_to_own;
    face_nei_to_own.z = N_AMR_face_inner_last + 2;
    if (face_type_own == 0)
    {
        point_index_own.x = vertex_own.y;
        point_index_own.y = vertex_own.z;
        point_index_own.z = vertex_own.x;
        point_index_nei.x = vertex_nei.y;
        point_index_nei.y = vertex_nei.z;
        point_index_nei.z = vertex_nei.x;
    }
    if (face_type_own == 1)
    {
        point_index_own.x = vertex_own.y;
        point_index_own.y = vertex_own.w;
        point_index_own.z = vertex_own.x;
        point_index_nei.x = vertex_nei.y;
        point_index_nei.y = vertex_nei.w;
        point_index_nei.z = vertex_nei.x;
    }
    if (face_type_own == 2)
    {
        point_index_own.x = vertex_own.z;
        point_index_own.y = vertex_own.w;
        point_index_own.z = vertex_own.x;
        point_index_nei.x = vertex_nei.z;
        point_index_nei.y = vertex_nei.w;
        point_index_nei.z = vertex_nei.x;
    }
    if (face_type_own == 3)
    {
        point_index_own.x = vertex_own.z;
        point_index_own.y = vertex_own.w;
        point_index_own.z = vertex_own.y;
        point_index_nei.x = vertex_nei.z;
        point_index_nei.y = vertex_nei.w;
        point_index_nei.z = vertex_nei.y;
    }

    if (point_index_nei.x == point_index_own.x)
    {
        face_nei_to_own.x = N_AMR_face_inner_last;
    }
    else if (point_index_nei.x == point_index_own.y)
    {
        face_nei_to_own.x = N_AMR_face_inner_last + 1;
    }
    else if (point_index_nei.x == point_index_own.z)
    {
        face_nei_to_own.x = i;
    }
    if (point_index_nei.y == point_index_own.x)
    {
        face_nei_to_own.y = N_AMR_face_inner_last;
    }
    else if (point_index_nei.y == point_index_own.y)
    {
        face_nei_to_own.y = N_AMR_face_inner_last + 1;
    }
    else if (point_index_nei.y == point_index_own.z)
    {
        face_nei_to_own.y = i;
    }
    if (point_index_nei.z == point_index_own.x)
    {
        face_nei_to_own.w = N_AMR_face_inner_last;
    }
    else if (point_index_nei.z == point_index_own.y)
    {
        face_nei_to_own.w = N_AMR_face_inner_last + 1;
    }
    else if (point_index_nei.z == point_index_own.z)
    {
        face_nei_to_own.w = i;
    }
    return face_nei_to_own;
}

__global__ void update_AMR_face_in_face(int* AMR_owner, int* AMR_neighbor, const int* d_owner, const int* d_neighbor, const int *d_if_cell_refinement, const int *d_if_face_refinement,
    const int3* face, const int4 *vertex, const double3 * point, int N_cell, int N_face, int N_inner_face, int AMR_N_add_face_in_inner_face, int AMR_N_add_face_in_cell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_inner_face)
    {
        int N_start = (i == 0) ? 0 : d_if_face_refinement[i - 1];
        if (d_if_face_refinement[i] - N_start > 0)
        {
            int owner_ID = d_owner[i];
            int neighbor_ID = d_neighbor[i];
            int N_AMR_face_inner_last = (i == 0) ? N_inner_face : N_inner_face + 3 * d_if_face_refinement[i - 1];
            bool if_own_AMR = d_if_cell_refinement[owner_ID] > 0 ? 1 : 0;
            bool if_nei_AMR = d_if_cell_refinement[neighbor_ID] > 0 ? 1 : 0;
            int f_own_type = d_face_type(face[i], vertex[owner_ID]);
            int f_nei_type = d_face_type(face[i], vertex[neighbor_ID]);

            if (if_own_AMR == 1 && if_nei_AMR == 1)
            {
                int4 own_cell_index = d_cell_index(f_own_type, N_cell, owner_ID);
                AMR_owner[N_AMR_face_inner_last] = own_cell_index.x;
                AMR_owner[N_AMR_face_inner_last + 1] = own_cell_index.y;
                AMR_owner[N_AMR_face_inner_last + 2] = own_cell_index.z;
                AMR_owner[i] = own_cell_index.w;

                int4 nei_cell_index = d_cell_index(f_nei_type, N_cell, neighbor_ID);
                int4 face_nei_to_own = d_map_face_nei_to_own(f_own_type, f_nei_type, vertex[owner_ID], vertex[neighbor_ID], N_AMR_face_inner_last, i);
                AMR_neighbor[face_nei_to_own.x] = nei_cell_index.x;
                AMR_neighbor[face_nei_to_own.y] = nei_cell_index.y;
                AMR_neighbor[face_nei_to_own.z] = nei_cell_index.z;
                AMR_neighbor[face_nei_to_own.w] = nei_cell_index.w;
            }
            else if (if_own_AMR == 1)
            {
                int4 own_cell_index = d_cell_index(f_own_type, N_cell, owner_ID);
                AMR_owner[N_AMR_face_inner_last] = own_cell_index.x;
                AMR_owner[N_AMR_face_inner_last + 1] = own_cell_index.y;
                AMR_owner[N_AMR_face_inner_last + 2] = own_cell_index.z;
                AMR_owner[i] = own_cell_index.w;
                AMR_neighbor[N_AMR_face_inner_last] = neighbor_ID;
                AMR_neighbor[N_AMR_face_inner_last + 1] = neighbor_ID;
                AMR_neighbor[N_AMR_face_inner_last + 2] = neighbor_ID;
                AMR_neighbor[i] = neighbor_ID;
            }
            else if (if_nei_AMR == 1)
            {
                int4 nei_cell_index = d_cell_index(f_nei_type, N_cell, neighbor_ID);
                AMR_neighbor[N_AMR_face_inner_last] = nei_cell_index.x;
                AMR_neighbor[N_AMR_face_inner_last + 1] = nei_cell_index.y;
                AMR_neighbor[N_AMR_face_inner_last + 2] = nei_cell_index.z;
                AMR_neighbor[i] = nei_cell_index.w;
                AMR_owner[N_AMR_face_inner_last] = owner_ID;
                AMR_owner[N_AMR_face_inner_last + 1] = owner_ID;
                AMR_owner[N_AMR_face_inner_last + 2] = owner_ID;
                AMR_owner[i] = owner_ID;
            }
        }
    }
}

__global__ void update_AMR_face_in_cell(int* AMR_owner, int* AMR_neighbor, const int* d_owner, const int* d_neighbor, const int *d_if_cell_refinement, const int *d_if_face_refinement,
    const int3* face, const int4 *vertex, const double3 * point, int N_cell, int N_face, int N_inner_face, int AMR_N_add_face_in_inner_face, int AMR_N_add_face_in_cell)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_cell)
    {
        if (d_if_cell_refinement[i] > 0)
        {
            int N_AMR_face_inner_last = N_inner_face + AMR_N_add_face_in_inner_face;
            AMR_owner[N_AMR_face_inner_last] = i;
            AMR_neighbor[N_AMR_face_inner_last] = N_cell + 4;
            AMR_owner[N_AMR_face_inner_last + 1] = N_cell;
            AMR_neighbor[N_AMR_face_inner_last + 1] = N_cell + 3;
            AMR_owner[N_AMR_face_inner_last + 2] = N_cell + 1;
            AMR_neighbor[N_AMR_face_inner_last + 2] = N_cell + 6;
            AMR_owner[N_AMR_face_inner_last + 3] = N_cell + 2;
            AMR_neighbor[N_AMR_face_inner_last + 3] = N_cell + 5;
            AMR_owner[N_AMR_face_inner_last + 4] = N_cell + 3;
            AMR_neighbor[N_AMR_face_inner_last + 4] = N_cell + 4;
            AMR_owner[N_AMR_face_inner_last + 5] = N_cell + 3;
            AMR_neighbor[N_AMR_face_inner_last + 5] = N_cell + 6;
            AMR_owner[N_AMR_face_inner_last + 6] = N_cell + 5;
            AMR_neighbor[N_AMR_face_inner_last + 6] = N_cell + 6;
            AMR_owner[N_AMR_face_inner_last + 7] = N_cell + 4;
            AMR_neighbor[N_AMR_face_inner_last + 7] = N_cell + 5;
        }
    }
}

void print_vtk(std::string name, int N_point, double3* point, int N_cell, int4* vertex)
{
    char filename[50];
    snprintf(filename, 50, "\\%s.vtk", name.c_str());

    FILE* fp = fopen(filename, "a");
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "%s\n", filename);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
    fprintf(fp, "POINTS %d double\n", N_point);
    for (int i = 0; i < N_point; i++)
    {
        fprintf(fp, "%f %f %f\n", point[i].x, point[i].y, point[i].z);
    }
    fprintf(fp, "CELLS %d %d\n", N_cell, 5 * N_cell);
    for (int i = 0; i < N_cell; i++)
    {
        fprintf(fp, "4 %d %d %d %d\n", vertex[i].x, vertex[i].y, vertex[i].z, vertex[i].w);
    }
    fprintf(fp, "CELL_TYPES %d\n", N_cell);
    for (int i = 0; i < N_cell; i++)
    {
        fprintf(fp, "10\n");
    }
    fclose(fp);
}

void main()
{
    //Build an initial model. Please check our AMR paper.
    int N_point = 8;
    int N_cell = 5;
    int N_face = 16;
    int N_inner_face = 4;

    double3* h_point = new double3[N_point];
    h_point[0] = double3{ 0,0,0 };
    h_point[1] = double3{ 1,0,0 };
    h_point[2] = double3{ 0,1,0 };
    h_point[3] = double3{ 1,1,0 };
    h_point[4] = double3{ 0,0,1 };
    h_point[5] = double3{ 1,0,1 };
    h_point[6] = double3{ 0,1,1 };
    h_point[7] = double3{ 1,1,1 };

    int* h_owner = new int[N_face];
    int* h_neighbor = new int[N_inner_face];

    h_owner[0] = 0;
    h_owner[1] = 1;
    h_owner[2] = 2;
    h_owner[3] = 3;

    h_owner[4] = 0;
    h_owner[5] = 2;
    h_owner[6] = 3;
    h_owner[7] = 1;
    h_owner[8] = 0;
    h_owner[9] = 3;
    h_owner[10] = 2;
    h_owner[11] = 1;
    h_owner[12] = 0;
    h_owner[13] = 1;
    h_owner[14] = 3;
    h_owner[15] = 2;

    h_neighbor[0] = 4;
    h_neighbor[1] = 4;
    h_neighbor[2] = 4;
    h_neighbor[3] = 4;

    int3* h_face = new int3[N_face];
    h_face[0] = int3{ 2, 1, 4 };
    h_face[1] = int3{ 4, 1, 7 };
    h_face[2] = int3{ 4, 2, 7 };
    h_face[3] = int3{ 2, 1, 7 };
    h_face[4] = int3{ 0, 2, 4 };
    h_face[5] = int3{ 4, 2, 6 };
    h_face[6] = int3{ 1, 3, 7 };
    h_face[7] = int3{ 1, 7, 5 };
    h_face[8] = int3{ 0, 1, 2 };
    h_face[9] = int3{ 1, 3, 2 };
    h_face[10] = int3{ 6, 4, 7 };
    h_face[11] = int3{ 4, 5, 7 };
    h_face[12] = int3{ 0, 1, 4 };
    h_face[13] = int3{ 4, 1, 5 };
    h_face[14] = int3{ 2, 3, 7 };
    h_face[15] = int3{ 2, 7, 6 };

    int4* h_vertex = new int4[N_cell];
    h_vertex[0] = int4{ 0, 1, 2, 4 };
    h_vertex[1] = int4{ 4, 1, 7, 5 };
    h_vertex[2] = int4{ 4, 7, 2, 6 };
    h_vertex[3] = int4{ 2, 1, 3, 7 };
    h_vertex[4] = int4{ 4, 1, 2, 7 };

    //Only the 5th cell is refined
    int* h_if_cell_refinement = new int[N_cell];
    h_if_cell_refinement[0] = 0;
    h_if_cell_refinement[1] = 0;
    h_if_cell_refinement[2] = 0;
    h_if_cell_refinement[3] = 0;
    h_if_cell_refinement[4] = 1;

    //Create new GPU arrays to store the topogy of the parent mesh
    double3* d_point;
    int* d_owner, * d_neighbor;
    int4* d_vertex;
    int3* d_face;
    int* d_if_cell_refinement, *d_if_face_refinement;

    cudaError_t cudaStatus = cudaSetDevice(0);
    cudaStatus = cudaMalloc((void**)&d_point, N_point * sizeof(double3));
    cudaStatus = cudaMalloc((void**)&d_owner, N_face * sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_face, N_face * sizeof(int3));
    cudaStatus = cudaMalloc((void**)&d_neighbor, N_inner_face * sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_vertex, N_cell * sizeof(int4));
    cudaStatus = cudaMalloc((void**)&d_if_cell_refinement, N_cell * sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_if_face_refinement, N_face * sizeof(int));

    cudaMemcpy(d_point, h_point, N_point * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_owner, h_owner, N_face * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_face, h_face, N_face * sizeof(int3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbor, h_neighbor, N_inner_face * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertex, h_vertex, N_cell * sizeof(int4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_if_cell_refinement, h_if_cell_refinement, N_cell * sizeof(int), cudaMemcpyHostToDevice);

    int blocknum_face = (N_face + threadnum - 1) / threadnum + 1;
    int blocknum_cell = (N_cell + threadnum - 1) / threadnum + 1;
    check_AMR_face << < blocknum_face, threadnum >> > (d_if_face_refinement, d_if_cell_refinement, d_owner, d_neighbor, N_face, N_inner_face);

    //Determine new number of points, cells, faces
    int AMR_N_point = N_point + 6 * thrust::reduce(thrust::device, d_if_cell_refinement, d_if_cell_refinement + N_cell);
    int AMR_N_cell = N_cell + 7 * thrust::reduce(thrust::device, d_if_cell_refinement, d_if_cell_refinement + N_cell);
    int AMR_N_add_face_in_inner_face = 3 * thrust::reduce(thrust::device, d_if_face_refinement, d_if_face_refinement + N_inner_face);
    int AMR_N_add_face_in_cell = 8 * thrust::reduce(thrust::device, d_if_cell_refinement, d_if_cell_refinement + N_cell);
    int AMR_N_inner_Face = N_inner_face + AMR_N_add_face_in_inner_face + AMR_N_add_face_in_cell;
    int AMR_N_Face = AMR_N_inner_Face + (N_face - N_inner_face) + 3 * thrust::reduce(thrust::device, d_if_face_refinement + N_inner_face, d_if_face_refinement + N_face);

    //Update numbering list
    thrust::inclusive_scan(thrust::device, d_if_face_refinement, d_if_face_refinement + N_face, d_if_face_refinement);

    //Create new GPU arrays to store new topogy of the AMR mesh
    double3* d_AMR_point;
    int* d_AMR_owner, * d_AMR_neighbor;
    int4* d_AMR_vertex;
    cudaStatus = cudaMalloc((void**)&d_AMR_point, AMR_N_point * sizeof(double3));
    cudaStatus = cudaMalloc((void**)&d_AMR_owner, AMR_N_Face * sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_AMR_neighbor, AMR_N_inner_Face * sizeof(int));
    cudaStatus = cudaMalloc((void**)&d_AMR_vertex, AMR_N_cell * sizeof(int4));

    //Copy the information of parent mesh to the AMR mesh
    cudaMemcpy(d_AMR_point, d_point, N_point * sizeof(double3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_AMR_owner, d_owner, N_inner_face * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_AMR_owner + AMR_N_inner_Face, d_owner + N_inner_face, (N_face - N_inner_face) * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_AMR_neighbor, d_neighbor, N_inner_face * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_AMR_vertex, d_vertex, N_cell * sizeof(int4), cudaMemcpyDeviceToDevice);

    //Construct the AMR topology
    update_AMR_point_vertex << < blocknum_cell, threadnum >> > (d_AMR_vertex, d_AMR_point, d_point, d_if_cell_refinement, d_vertex, N_cell, N_point);

    update_AMR_face_in_face << < blocknum_face, threadnum >> > (d_AMR_owner, d_AMR_neighbor, d_AMR_owner, d_AMR_neighbor, d_if_cell_refinement, d_if_face_refinement,
        d_face, d_vertex, d_point, N_cell, N_face, N_inner_face, AMR_N_add_face_in_inner_face, AMR_N_add_face_in_cell);

    update_AMR_face_in_cell << < blocknum_cell, threadnum >> > (d_AMR_owner, d_AMR_neighbor, d_AMR_owner, d_AMR_neighbor, d_if_cell_refinement, d_if_face_refinement,
        d_face, d_vertex, d_point, N_cell, N_face, N_inner_face, AMR_N_add_face_in_inner_face, AMR_N_add_face_in_cell);

    //Copy GPU array to CPU array for visualization
    double3* h_AMR_point = new double3[AMR_N_point];
    int* h_AMR_owner = new int[AMR_N_Face];
    int* h_AMR_neighbor = new int[AMR_N_inner_Face];
    int4* h_AMR_vertex = new int4[AMR_N_cell];

    cudaMemcpy(h_AMR_point, d_AMR_point, AMR_N_point * sizeof(double3), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_AMR_owner, d_AMR_owner, AMR_N_Face * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_AMR_neighbor, d_AMR_neighbor, AMR_N_inner_Face * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_AMR_vertex, d_AMR_vertex, AMR_N_cell * sizeof(int4), cudaMemcpyDeviceToHost);

    //output .vtk file for visualization
    print_vtk("AMR_Mesh", AMR_N_point, h_AMR_point, AMR_N_cell, h_AMR_vertex);
    print_vtk("Background_Mesh", N_point, h_point, N_cell, h_vertex);

    //free memory
    delete[] h_AMR_point;
    delete[] h_AMR_owner;
    delete[] h_AMR_neighbor;
    delete[] h_AMR_vertex;
    delete[] h_point;
    delete[] h_owner;
    delete[] h_neighbor;
    delete[] h_vertex;
    delete[] h_if_cell_refinement;

    cudaFree(d_AMR_point);
    cudaFree(d_AMR_neighbor);
    cudaFree(d_AMR_owner);
    cudaFree(d_AMR_vertex);
    cudaFree(d_point);
    cudaFree(d_owner);
    cudaFree(d_neighbor);
    cudaFree(d_vertex);
    cudaFree(d_face);
    cudaFree(d_if_cell_refinement);
    cudaFree(d_if_face_refinement);
}