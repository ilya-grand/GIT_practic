#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<random>
#include<cmath>
#include"mpi.h"

#include<iostream>
using namespace std;

#pragma warning(disable : 4996)

struct Point {
	double x, y, z;
};

struct Near_points {
	int index;
	double distance;
};

double GetDist_2(Point a, Point b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}

MPI_Datatype create_struct_point()
{
	MPI_Datatype new_type;

	Point tmp;

	int block_len[3] = { 1, 1, 1 };
	MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	MPI_Aint displs[3];

	MPI_Aint base;
	MPI_Get_address(&tmp, &base);
	MPI_Get_address(&tmp.x, &displs[0]);
	MPI_Get_address(&tmp.y, &displs[1]);
	MPI_Get_address(&tmp.z, &displs[2]);

	for (int i = 0; i < 3; i++)
	{
		displs[i] -= base;
	}

	MPI_Type_create_struct(3, block_len, displs, types, &new_type);
	MPI_Type_commit(&new_type);

	return new_type;
}

MPI_Datatype create_struct_near_points()
{
	MPI_Datatype new_type;

	Near_points tmp;

	int block_len[2] = { 1, 1 };
	MPI_Datatype types[2] = { MPI_INT, MPI_DOUBLE };
	MPI_Aint displs[2];

	MPI_Aint base;
	MPI_Get_address(&tmp, &base);
	MPI_Get_address(&tmp.index, &displs[0]);
	MPI_Get_address(&tmp.distance, &displs[1]);

	for (int i = 0; i < 2; i++)
	{
		displs[i] -= base;
	}

	MPI_Type_create_struct(2, block_len, displs, types, &new_type);
	MPI_Type_commit(&new_type);

	return new_type;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int rank, size;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N_points = 1000;

	if (rank == 0 && argc == 2)
	{
		N_points = atoi(argv[1]);
	}
	
	MPI_Bcast(&N_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

	vector<Point> mas_centers(N_points);

	if (rank == 0)
	{
		mt19937 rng(42);
		uniform_real_distribution<double> rnd(0.0, 1.0);

		for (int i = 0; i < N_points; i++)
		{
			mas_centers[i].x = rnd(rng);
			mas_centers[i].y = rnd(rng);
			mas_centers[i].z = rnd(rng);
		}
	}

	MPI_Datatype MPI_Point = create_struct_point();
	MPI_Datatype MPI_Near_Point = create_struct_near_points();

	MPI_Bcast(mas_centers.data(), N_points, MPI_Point, 0, MPI_COMM_WORLD);
	
	int porsion = N_points / size;
	
	int start_index = rank * porsion;
	int end_index;
	if (rank == size - 1)
	{
		end_index = N_points;
	}
	else
	{
		end_index = start_index + porsion;
	}
	int local_N_points = end_index - start_index;

	vector<Near_points> local_result(local_N_points);

	MPI_Barrier(MPI_COMM_WORLD);
	double start_time = MPI_Wtime();

	for (int i = start_index; i < end_index; i++)
	{
		double short_dist = 10e9;
		int need_index = -1;

		for (int j = 0; j < N_points; j++)
		{
			if (i != j)
			{
				double distance = GetDist_2(mas_centers[i], mas_centers[j]);

				if (distance < short_dist)
				{
					short_dist = distance;
					need_index = j;
				}
			}
		}

		local_result[i - start_index].distance = short_dist;
		local_result[i - start_index].index = need_index;
	}

	vector<int> count_points_on_proc(size);
	vector<int> displs(size);
	for (int i = 0; i < size; i++)
	{
		int s_ind = i * porsion;
		int e_ind;
		if (i == size - 1)
		{
			e_ind = N_points;
		}
		else
		{
			e_ind = s_ind + porsion;
		}
		
		count_points_on_proc.at(i) = e_ind - s_ind;
		displs.at(i) = s_ind;
	}

	vector<Near_points> global_result;
	if (rank == 0)
	{
		global_result.resize(N_points);
	}

	MPI_Gatherv(local_result.data(), local_N_points, MPI_Near_Point, global_result.data(), count_points_on_proc.data(), displs.data(), MPI_Near_Point, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double end_time = MPI_Wtime();

	if (rank == 0)
	{
		printf("Neighbor search time: _%.5f_ seconds on %d processors _%d_ points\n", end_time - start_time, size, N_points);
	}

	MPI_Type_free(&MPI_Point);
	MPI_Type_free(&MPI_Near_Point);

	MPI_Finalize();

	return 0;
}