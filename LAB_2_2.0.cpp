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
	int i, index;
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

	MPI_Datatype my_MPI_Point = create_struct_point();
	MPI_Datatype my_MPI_Near_Point = create_struct_near_points();

	MPI_Bcast(mas_centers.data(), N_points, my_MPI_Point, 0, MPI_COMM_WORLD);

	if (size == 1)
	{
		vector<Near_points> global_result;

		MPI_Barrier(MPI_COMM_WORLD);
		double start_time = MPI_Wtime();

		for (int i = 0; i < N_points; i++)
		{
			if (i == 0)
			{
				continue;
			}
			double short_dist = 10e9;
			int need_index = -1;

			for (int j = 0; j < i; j++)
			{
				double distance = GetDist_2(mas_centers[i], mas_centers[j]);

				if (distance < short_dist)
				{
					short_dist = distance;
					need_index = j;
				}
			}

			global_result.push_back(Near_points{ i, need_index, short_dist });
		}

		MPI_Barrier(MPI_COMM_WORLD);
		double end_time = MPI_Wtime();

		if (rank == 0)
		{
			printf("Neighbor search time: _%.5f_ seconds on %d processors _%d_ points\n", end_time - start_time, size, N_points);
		}
	}
	else
	{
		vector<Near_points> local_result;

		MPI_Barrier(MPI_COMM_WORLD);
		double start_time = MPI_Wtime();

		for (int i = rank; i < N_points; i += size)
		{
			if (i == 0)
			{
				continue;
			}
			double short_dist = 10e9;
			int need_index = -1;

			for (int j = 0; j < i; j++)
			{
				double distance = GetDist_2(mas_centers[i], mas_centers[j]);

				if (distance < short_dist)
				{
					short_dist = distance;
					need_index = j;
				}
			}

			local_result.push_back(Near_points{ i, need_index, short_dist });
		}

		vector<int> count_points_on_proc(size), displs(size);

		int local_size = local_result.size();
		MPI_Gather(&local_size, 1, MPI_INT, count_points_on_proc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

		vector<Near_points> global_result;
		int total = 0;

		if (rank == 0)
		{
			for (int i = 0; i < size; i++)
			{
				displs[i] = total;
				total += count_points_on_proc[i];
			}
			global_result.resize(total);
		}

		MPI_Gatherv(local_result.data(), local_size, my_MPI_Near_Point, global_result.data(), count_points_on_proc.data(), displs.data(), my_MPI_Near_Point, 0, MPI_COMM_WORLD);

		MPI_Barrier(MPI_COMM_WORLD);
		double end_time = MPI_Wtime();

		if (rank == 0)
		{
			printf("Neighbor search time: _%.5f_ seconds on %d processors _%d_ points\n", end_time - start_time, size, N_points);
		}

		MPI_Type_free(&my_MPI_Near_Point);
	}

	MPI_Type_free(&my_MPI_Point);

	MPI_Finalize();

	return 0;
}