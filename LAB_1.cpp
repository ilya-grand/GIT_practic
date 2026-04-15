#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<random>
#include<cmath>
#include<omp.h>

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

int GetCellIndex(double coord, int grid_size)
{
	int index = (int)(coord * grid_size);

	if (index >= grid_size)
	{
		index = grid_size - 1;
	}

	return index;
}

int GetCellId(int Idx, int Idy, int Idz, int grid_size)
{
	return Idx + Idy * grid_size + Idz * grid_size * grid_size;
}

void Control_count_points_in_grid(int count, vector<vector<int>> grid)
{
	int counts = 0;

	for (int i = 0; i < grid.size(); i++)
	{
		counts += grid.at(i).size();
	}

	if (counts == count)
	{
		printf("Порядок\n");
	}
	else
	{
		printf("Несоответствие количества точек\n");
	}
}

int main(int argc, char *argv[])
{
	SetConsoleOutputCP(1251);

	int N_points = 5000000;

	if (argc == 2)
	{
		N_points = atoi(argv[1]);
	}

	int grid_size = cbrt(N_points);
	int all_count_cells = grid_size * grid_size * grid_size;

	vector<Point> mas_centers(N_points);
	vector<vector<int>> grid(all_count_cells);
	vector<Near_points> result(N_points);

	mt19937 rng(42);
	uniform_real_distribution<double> rnd(0.0, 1.0);

	for (int i = 0; i < N_points; i++)
	{
		mas_centers[i].x = rnd(rng);
		mas_centers[i].y = rnd(rng);
		mas_centers[i].z = rnd(rng);

		int Idx = GetCellIndex(mas_centers[i].x, grid_size);
		int Idy = GetCellIndex(mas_centers[i].y, grid_size);
		int Idz = GetCellIndex(mas_centers[i].z, grid_size);

		int cell_ind = GetCellId(Idx, Idy, Idz, grid_size);
		grid[cell_ind].push_back(i);
	}

	//Control_count_points_in_grid(N_points, grid);

	double start_time = omp_get_wtime();

#pragma omp parallel for shared(N_points, mas_centers, grid, grid_size)
	for (int i = 0; i < N_points; i++)
	{
		Point p = mas_centers[i];

		int Idx = GetCellIndex(p.x, grid_size);
		int Idy = GetCellIndex(p.y, grid_size);
		int Idz = GetCellIndex(p.z, grid_size);

		double short_dist = 10e9;
		int need_index = -1;

		int r = 1;

		while (need_index == -1)
		{
			for (int dx = -r; dx <= r; dx++)
			{
				for (int dy = -r; dy <= r; dy++)
				{
					for (int dz = -r; dz <= r; dz++)
					{
						int Id_cx = Idx + dx;
						int Id_cy = Idy + dy;
						int Id_cz = Idz + dz;

						if (Id_cx < 0 || Id_cy < 0 || Id_cz < 0 || Id_cx >= grid_size || Id_cy >= grid_size || Id_cz >= grid_size)
						{
							continue;
						}

						int cell_cind = GetCellId(Id_cx, Id_cy, Id_cz, grid_size);

						int size_Ccell = grid[cell_cind].size();

						for (int j = 0; j < size_Ccell; j++)
						{
							int k = grid[cell_cind].at(j);

							if (k == i)
							{
								continue;
							}

							double distance = GetDist_2(mas_centers[i], mas_centers[k]);

							if (distance < short_dist)
							{
								short_dist = distance;
								need_index = k;
							}
						}
					}
				}
			}

			r++;
		}

		result[i].distance = short_dist;
		result[i].index = need_index;
	}

	double end_time = omp_get_wtime();

	printf("Время поиска соседей: _%.5f_ секунд для _%d_ центров ячеек Вороного\n", end_time - start_time, N_points);

	/*FILE* file_out = fopen("Центра ячеек Вороного.txt", "w");

	for (int i = 0; i < N_points; i++)
	{
		fprintf(file_out, "%.5f %.5f %.5f\n", mas_centers[i].x, mas_centers[i].y, mas_centers[i].z);
	}*/

	return 0;
}