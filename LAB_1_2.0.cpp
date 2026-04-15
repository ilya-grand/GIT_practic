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

int main(int argc, char* argv[])
{
	SetConsoleOutputCP(1251);

	int N_points = 1000;

	if (argc == 2)
	{
		N_points = atoi(argv[1]);
	}

	vector<Point> mas_centers(N_points);
	vector<Near_points> result(N_points);

	mt19937 rng(42);
	uniform_real_distribution<double> rnd(0.0, 1.0);

	for (int i = 0; i < N_points; i++)
	{
		mas_centers[i].x = rnd(rng);
		mas_centers[i].y = rnd(rng);
		mas_centers[i].z = rnd(rng);
	}

	double start_time = omp_get_wtime();

#pragma omp parallel for shared(N_points, mas_centers)
	for (int i = 0; i < N_points; i++)
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