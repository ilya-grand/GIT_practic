#include<Windows.h>
#include<stdio.h>
#include<stdlib.h>
#include<device_launch_parameters.h>
#include<cuda_runtime.h>
#include<ctime>
#include<cmath>
#include<algorithm>
#include<thrust/device_ptr.h>
#include<thrust/extrema.h>
using namespace std;

#define h_x 0.01
#define h_y 0.02
#define b 1
#define d 1
#define threads 256
#define EPS 0.0001

void Show_current_state(double* state, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			printf("%.5f ", state[i * m + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void CPU_Calculate(double* p_st, double* n_st, int n, int m, double tau)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == 0 || i == n - 1 || j == 0 || j == m - 1)
			{
				n_st[i * m + j] = p_st[i * m + j];
			}
			else
			{
				n_st[i * m + j] = ((p_st[(i + 1) * m + j] + p_st[(i - 1) * m + j]) / pow(h_x, 2) + (p_st[i * m + (j + 1)] + p_st[i * m + (j - 1)]) / pow(h_y, 2)) / (2. / pow(h_x, 2) + 2. / pow(h_y, 2));
			}
		}
	}
}

__global__
void GPU_Calculate(double* p_st, double* n_st, int n, int m, double tau)
{
	int lin_ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (lin_ind < n * m)
	{
		int i = lin_ind / m;
		int j = lin_ind % m;

		if (i == 0 || i == n - 1 || j == 0 || j == m - 1)
		{
			n_st[i * m + j] = p_st[i * m + j];
		}
		else
		{
			n_st[i * m + j] = ((p_st[(i + 1) * m + j] + p_st[(i - 1) * m + j]) / pow(h_x, 2) + (p_st[i * m + (j + 1)] + p_st[i * m + (j - 1)]) / pow(h_y, 2)) / (2. / pow(h_x, 2) + 2. / pow(h_y, 2));
		}
	}
}

void CPU_Make_list_dif(double* p_st, double* n_st, double* diff, int n, int m)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			diff[i * m + j] = abs(p_st[i * m + j] - n_st[i * m + j]);
		}
	}
}

__global__
void GPU_Make_list_dif(double* p_st, double* n_st, double* diff, int n, int m)
{
	int lin_ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (lin_ind < n * m)
	{
		int i = lin_ind / m;
		int j = lin_ind % m;

		diff[i * m + j] = abs(p_st[i * m + j] - n_st[i * m + j]);
	}
}

int main()
{
	SetConsoleOutputCP(1251);

	double tau = 0.5 / (1. / pow(h_x, 2) + 1. / pow(h_y, 2));

	int m, n;
	int size;
	int blocks;

	bool flag = true;
	double CPU_norm, GPU_norm;

	double* tmp;

	double* h_U_prev, * h_U_next, * h_err_list;
	double* d_U_prev, * d_U_next, * d_err_list;

	clock_t CPU_time_start, CPU_time_end;
	double CPU_time;
	cudaEvent_t GPU_time_start, GPU_time_end;
	float GPU_time = 0.0f;

	m = int(double(b) / h_x) + 1;
	n = int(double(d) / h_y) + 1;

	blocks = n * m / threads + 1;

	size = n * m * sizeof(double);

	h_U_prev = (double*)malloc(size);
	h_U_next = (double*)malloc(size);
	h_err_list = (double*)malloc(size);

	cudaMalloc((void**)&d_U_prev, size);
	cudaMalloc((void**)&d_U_next, size);
	cudaMalloc((void**)&d_err_list, size);

	cudaEventCreate(&GPU_time_start);
	cudaEventCreate(&GPU_time_end);

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			if (i == 0)
			{
				h_U_prev[i * m + j] = exp(1. - h_x * j);
			}
			if (j == 0)
			{
				h_U_prev[i * m + j] = exp(1. - h_y * i);
			}
			if (i == n - 1 || j == m - 1)
			{
				h_U_prev[i * m + j] = 1;
			}
			if (i > 0 && j > 0 && i < n - 1 && j < m - 1)
			{
				h_U_prev[i * m + j] = 0;
			}
		}
	}

	//Show_current_state(h_U_prev, n, m);

	cudaMemcpy(d_U_prev, h_U_prev, size, cudaMemcpyHostToDevice);

	CPU_time_start = clock();

	while (flag == true)
	{
		CPU_Calculate(h_U_prev, h_U_next, n, m, tau);

		CPU_Make_list_dif(h_U_prev, h_U_next, h_err_list, n, m);

		CPU_norm = *std::max_element(h_err_list, h_err_list + n * m);

		if (CPU_norm < EPS)
		{
			flag = false;
		}

		tmp = h_U_next;
		h_U_next = h_U_prev;
		h_U_prev = tmp;

		//Show_current_state(h_U_prev, n, m);
	}

	CPU_time_end = clock();

	CPU_time = (double)(CPU_time_end - CPU_time_start) / CLOCKS_PER_SEC * 1000;

	//printf("GPU\n");

	flag = true;

	cudaEventRecord(GPU_time_start);

	while (flag == true)
	{
		GPU_Calculate << <blocks, threads >> > (d_U_prev, d_U_next, n, m, tau);

		GPU_Make_list_dif << <blocks, threads >> > (d_U_prev, d_U_next, d_err_list, n, m);

		thrust::device_ptr<double> dev_ptr(d_err_list);
		GPU_norm = *thrust::max_element(dev_ptr, dev_ptr + n * m);

		if (GPU_norm < EPS)
		{
			flag = false;
		}

		tmp = d_U_next;
		d_U_next = d_U_prev;
		d_U_prev = tmp;

		/*cudaMemcpy(h_U_prev, d_U_prev, size, cudaMemcpyDeviceToHost);
		Show_current_state(h_U_prev, n, m);*/
	}

	cudaEventRecord(GPU_time_end);
	cudaEventSynchronize(GPU_time_end);
	cudaEventElapsedTime(&GPU_time, GPU_time_start, GPU_time_end);

	FILE* out_file = fopen("Выходные данные.txt", "w");

	if (out_file)
	{
		fprintf(out_file, "Простанственный шаг по переменной X: %.5f\n", h_x);
		fprintf(out_file, "Простанственный шаг по переменной Y: %.5f\n", h_y);
		fprintf(out_file, "Временной шаг: %.10f\n", tau);
		fprintf(out_file, "Погрешность: %.10f\n", EPS);
		fprintf(out_file, "Размерность матриц на каждом временном слое: %dx%d\n", m, n);
		fprintf(out_file, "Время вычислений на CPU: %.5f\n", CPU_time);
		fprintf(out_file, "Норма на CPU: %.5f\n", CPU_norm);
		fprintf(out_file, "Число потоков: %d\n", threads);
		fprintf(out_file, "Время вычислений на GPU: %.5f\n", GPU_time);
		fprintf(out_file, "Норма на GPU: %.5f\n", GPU_norm);
	}

	system("pause");

	return 0;
}