#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include<algorithm>

#include "dependence.h"
const int NUM_OF_DATA = 6496;
const int maxn = NUM_OF_DATA;
const int ALL_FEATURE = 11;
const int CLASSES = 1;


float Data[maxn][ALL_FEATURE];
float Label[maxn];

/*
float Data[maxn][ALL_FEATURE] = {-1,1,1,
		-1,-0.5,-1,
		-1,3,1,
		-1,-2,-1};;
float Label[maxn] = {1,-1,1,-1}; 
*/
using namespace std;
float toNum(string a)
{
	float ans = 0;
	float points = 0;
	int p;
	for(p = 0; p < a.length(); p++)
	{
		if (a[p] == '.')
			break;
		ans = ans * 10 + (a[p] - '0');

	}
	for(int i=a.length() - 1; i > p; i--)
		points = points / 10 + (a[i]- '0');
	ans += points / 10;
	return ans;
}
void initialInput()
{
	for(int i=0; i < ALL_FEATURE; i++)
	{
		float mins = 1000000;
		float maxs = -1000000;
		for(int j=0; j< NUM_OF_DATA; j++)
		{
			mins = min(mins, Data[j][i]);
			maxs = max(maxs, Data[j][i]);
		}
		for(int j=0; j< NUM_OF_DATA; j++)
			Data[j][i] = (Data[j][i] - mins) / (maxs - mins);
	}
}

bool LoadData(char* path){
	ifstream FileIn; 
	try{FileIn.open(path);}
	catch (exception e){return false;}

	string line;
	bool f = 0;
	int cnt = 0;
	int pos_num = 0;
	int neg_num = 0;   
	while(getline(FileIn, line))
	{
		if(f == 0)
		{
			f = 1;
			continue;
		}

		vector<float> temp;

		int before=0;
		for(int i = 0; i < line.length();i++)
		{
			if(line[i] == ';' || line[i] == '\n')
			{
				string sub = line.substr(before, i-before);
				temp.push_back(toNum(sub));
				before = i + 1;
			}
		}
		temp.push_back(toNum(line.substr(before, line.length() - before)));
		for(int i = 0; i < ALL_FEATURE; i++)
			Data[cnt][i] = temp[i];
		if (temp[ALL_FEATURE] > 5)
		{
			Label[cnt] = 1;
			pos_num += 1;
		}
		else
		{
			Label[cnt] = -1;
			neg_num += 1;
		}
		cnt++;
	}
	cout << "data loading done. \ntotal number of data is: " << cnt << endl;
	cout << "number of pos is: " << pos_num << "number of neg is: " << neg_num <<endl;
	FileIn.close();
}
void ShowData(){
	for(int i = 0; i < NUM_OF_DATA; i++)
	{
		for(int j = 0; j < ALL_FEATURE; j++)
			cout << Data[i][j] << ' ';
		cout << Label[i] << endl;
	}
}
void matrixMulCpu(float* A, float* B, float* C, int m, int n, int k )
{
	/*A[m, k] x B[k, n] = C[m, n]*/
	float sum = 0.0f;
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			for(int l = 0; l < k; l++)
			{
				sum += A[i*k + l] * B[l*n + j];
			}
			C[i*n + j] = sum;
			sum = 0.0;
		}
	}
}

__global__ void matrixMulGpu(float* A, float* B, float* C, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    for(int i =0; i < k; i++)
    {
        fCVal += A[nRow * k + i] * B[i * n + nCol];
    }

    C[nRow * n + nCol] = fCVal;
}



void sumMatrix2D_CPU(float * MatA,float * MatB,float * MatC,int nx,int ny, float ratio_a, float ratio_b)
{ 
	float * a=MatA;
	float * b=MatB;
	float * c=MatC;
	for(int j=0;j<ny;j++)
	{ 
		for(int i=0;i<nx;i++)
		{ 
			c[i]=ratio_a * a[i]+ ratio_b * b[i];
		}
		c+=nx;
		b+=nx;
		a+=nx;
	}
}

__global__ void sumMatrix2D_GPU(float * MatA,float * MatB,float * MatC,int nx,int ny, float ratio_a, float ratio_b)
 {
 	int ix=threadIdx.x+blockDim.x*blockIdx.x;
 	int iy=threadIdx.y+blockDim.y*blockIdx.y;
 	int idx=ix+iy*ny;
 	if (ix<nx && iy<ny)
 	{
 		MatC[idx]=ratio_a * MatA[idx]+ ratio_b * MatB[idx];
 	}
 }



void matrixMulCpu_transpose(float* A, float* B, float* C, int m, int n, int k)
{
	/*A[k, m] x B[k, n] ->= C[m, n]*/
	float sum = 0.0f;
	for(int i = 0; i < m; i++)
	{
		for(int j = 0; j < n; j++)
		{
			for(int l = 0; l < k; l++)
			{
				sum += A[l*m + i] * B[l*n + j];

			}
			C[i*n + j] = sum;
			sum = 0.0;
		}
	}
}

__global__ void matrixMulGpu_transpose(float* A, float* B, float* C, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    float fCVal = 0.0f;

    for(int i =0; i < k; i++)
    {
        fCVal += A[i * m + nRow] * B[i * n + nCol];
    }
    C[nRow * n + nCol] = fCVal;
}


void Preceptron_cpu()
{
	/*initial matrix W*/
	//double iStart_t0 = cpuSecond();

	printf("strating...\n");
	int nxy_w = ALL_FEATURE * CLASSES;
	
	int nBytes = nxy_w * sizeof(float);
	float* W_host = (float*)malloc(nBytes);
	initialData(W_host, nxy_w);

	int nxy_x = NUM_OF_DATA * ALL_FEATURE;
	nBytes = nxy_x * sizeof(float);
	float* X_host = (float*)malloc(nBytes);

	int nxy_y = NUM_OF_DATA * 1;
	nBytes = nxy_y * sizeof(float);
	float* Y_host = (float*)malloc(nBytes);

	for(int i=0; i < NUM_OF_DATA; i++ )
		for(int j=0; j < ALL_FEATURE; j++ )
			X_host[i * ALL_FEATURE + j] = Data[i][j];
		
	for(int i=0; i < nxy_y; i++ )
		Y_host[i] = Label[i];



	double l=0.0001;
	float Y1[NUM_OF_DATA] = {0};	

	//matrixMulCpu(X_host, W_host, Y1, NUM_OF_DATA, CLASSES, ALL_FEATURE);

	//backpropation
	
	float tmp_res[NUM_OF_DATA] = {0};
	float tmp_res3[ALL_FEATURE] = {0};
	int early_stop = 0;
	float best_acc = 0;
	double iElaps1 = 0;
	double iElaps2 = 0;
	double iElaps3 = 0;
	double iElaps4 = 0;
	double iElaps = 0;

	double iStart;
	double iStart_t;
	double iStart_t0 = cpuSecond();

	for(int idx = 0; idx < 500; idx++)
	{
		iStart=cpuSecond();
		iStart_t=cpuSecond();

		matrixMulCpu(X_host, W_host, Y1, NUM_OF_DATA, CLASSES, ALL_FEATURE);
		iElaps1 += cpuSecond()-iStart;
		
		iStart=cpuSecond();
		sumMatrix2D_CPU(Label, Y1, tmp_res, NUM_OF_DATA, CLASSES, 1, -1 );
		iElaps2 += cpuSecond()-iStart;
		
		iStart=cpuSecond();
		matrixMulCpu_transpose(X_host, tmp_res, tmp_res3, ALL_FEATURE, CLASSES, NUM_OF_DATA);
		iElaps3 += cpuSecond()-iStart;
		
		iStart=cpuSecond();
		sumMatrix2D_CPU(W_host, tmp_res3, W_host, ALL_FEATURE, CLASSES, 1, l);
		iElaps4 += cpuSecond()-iStart;
		iElaps += cpuSecond()-iStart_t;


		int correct = 0;
		for(int i = 0; i < NUM_OF_DATA; i++)
		{
			if(Y1[i] < 0)
				Y1[i] = -1.;
			else 
				Y1[i] = 1.;
			if(int(Y1[i]) == int(Label[i]))
				correct += 1;
			/*
			if( i < 5)
				printf("label:%f, real:%f ", Label[i], Y1[i]);
			*/
		}
		//printf("\n");
		float acc = correct * 1. / NUM_OF_DATA * 100;
		best_acc = max(best_acc, acc);

		//cout << "the epoch: " << idx << " accuracy is: " << acc << '%'<< endl;

		/*
		if(early_stop == 10)
		{
			cout << "Early stop at epoch: " << idx << " accuracy is  " << best_acc << '%'<< endl;
			break;
		}
		*/
	}
	cout << "Best accuracy is  " << best_acc << '%'<< endl;
	printf("CPU Execution Step1 Time elapsed %f sec\n",iElaps1);
	printf("CPU Execution Step2 Time elapsed %f sec\n",iElaps2);
	printf("CPU Execution Step3 Time elapsed %f sec\n",iElaps3);
	printf("CPU Execution Step4 Time elapsed %f sec\n",iElaps4);
	printf("CPU Execution Total Time elapsed %f sec\n",iElaps);
	printf("CPU Execution ALL Time elapsed %f sec\n",cpuSecond() - iStart_t0);


	for(int i = 0; i < ALL_FEATURE; i++)
		cout << W_host[i] << ' ' ;
	cout << endl;


}

void Preceptron_gpu()
{
	/*initial matrix W*/

	
	printf("strating...\n");
	initDevice(0);
	int nxy_w = ALL_FEATURE * CLASSES;
	
	int nBytes = nxy_w * sizeof(float);

	int nxy_x = NUM_OF_DATA * ALL_FEATURE;
	nBytes = nxy_x * sizeof(float);
	float* X_host = (float*)malloc(nBytes);

	int nxy_y = NUM_OF_DATA * 1;
	nBytes = nxy_y * sizeof(float);
	float* Y_host = (float*)malloc(nBytes);

	for(int i=0; i < NUM_OF_DATA; i++ )
		for(int j=0; j < ALL_FEATURE; j++ )
			X_host[i * ALL_FEATURE + j] = Data[i][j];
		
	for(int i=0; i < nxy_y; i++ )
		Y_host[i] = Label[i];



	double l=0.0001;
	float Y1[NUM_OF_DATA] = {0};	

	//matrixMulCpu(X_host, W_host, Y1, NUM_OF_DATA, CLASSES, ALL_FEATURE);

	//backpropation
	
	float tmp_res[NUM_OF_DATA] = {0};
	float tmp_res3[ALL_FEATURE] = {0};
	int early_stop = 0;
	float best_acc = 0;

	double iElaps1 = 0;
	double iElaps2 = 0;
	double iElaps3 = 0;
	double iElaps4 = 0;
	double iElaps = 0;

	double iStart;
	double iStart_t;

	/*GPU Version*/
	float* W_host_gpu = (float*)malloc(nBytes);
	initialData(W_host_gpu, nxy_w);
	for(int i =0 ; i < NUM_OF_DATA; i++)
		Y1[i] = 0;
	float Y1_gpu[NUM_OF_DATA] = {0};	
	float tmp_res_gpu[NUM_OF_DATA] = {0};

	float tmp_res3_gpu[ALL_FEATURE] = {0};
	float W_tmp_gpu[ALL_FEATURE] = {0};


	float *X_dev = NULL;
	float *W_dev = NULL;
	float *T3_dev = NULL;	
	float *T_dev = NULL;
	float *Label_dev = NULL;
	float *Y1_dev = NULL;	


	int bytes_x = NUM_OF_DATA * ALL_FEATURE *sizeof(float);
	int bytes_w = ALL_FEATURE * CLASSES * sizeof(float);
	int bytes_t3 = ALL_FEATURE * CLASSES * sizeof(float);

	int bytes_t = NUM_OF_DATA * CLASSES * sizeof(float);
	int bytes_label = NUM_OF_DATA * CLASSES * sizeof(float);
	int bytes_y1 = NUM_OF_DATA * CLASSES * sizeof(float);


	CHECK(cudaMalloc((void**)&X_dev,bytes_x));
	CHECK(cudaMalloc((void**)&W_dev,bytes_w));
	CHECK(cudaMalloc((void**)&T3_dev,bytes_t3));

	CHECK(cudaMalloc((void**)&T_dev,bytes_t));
	CHECK(cudaMalloc((void**)&Label_dev,bytes_label));
	CHECK(cudaMalloc((void**)&Y1_dev,bytes_y1));


	CHECK(cudaMemcpy(X_dev,X_host,bytes_x,cudaMemcpyHostToDevice));


	int size1 = 2;

	CHECK(cudaMemcpy(Label_dev,Label,bytes_label,cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(T_dev,tmp_res_gpu, bytes_t,cudaMemcpyHostToDevice));

	// block_0, grid_0
	int dimx0 = 1;
	int dimy0 = size1;
	dim3 block_0(dimx0,dimy0);
	dim3 grid_0((CLASSES-1)/block_0.x+1,(NUM_OF_DATA-1)/block_0.y+1);


	// block_1, grid_1
	int dimx1 = size1;
	int dimy1 = 1;
	dim3 block_1(dimx1,dimy1);
	dim3 grid_1((NUM_OF_DATA-1)/block_1.x+1,(CLASSES-1)/block_1.y+1);


	// block_2, grid_2
	int dimx2 = 1;
	int dimy2 = 11;
	dim3 block_2(dimx2,dimy2);
	dim3 grid_2((CLASSES-1)/block_2.x+1,(ALL_FEATURE-1)/block_2.y+1);


	// block_3, grid_3
	int dimx3 = 11;
	int dimy3 = 1;
	dim3 block_3(dimx3,dimy3);
	dim3 grid_3((ALL_FEATURE-1)/block_3.x+1,(CLASSES-1)/block_3.y+1);



	double iStart_t0 = cpuSecond();
	//CHECK(cudaMemcpy(Y1_dev, Y1, bytes_y1, cudaMemcpyHostToDevice));
	for(int idx = 0; idx < 500; idx++)
	{
		iStart=cpuSecond();
		iStart_t=cpuSecond();
		matrixMulGpu<<<grid_0,block_0>>>(X_dev, W_dev, Y1_dev, NUM_OF_DATA, CLASSES, ALL_FEATURE);
		iElaps1 += cpuSecond()-iStart;

		iStart=cpuSecond();
		sumMatrix2D_GPU<<<grid_1,block_1>>>(Label_dev, Y1_dev, T_dev, NUM_OF_DATA, CLASSES, 1, -1);
		iElaps2 += cpuSecond()-iStart;
		
		iStart=cpuSecond();
		matrixMulGpu_transpose<<<grid_2,block_2>>>(X_dev, T_dev, T3_dev, ALL_FEATURE, CLASSES, NUM_OF_DATA);
		iElaps3 += cpuSecond()-iStart;
		
		iStart=cpuSecond();
		sumMatrix2D_GPU<<<grid_3,block_3>>>(W_dev, T3_dev, W_dev, ALL_FEATURE, CLASSES, 1, l);
		iElaps4 += cpuSecond()-iStart;
		iElaps += cpuSecond()-iStart_t;

		CHECK(cudaMemcpy(Y1_gpu, Y1_dev,bytes_y1,cudaMemcpyDeviceToHost));


		int correct = 0;
		/*
		for(int i = 0; i < 5; i++)
			cout << Y1[i] << ' ';
		cout << endl;
		*/
		
		for(int i = 0; i < NUM_OF_DATA; i++)
		{
			if(Y1_gpu[i] < 0)
				Y1_gpu[i] = -1.;
			else 
				Y1_gpu[i] = 1.;
			if(int(Y1_gpu[i]) == int(Label[i]))
				correct += 1;
		
		}
		float acc = correct * 1. / NUM_OF_DATA * 100;
		best_acc = max(best_acc, acc);
		

	}
	cout << "Best accuracy is  " << best_acc << '%'<< endl;
	printf("GPU Execution Step1 Time elapsed %f sec\n",iElaps1);
	printf("GPU Execution Step2 Time elapsed %f sec\n",iElaps2);
	printf("GPU Execution Step3 Time elapsed %f sec\n",iElaps3);
	printf("GPU Execution Step4 Time elapsed %f sec\n",iElaps4);
	printf("GPU Execution Total Time elapsed %f sec\n",iElaps);
	printf("GPU Execution ALL Time elapsed %f sec\n",cpuSecond() - iStart_t0);

	for(int i = 0; i < ALL_FEATURE; i++)
		cout << W_host_gpu[i] << ' ' ;
	cout << endl;

	

}





int main ()
{
	 
	char* path = "data/winequality-white.csv";
	LoadData(path);
	initialInput();
	//sampleData();
	//ShowData();
	Preceptron_cpu();
	Preceptron_gpu();

	return 0;
	
}