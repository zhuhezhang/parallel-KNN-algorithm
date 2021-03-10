/*基于KNN算法的分类器
	  假如你有一个朋友一直使用在线约会网站寻找合适自己的约会对象。尽管约会网站会推荐不同的人选，
但她没有从中找到喜欢的人。经过一番总结，她发现曾交往过三种类型的人：
	  1、不喜欢的人
	  2、喜欢的人
	  3、很喜欢的人
	  尽管发现了上述规律，你的朋友依然无法将约会网站推荐的匹配对象归入恰当的分类。朋友希望我们
的分类软件可以更好的帮助她将匹配对象划分到确切的分类中。根据她的要求，收集了多条数据，数据包
含三个特征（每年获得的飞行常客里程数，玩视频游戏所耗时间百分比，每周消费的冰淇淋公升数），而
标签类型为(1代表“不喜欢的人”，2代表“喜欢的人”，3代表“很喜欢的人”)*/


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include "mpi.h"


#define XUNLIANFILENAME "xunlian.txt"//训练数据集的文件名
#define RESULTFILENAME "result.txt"  //输出结果的文件名
#define CESHIFILENAME "ceshi.txt"	 //测试数据集的文件名
#define XUNLIANNUM 1000              //训练样本点个数
#define XUNLIANDIM 4			     //训练样本点维度
#define CESHINUM 100				     //测试样本点个数
#define CESHIYANZHENGDIM 3		     //测试/验证样本点维度
#define YANZHENGNUM 100			     //验证样本点个数（这里取训练集前100个）
#define ISYANZHENG 1                //是否验证（1是0否）
#define K 7							 //KNN算法的K值
#define TYPECOUNT 3				     //种类的个数，也就是不喜欢、喜欢、很喜欢这三种


//读文件（文件名、样本点个数、样本点维度、样本所属类别、样本数据集
void Read(char* filename, int number, int dim, int* category, double* dataSet)
{
	FILE* fp = fopen(filename, "r");
	if (fp == NULL)
	{
		printf("文件%s打开出错！\n", filename);
		exit(1);
	}
	setvbuf(fp, NULL, _IOFBF, BUFSIZ);//设置完全缓存模式

	for (int i = 0; i < number; i++)//样本一个一个读取
	{
		for (int j = 0; j < CESHIYANZHENGDIM; j++)
		{
			fscanf(fp, "%lf", &dataSet[i * CESHIYANZHENGDIM + j]);//读取数据
		}
		if (dim == XUNLIANDIM)//如果是训练集数据文件
		{
			fscanf(fp, "%d", &category[i]);//样本点所属类别
		}
	}
	fclose(fp);
}


//写文件（结果数据集）
void Write(int* result)
{
	FILE* fp = fopen(RESULTFILENAME, "w");
	if (fp == NULL)
	{
		printf("文件%s打开出错！\n", RESULTFILENAME);
		exit(1);
	}
	setvbuf(fp, NULL, _IOFBF, BUFSIZ);                  //设置完全缓存模式
	for (int i = 0; i < CESHINUM; i++)
	{
		fprintf(fp, "%d\n", result[i]);
	}
	fclose(fp);
}


//在训练/测试数据集中找出最值
void FindMinMaxValue(int num, double* minValue, double* maxValue, double* data)
{                     //样本点个数、存储测试/训练数据集最小、最大值数组、训练/测试数据集
	for (int i = 0; i < CESHIYANZHENGDIM;i++)//先给初始值
	{
		minValue[i] = data[i];
		maxValue[i] = data[i];
	}
	for (int i = 0; i < num;i++)//计算最小最大值
	{
		for (int j = 0; j < CESHIYANZHENGDIM;j++)
		{
			if (minValue[j] > data[i * CESHIYANZHENGDIM + j])//找最小值	
			{
				minValue[j] = data[i * CESHIYANZHENGDIM + j];
			}
			if (maxValue[j] < data[i * CESHIYANZHENGDIM + j])//找最大值	
			{
				maxValue[j] = data[i * CESHIYANZHENGDIM + j];
			}
		}
	}
}


//通过对比在训练集、测试集中的最值找出最终的最值
void FindMinMaxValueOfAll(double* xunlian_min, double* xunlian_max, double* ceshi_min, double* ceshi_max, double* min, double* max)
{							//训练数据集最值、测试数据集最值、最终的最值
	for (int i = 0;i < CESHIYANZHENGDIM;i++)
	{
		if (xunlian_min[i] >= ceshi_min[i])//找出最小值
		{
			min[i] = ceshi_min[i];
		}
		else
		{
			min[i] = xunlian_min[i];
		}

		if (xunlian_max[i] <= ceshi_max[i])//找出最大值
		{
			max[i] = ceshi_max[i];
		}
		else
		{
			max[i] = xunlian_max[i];
		}
	}
}


//归一化训练集/测试集数据（样本点个数、要归一化的数据、归一后的数据、最小最大值
void GuiYiData(int number, double* data, double* dataGuiyi, double* minValueOfAll, double* maxValueOfAll)
{
	for (int i = 0; i < number;i++)
	{
		for (int j = 0; j < CESHIYANZHENGDIM;j++)
		{
			dataGuiyi[i * CESHIYANZHENGDIM + j] = (data[i * CESHIYANZHENGDIM + j] - minValueOfAll[j]) / (maxValueOfAll[j] - minValueOfAll[j]);
		}
	}
}


//计算测试/验证集和训练集的样本点之间的欧式距离（归一化后训练数据、归一化后的测试数据、欧式距离、样本点个数）
void CalEuclidDistance(double* xunlianDataGuiyi, double* ceshiDataGuiyi, double* euclidDistance, int num)
{
	for (int i = 0;i < num;i++)
	{
		for (int j = 0;j < XUNLIANNUM;j++)
		{
			euclidDistance[i * XUNLIANNUM + j] = sqrt(pow((xunlianDataGuiyi[j * CESHIYANZHENGDIM] - ceshiDataGuiyi[i * CESHIYANZHENGDIM]), 2.0) +
				pow((xunlianDataGuiyi[j * CESHIYANZHENGDIM + 1] - ceshiDataGuiyi[i * CESHIYANZHENGDIM + 1]), 2.0) +
				pow((xunlianDataGuiyi[j * CESHIYANZHENGDIM + 2] - ceshiDataGuiyi[i * CESHIYANZHENGDIM + 2]), 2.0));
		}
	}
}


//快速排序算法（要排序的欧式距离数组、所属类别、数组左边界、数组右边界）
void QuikSort(double* euclidDistance, int* category, int left, int right)
{
	if (left >= right)//如果左边索引大于或者等于右边的索引就代表已经整理完成一个组了
	{
		return;
	}
	int i = left;
	int j = right;
	double key = euclidDistance[left];
	int key2 = category[left];

	while (i < j)
	{
		while (i < j && key <= euclidDistance[j])//寻找结束的条件，1，找到一个小于或者大于key的数（大于或小于取决于你想升序还是降序）
		{											//2，没有符合条件1的，并且i与j的大小没有反转
			j--;//向前寻找
		}
		euclidDistance[i] = euclidDistance[j];//找到一个这样的数后就把它赋给前面的被拿走的i的值（如果第一次循环且key是a[left]，那么就是给key）
		category[i] = category[j];//所属类别也要跟着交换

		while (i < j && key >= euclidDistance[i])//这是i在当组内向前寻找，同上，不过注意与key的大小关系停止循环和上面相反，
		{							//因为排序思想是把数往两边扔，所以左右两边的数大小与key的关系相反
			i++;
		}
		euclidDistance[j] = euclidDistance[i];
		category[j] = category[i];
	}

	euclidDistance[i] = key;//当在当组内找完一遍以后就把中间数key回归
	category[i] = key2;
	QuikSort(euclidDistance, category, left, i - 1);//最后用同样的方式对分出来的左边的小组进行同上的做法
	QuikSort(euclidDistance, category, i + 1, right);//用同样的方式对分出来的右边的小组进行同上的做法
}


//利用快速排序算法对计算的欧式距离进行排序，注意：是以XUNLIANNUM为一组进行排序（欧式距离、所属类别、样本点个数）
void SortEuclidDistance(double* euclidDistance, int* category, int num)
{
	for (int i = 0;i < num;i++)
	{
		QuikSort(euclidDistance, category, i * XUNLIANNUM, i * XUNLIANNUM + XUNLIANNUM - 1);//利用快速排序算法对欧式距离进行排序
	}
}


//根据K值逐个计算出数据的所属类别
void CalResult(int* categoryOfAll, int* result, int num)
{
	for (int i = 0;i < num;i++)//逐个计算数据的所属类别
	{
		int typeCount[TYPECOUNT] = { 0,0,0 };//初始化
		int type[] = { 1,2,3 };//数据分为1、2、3类
		for (int j = 0;j < K;j++)//根据K值算出该测试数据的所属类别
		{
			if (categoryOfAll[i * XUNLIANNUM + j] == 1)
			{
				typeCount[0]++;
			}
			else if (categoryOfAll[i * XUNLIANNUM + j] == 2)
			{
				typeCount[1]++;
			}
			else
			{
				typeCount[2]++;
			}
		}
		if (typeCount[0] > typeCount[1] && typeCount[0] > typeCount[2])
		{
			result[i] = type[0];
		}
		else if (typeCount[1] > typeCount[0] && typeCount[1] > typeCount[2])
		{
			result[i] = type[1];
		}
		else
		{
			result[i] = type[2];
		}
	}
}


//主函数
int main(int argc, char* argv[])
{
	int myid, numprocess;//进程id、进程个数
	double start, finish;//开始、结束时间
	MPI_Init(&argc, &argv);//初始化
	start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);//获取当前进程id
	MPI_Comm_size(MPI_COMM_WORLD, &numprocess);//获取总进程数

	if (XUNLIANNUM % numprocess != 0)//只支持平均分配情况，若不是，则退出
	{
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (CESHINUM % numprocess != 0)
	{
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	if (YANZHENGNUM % numprocess != 0)
	{
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	double* xunlianData = (double*)malloc(sizeof(double) * (XUNLIANNUM * CESHIYANZHENGDIM));//训练数据
	double* ceshiData = (double*)malloc(sizeof(double) * CESHINUM * CESHIYANZHENGDIM);//测试数据
	double* yanzhengData = (double*)malloc(sizeof(double) * YANZHENGNUM * CESHIYANZHENGDIM);//验证集数据，这里验证集为训练集前100个
	int* category = (int*)malloc(sizeof(int) * XUNLIANNUM);//训练数据对应所属类别（也就是1/2/3）
	int* categoryOfYanzheng = (int*)malloc(sizeof(int) * YANZHENGNUM);//给出验证集数据，这里验证集为训练集前100个
	double* minValueOfCeshi = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//测试集各个数据的最小值(分别是里程数、时间比、公斤数的最小值)
	double* maxValueOfCeshi = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//测试集各个数据的最大值(分别是里程数、时间比、公斤数的最大值)
	double* minValueOfXunlian = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//训练集各个数据的最小值
	double* maxValueOfXunlian = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//训练集各个数据的最大值
	double* minValue = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//训练集对比测试集各个数据的之后得出最小值
	double* maxValue = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//训练集对比测试集各个数据的之后得出最大值
	double* minValueOfAll = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//每个进程找到的部分数据的最小值规约,取所有数据的最小值
	double* maxValueOfAll = (double*)malloc(sizeof(double) * CESHIYANZHENGDIM);//每个进程找到的部分数据的最大值规约,取所有数据的最大值
	double* xunlianDataGuiyi = (double*)malloc(sizeof(double) * XUNLIANNUM * CESHIYANZHENGDIM);//归一化后的训练数据
	int* result = (int*)malloc(sizeof(int) * CESHINUM);//存放测试集结果
	int* result_yanzheng = (int*)malloc(sizeof(int) * YANZHENGNUM);//验证集结果

	int batch_test = CESHINUM / numprocess;//平分测试集时每个进程应处理的样本个数
	int batch_train = XUNLIANNUM / numprocess;//平分训练集时每个进程应处理的样本个数
	int batch_yanzheng = YANZHENGNUM / numprocess;//平分验证集时每个进程应处理的样本个数

	double* data_test_buffer = (double*)malloc(sizeof(double) * batch_test * CESHIYANZHENGDIM);//用于接收分发的测试集数据
	double* data_yanzheng_buffer = (double*)malloc(sizeof(double) * batch_yanzheng * CESHIYANZHENGDIM);//用于接收分发的验证集数据
	double* ceshiDataGuiyi = (double*)malloc(sizeof(double) * batch_test * CESHIYANZHENGDIM);//归一化后的测试数据
	double* yanzhengDataGuiyi = (double*)malloc(sizeof(double) * (batch_yanzheng * CESHIYANZHENGDIM));//归一化后验证集数据
	int* categoryOfAll = (int*)malloc(sizeof(int) * XUNLIANNUM * batch_test);//将训练数据所属类别共XUNLIANNUM为一块进行batch_test次复制，后面欧式的排序用到
	int* categoryOfYanzheng_all = (int*)malloc(sizeof(int) * XUNLIANNUM * batch_yanzheng);//将验证数据所属类别共XUNLIANNUM为一块进行batch_yanzheng次复制，欧式的排序用到
	double* euclidDistance = (double*)malloc(sizeof(double) * XUNLIANNUM * batch_test);//存放测试集和训练集之间的欧式距离，前XUNLIANNUM个位为第1个测试样本点的，依此类推
	double* distance_yanzheng = (double*)malloc(sizeof(double) * (XUNLIANNUM * YANZHENGNUM));//验证集和训练集的欧式距离
	int* result_buffer = (int*)malloc(sizeof(int) * batch_test);//部分测试集结果
	int* result_yanzheng_buffer = (int*)malloc(sizeof(int) * batch_yanzheng);//部分验证集结果

	if (myid == 0)//0号进程读入训练集数据
	{
		Read(XUNLIANFILENAME, XUNLIANNUM, XUNLIANDIM, category, xunlianData);
	}
	if (myid == 1)//1号进程读入测试集数据
	{
		Read(CESHIFILENAME, CESHINUM, CESHIYANZHENGDIM, category, ceshiData);
	}
	if (ISYANZHENG == 1 && myid == 2)//如果需要验证，2号进程读入验证集数据
	{
		Read(XUNLIANFILENAME, YANZHENGNUM, XUNLIANDIM, categoryOfYanzheng, yanzhengData);
	}

	MPI_Bcast(xunlianData, XUNLIANNUM * CESHIYANZHENGDIM, MPI_DOUBLE, 0, MPI_COMM_WORLD);//0号进程将读取的训练数据广播
	MPI_Bcast(category, XUNLIANNUM, MPI_INT, 0, MPI_COMM_WORLD);//0号进程广播训练数据对应的类别
	MPI_Scatter(ceshiData, batch_test * CESHIYANZHENGDIM, MPI_DOUBLE, data_test_buffer, batch_test * CESHIYANZHENGDIM, MPI_DOUBLE, 1, MPI_COMM_WORLD);//进程1分发测试数据
	if (ISYANZHENG == 1)
	{
		MPI_Scatter(yanzhengData, batch_yanzheng * CESHIYANZHENGDIM, MPI_DOUBLE, data_yanzheng_buffer, batch_yanzheng * CESHIYANZHENGDIM, MPI_DOUBLE, 2, MPI_COMM_WORLD);//进程2分发验证数据
	}

	FindMinMaxValue(batch_train, minValueOfXunlian, maxValueOfXunlian, xunlianData);//找到该进程被分发的训练数据的最值
	FindMinMaxValue(batch_test, minValueOfCeshi, maxValueOfCeshi, data_test_buffer);//找到该进程被分发的测试数据的最值
	FindMinMaxValueOfAll(minValueOfXunlian, maxValueOfXunlian, minValueOfCeshi, maxValueOfCeshi, minValue, maxValue);//通过对比在训练集、测试集中的最值找出它们的最值
	MPI_Allreduce(minValue, minValueOfAll, CESHIYANZHENGDIM, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);//将每个进程找到的部分数据的最小值规约,取所有数据的最小值，并将最小值分发给所有进程
	MPI_Allreduce(maxValue, maxValueOfAll, CESHIYANZHENGDIM, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);//将每个进程找到的部分数据的最大值规约,取所有数据的最大值，并将最大值分发给所有进程

	for (int i = 0; i < XUNLIANNUM; i++)//归一化训练数据
	{
		for (int j = 0; j < CESHIYANZHENGDIM; j++)
		{
			int aim = i * CESHIYANZHENGDIM + j;
			if ((maxValueOfAll[j] - minValueOfAll[j]) != 0)
			{
				xunlianDataGuiyi[aim] = (xunlianData[aim] - minValueOfAll[j]) / (maxValueOfAll[j] - minValueOfAll[j]);
			}
		}
	}
	GuiYiData(batch_test, data_test_buffer, ceshiDataGuiyi, minValueOfAll, maxValueOfAll);//归一化测试集
	CalEuclidDistance(xunlianDataGuiyi, ceshiDataGuiyi, euclidDistance, batch_test);//计算欧式距离

	for (int i = 0;i < batch_test;i++)//以训练数据所属类别共XUNLIANNUM为一块进行batch_test次复制，欧式的排序用到
	{
		for (int j = 0;j < XUNLIANNUM;j++)
		{
			categoryOfAll[i * XUNLIANNUM + j] = category[j];
		}
	}
	SortEuclidDistance(euclidDistance, categoryOfAll, batch_test);//对计算的欧式距离进行排序，注意：是以XUNLIANNUM为一组进行排序的
	CalResult(categoryOfAll, result_buffer, batch_test);//根据K值逐个计算出测试数据的所属类别

	MPI_Gather(result_buffer, batch_test, MPI_INT, result, batch_test, MPI_INT, 1, MPI_COMM_WORLD);//往1号进程聚集各个进程计算的数据的所属类别

	if (ISYANZHENG == 1)
	{
		GuiYiData(batch_yanzheng, data_yanzheng_buffer, yanzhengDataGuiyi, minValueOfAll, maxValueOfAll);//归一化数据
		CalEuclidDistance(xunlianDataGuiyi, yanzhengDataGuiyi, distance_yanzheng, batch_yanzheng);//计算欧式距离
		for (int i = 0;i < batch_yanzheng;i++)//以验证数据所属类别共XUNLIANNUM为一块进行batch_yanzheng次复制，欧式的排序用到
		{
			for (int j = 0;j < XUNLIANNUM;j++)
			{
				categoryOfYanzheng_all[i * XUNLIANNUM + j] = category[j];
			}
		}
		SortEuclidDistance(distance_yanzheng, categoryOfYanzheng_all, batch_yanzheng);///计算欧式距离
		CalResult(categoryOfYanzheng_all, result_yanzheng_buffer, batch_yanzheng);//计算所属类别
		MPI_Gather(result_yanzheng_buffer, batch_yanzheng, MPI_INT, result_yanzheng, batch_yanzheng, MPI_INT, 2, MPI_COMM_WORLD);//往2号进程集聚验证集计算结果
	}

	if (myid == 1)
	{
		Write(result);//往文件输出测试集结果
	}
	if (myid == 2 && ISYANZHENG == 1)
	{
		int num_error = 0;//分类错误的样本个数
		for (int i = 0;i < YANZHENGNUM;i++)//计算错误率
		{
			if (result_yanzheng[i] != category[i])
				num_error++;
		}
		double rate_error = (double)num_error / (double)YANZHENGNUM;
		printf("错误个数：%d\n", num_error);
		printf("验证集错误率为：%lf\n", rate_error);
	}

	free(xunlianData);//释放动态分配数组
	free(ceshiData);
	free(yanzhengData);
	free(category);
	free(categoryOfYanzheng);
	free(minValueOfCeshi);
	free(maxValueOfCeshi);
	free(minValueOfXunlian);
	free(maxValueOfXunlian);
	free(minValue);
	free(maxValue);
	free(minValueOfAll);
	free(maxValueOfAll);
	free(xunlianDataGuiyi);
	free(ceshiDataGuiyi);
	free(yanzhengDataGuiyi);
	free(result);
	free(result_yanzheng);
	free(data_test_buffer);
	free(data_yanzheng_buffer);
	free(categoryOfAll);
	free(categoryOfYanzheng_all);
	free(euclidDistance);
	free(distance_yanzheng);
	free(result_buffer);
	free(result_yanzheng_buffer);

	xunlianData = NULL;//指针指空，避免出现野指针
	ceshiData = NULL;
	yanzhengData = NULL;
	category = NULL;
	categoryOfYanzheng = NULL;
	minValueOfCeshi = NULL;
	maxValueOfCeshi = NULL;
	minValueOfXunlian = NULL;
	maxValueOfXunlian = NULL;
	minValue = NULL;
	maxValue = NULL;
	minValueOfAll = NULL;
	maxValueOfAll = NULL;
	xunlianDataGuiyi = NULL;
	ceshiDataGuiyi = NULL;
	yanzhengDataGuiyi = NULL;
	result = NULL;
	result_yanzheng = NULL;
	data_test_buffer = NULL;
	data_yanzheng_buffer = NULL;
	categoryOfAll = NULL;
	maxValueOfAll = NULL;
	categoryOfYanzheng_all = NULL;
	euclidDistance = NULL;
	distance_yanzheng = NULL;
	result_buffer = NULL;
	result_yanzheng_buffer = NULL;

	MPI_Barrier(MPI_COMM_WORLD);//等待所有进程运行结束
	finish = MPI_Wtime();
	MPI_Finalize();//释放MPI资源
	if (myid == 0)//进程0打印总的运行时间
	{
		printf("Running time of program is %lf seconds\n" (double)finish - start);
	}
	return 0;
}