#ifndef CUDAKERNELS_H_
#define CUDAKERNELS_H_
#include <vector>
#include <iostream>
// //refer to 'GPU-SME- k NN: Scalable and memory efficient k NN and lazy learning using GPUs'
// //I change the algorithm to compute batch data
// /* Computes a chunk of the distance matrix */
// __global__ void distanceKernel(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, float* matrix, int chunkTrain, int chunkTest, int selections);
// /* Performs the quicksort-based selection on a chunk of the distance matrix */
// __global__ void quickSelection(float* globalRows, int* testTrainSizes, int testSize, int trainChunk, int testChunk, long int* returnPossitions, float* returnValues, int selections, float* quickRows, long int* quickActualIndex, long int* quickNextIndex, float* quickRowsRight, long int* quickIndexRight);
// /* Performs the SQRT operation on the neighborhood distances and copy the index values to their final structures */
// __global__ void cuSqrtAndCopy(float* valuesSource, long int* indexSource, float* valuesDestinationAll, long int* indexDestinationAll, int testChunk, int testSize, int totalValues, int iterations);
// //training(batch*nVariables*trainSize),test(nVariables*testSize),testSize is all test points number sum
// //testTrainBatchIds(testSize),testTrainSizes(testSize)
// void batchKnn(float* training, float* test, int* testTrainBatchIds, int* testTrainSizes, int nVariables, int trainSize, int testSize, int k, float* testKnnDistances, long int* testKnnIndexs);
class MCGpu{
public:
	~MCGpu();
	static MCGpu& Get(int device_id);	
	void MC(float* d_sdf_, float fTargetValue=0.0);
	bool init(int nx,int ny, int nz);
	void scaleVertices(float xstep,float ystep,float zstep,float xmin,float ymin,float zmin);
	std::vector<int> number_record_;
	float* d_points_coor_;
	long int* d_faces_index_;
private:
	MCGpu(int device_id);	
	int NX,NY,NZ,device_id_,mx,my,mz;
	// for multi gpu initial
	float* d_a2fVertexOffset_;
	int* d_a2iEdgeConnection_;
	float* d_a2fEdgeDirection_;
	int* d_aiCubeEdgeFlags_;
	int* d_a2iTriangleConnectionTable_;
	
	int* d_number_record_;
	int* d_edge_point_state_;	
	int* d_faces_ijkd_;
};
#endif /* CUDAKERNELS_H_ */