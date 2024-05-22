# AcceleratedAffinityPropagationClustering
## About
 Affinity Propagation Clustering Algorithm with CUDA acceleration.

 + Code compares the CPU implementation (ApcCpu) with two GPU implementations (ApcGpu and ApcGpuV2).
 + ApcGpuV2 is the one used for the term paper.
 + The cluster information for all points are stored in files specified in the terminal output. You can check the diff between to see they are the same.
 + A Python script (ApcPlot.py) is also included for plotting from the files. It requires matplotlib.

 Important Note: If you decide to load points from a file, you *MUST* change the *POINT_COUNT* and *POINT_DIM* parameters accordingly.


## Parameters
 You can change the parameters between main.cu:12 and main.cu:23 to see varying behaviour.

 + *GENERATE_POINTS* (default true): Boolean value. If true, generates random points for the algorithm. If false, the code uses the file with FILENAME as its input.
 + *POINT_COUNT*: The point count that will be clustered. Do not go above 500 for practical CPU execution times. (SET EVEN IF READING FROM FILE)
 + *POINT_DIM*: Dimension of the points. Default value is 2. (SET EVEN IF READING FROM FILE) 
 + *FILENAME*: Default value is "input.txt". Has 100 2D points for testing. 
 + *ITERATION_COUNT*: The number of iterations the algorithm goes through. For practicality, values above 100 are not recommended.

 All of the values below can be set to true or false independently.
 + *USE_GPU_V2*: Default true. Use the *best* GPU implementation.
 + *USE_CPU*: Default true. Uses the CPU implementation
 + *USE_GPU*: Default false. Uses *naive* GPU implementation. Not critical.