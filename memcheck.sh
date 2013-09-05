rm output.txt
cuda-memcheck --leak-check full RayTracer > output.txt;
