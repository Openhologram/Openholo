# Openholo
C/C++ Based Openholo library solution

***
## Check GPUs supported
https://en.wikipedia.org/wiki/CUDA

***
## Setting up a CUDA environment

### 1. Select and download the required version of the CUDA toolkit.
Link: https://developer.nvidia.com/cuda-toolkit-archive

![image](https://github.com/Openhologram/Openholo/assets/54168276/3e78a0e7-f244-4018-8e2e-ec7059ceeebb)

![image](https://github.com/Openhologram/Openholo/assets/54168276/d5c54e88-a0bc-469d-8d88-b66ed786b65a)

### 2. Install the CUDA toolkit. (Visual Studio should be shut down)

### 3. Set the environment variable 'CUDA_PATH'. 
![image](https://github.com/Openhologram/Openholo/assets/54168276/e4a32054-ea03-441f-989b-3765285408db)

### 4. Find and copy the files listed below in your installed CUDA toolkit path.
Major Version: A
Minor Version: B
- CUDA A.B.props
- CUDA A.B.targets
- CUDA A.B.xml
- Nvda.Build.CudaTasks.vA.B.dll

ex) Cuda 11.6 toolkit
CUDA 11.6.props and ...

### 5. Find and paste 'BuildCustomizations' in your installed visual studio path.
   => Depending on your version of Visual Studio, the path may be different.

### 6. Open the project file, and check the build dependencies.
![image](https://github.com/Openhologram/Openholo/assets/54168276/2131656e-7f66-41bf-83e1-56b19d0faf83)

![image](https://github.com/Openhologram/Openholo/assets/54168276/d6507e1e-1651-415a-821d-17a23112ee61)
