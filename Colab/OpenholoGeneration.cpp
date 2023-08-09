#include <ophPointCloud.h>
#include <ophDepthMap.h>
#include <ophTriMesh.h>
#include <ophLightField.h>
#include <ophWRP.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <limits.h>

string encodeToString(unsigned int enc)
{
	string name;
	switch (enc)
	{
	case 0: name = "Phase"; break;
	case 1: name = "Amplitude"; break;
	case 2: name = "Real"; break;
	case 3: name = "Imaginary"; break;
	default: name = "Unknown";
	}
	return name;
}

void ex_PointCloud(const char* conf, const char* input, unsigned int flag, unsigned int encode, unsigned int mode)
{
	printf("===== ex_PointCloud START =====\n");
	string diff = (flag == ophPointCloud::PC_DIFF_RS) ? "R-S" : "Fresnel";
	ophPointCloud* Hologram = new ophPointCloud();
	Hologram->SetMode(mode);	
	Hologram->readConfig(conf);
	Hologram->loadPointCloud(input);
	Hologram->generateHologram(flag);
	Hologram->encoding(encode);
	Hologram->normalize();

	char buf[PATH_MAX] = { 0, };
	sprintf(buf, "Result/PointCloud_%s_%s.bmp", diff.c_str(), encodeToString(encode).c_str());
	Hologram->save(buf, Hologram->getContext().waveNum * 8);
	Hologram->release();
	printf("====== ex_PointCloud END ======\n");
}

void ex_DepthMap(const char* conf, char* p[3], unsigned int encode, unsigned int mode)
{
	printf("====== ex_DepthMap START ======\n");
	ophDepthMap* Hologram = new ophDepthMap();
	Hologram->SetMode(mode);
	Hologram->readConfig(conf);	 // Read Config Parameters for Depth Map CGH]
	Hologram->readImageDepth(p[0], p[1], p[2]);   // Load Depth and RGB image
	Hologram->generateHologram();
	Hologram->encoding(encode);
	Hologram->normalize();

	char buf[PATH_MAX] = { 0, };
	sprintf(buf, "Result/DepthMap_%s.bmp", encodeToString(encode).c_str());
	Hologram->save(buf, Hologram->getContext().waveNum * 8);
	Hologram->release();
	printf("====== ex_DepthMap END ======\n");
}

void ex_LightField(const char* conf, const char* input, unsigned int encode, unsigned int mode)
{
	printf("====== ex_LightField START ======\n");
	ophLF* Hologram = new ophLF();
	Hologram->SetMode(mode);	
	Hologram->readConfig(conf); // "Config/Generation_PointCloud (RGB).xml"
	Hologram->loadLF(input, "bmp"); // "PointCloud & WRP/pointcloud_1470.ply"
	Hologram->generateHologram();
	((ophGen*)Hologram)->encoding(encode);
	Hologram->normalize();

	char buf[PATH_MAX] = { 0, };
	sprintf(buf, "Result/LightField_%s.bmp", encodeToString(encode).c_str());
	Hologram->save(buf, Hologram->getContext().waveNum * 8);
	Hologram->release();
	printf("====== ex_LightField END ======\n");
}

void ex_TriMesh(const char* conf, const char* input, unsigned int encode, unsigned int mode)
{
	printf("====== ex_TriMesh START ======\n");
	ophTri* Hologram = new ophTri();
	Hologram->SetMode(mode);
	Hologram->readConfig(conf);
	Hologram->loadMeshData(input, "ply");
	Hologram->generateHologram(Hologram->SHADING_FLAT);
	Hologram->encoding(encode);
	Hologram->normalize();

	char buf[PATH_MAX] = { 0, };
	sprintf(buf, "Result/TriMesh_%s.bmp", encodeToString(encode).c_str());
	Hologram->save(buf, Hologram->getContext().waveNum * 8);
	Hologram->release();

	printf("====== ex_TriMesh END ======\n");
}

void ex_WRP(const char* conf, const char* input, unsigned int encode, unsigned int mode)
{
	printf("====== ex_WRP START ======\n");
	ophWRP* Hologram = new ophWRP();
	Hologram->SetMode(mode);
	Hologram->readConfig(conf); // "Config/Generation_PointCloud (RGB).xml"
	Hologram->loadPointCloud(input); // "PointCloud & WRP/pointcloud_1470.ply"
	Hologram->generateHologram();
	((ophGen*)Hologram)->encoding(encode);
	Hologram->normalize();

	char buf[PATH_MAX] = { 0, };
	sprintf(buf, "Result/WRP_%s.bmp", encodeToString(encode).c_str());
	Hologram->save(buf, Hologram->getContext().waveNum * 8);
	Hologram->release();

	printf("====== ex_WRP END ======\n");
}

int main(int argc, char* argv[])
{
	printf("====== Openholo Test main V0.1 ======\n");
	char config[PATH_MAX] = { 0, };
	char input[PATH_MAX] = { 0, };
	char* p[3];

	int alg = 0;
	int flag = 0;
	int encode = 0;
	int mode = 0;

	printf("argc :  %d ", argc);

	for (int i = 1; i < argc; i++) {
		if (!strcmp("-a", argv[i])) { // check algorithm
			alg = atoi(argv[i + 1]);
			printf("alg = %d \n", alg);
		}
		else if (!strcmp("-c", argv[i])) { // check config path
			strcpy(config, argv[i + 1]);
			printf("config = %s \n", config);
		}
		else if (!strcmp("-i", argv[i])) { // check input path
			if (alg == 1) { // if, depthmap
				for (int j = 0; j < 3; j++) {
					p[j] = new char[12 * sizeof(argv[i + 1])];
					strcpy(p[j], argv[i + j + 1]);
				}
			}
			else if (alg >= 0 && alg < 5) {
				strcpy(input, argv[i + 1]);
			}
			else {
				printf("Invalid input path.\n");
				break;
			}
		}
		else if (!strcmp("-e", argv[i])) { // check encode method
			encode = atoi(argv[i + 1]);
			printf("encode = %u \n", encode);
		}
		else if (!strcmp("-m", argv[i])) { // check precision
			if (atoi(argv[i + 1]) == 0)
			{
				mode |= MODE_FLOAT;
				mode |= MODE_FASTMATH;
			}
			else if (atoi(argv[i + 1]) == 1)
				mode |= MODE_DOUBLE;
			printf("precision = %d\n", atoi(argv[i + 1]));
		}
		else if (!strcmp("-p", argv[i])) { // check mode
			if (atoi(argv[i + 1]) == 0)
				mode |= MODE_CPU;
			else if (atoi(argv[i + 1]) == 1)
				mode |= MODE_GPU;
			printf("mode = %d\n", atoi(argv[i + 1]));
		}
		else if (!strcmp("-f", argv[i])) { // check flag
			flag = atoi(argv[i + 1]);
			printf("flag = %d\n", flag);
		}
	}

	unsigned int select = (unsigned int)alg;
	switch (select)
	{
	case 0: ex_PointCloud(config, input, flag, encode, mode); break;
	case 1: ex_DepthMap(config, p, encode, mode); break;
	case 2: ex_LightField(config, input, encode, mode); break;
	case 3: ex_TriMesh(config, input, encode, mode); break;
	case 4: ex_WRP(config, input, encode, mode); break;
	default: printf("Invalid algorithm selected.\n");
	}
	printf("====== Openholo Test main V0.1 END ======\n");
}
