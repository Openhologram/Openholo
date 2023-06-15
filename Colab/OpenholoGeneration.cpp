#include <ophPointCloud.h>
#include <ophDepthMap.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void ex_Openholo(const char* conf, const char* input)
{
	printf("====== ex_Openholo START ======\n");
	//bool is_GPU = true;
	//ophPointCloud *Hologram = new ophPointCloud();
	//Hologram->SetMode(is_GPU);
	uint mode;
	mode |= MODE_GPU;
	mode |= MODE_FLOAT;
	mode |= MODE_FASTMATH;
	ophPointCloud *Hologram = new ophPointCloud();
	Hologram->SetMode(mode);



	printf("config path :  %s \n", conf);
	printf("input path :  %s \n", input);
	
	Hologram->readConfig(conf); // "Config/Generation_PointCloud (RGB).xml"
	Hologram->loadPointCloud(input); // "PointCloud & WRP/pointcloud_1470.ply"
	Hologram->generateHologram(ophPointCloud::PC_DIFF_RS);
	Hologram->encoding(ophGen::ENCODE_PHASE);
	Hologram->normalize();

	char buf[512] = {0,};
	sprintf(buf, "Result/PointCloud_phase.bmp");
	Hologram->save(buf, 24);
	Hologram->release();

	printf("====== ex_Openholo END ======\n");
}

void ex_DepthMap(const char* conf, char* p[3])
{
	printf("====== ex_DepthMap START ======\n");
	//bool is_GPU = true;
	//ophDepthMap* Hologram = new ophDepthMap();
	//Hologram->SetMode(is_GPU);
	uint mode;
	mode |= MODE_GPU;
	mode |= MODE_FLOAT;
	mode |= MODE_FASTMATH;
	ophDepthMap *Hologram = new ophDepthMap();
	Hologram->SetMode(mode);


	printf("config path :  %s \n", conf);
	printf("readImageDepth p[0] :  %s \n", p[0]);
	printf("readImageDepth p[1] :  %s \n", p[1]);
	printf("readImageDepth p[2] :  %s \n", p[2]);

	Hologram->readConfig(conf);	 // Read Config Parameters for Depth Map CGH]
	Hologram->readImageDepth(p[0], p[1], p[2]);   // Load Depth and RGB image
	int nChannel = Hologram->getContext().waveNum;
	Hologram->generateHologram();
	Hologram->encoding(ophGen::ENCODE_PHASE);
	Hologram->normalize();

	char buf[512] = {0,};
	sprintf(buf, "Result/DepthMap_phase.bmp");
	Hologram->save(buf, nChannel * 8);

	Hologram->release();

	printf("====== ex_DepthMap END ======\n");
}

int main(int argc, char* argv[])
{
	printf("====== Openholo Test main V0.1 ======\n");
	string str1 = "-a";
	string str2 = "-c";
	string str3 = "-i";
	
	char config[128] = {0,};
	char input[128] = {0,};
	const char* alg = "";
	char* p[3];
	

  printf("argc :  %d ", argc);

	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(str1.c_str(), argv[i])){
			alg = argv[i+1];
			printf("alg = %s \n", alg);
		} else if (0 == strcmp(str2.c_str(), argv[i])) {
			strcpy(config, argv[i + 1]);
			printf("config = %s \n", config);
		} else if (0 == strcmp(str3.c_str(), argv[i])) {
			if (0 == strcmp(alg, "1")){
				for (int j = 0; j < 3; j++) {
					p[j] = new char[12 * sizeof(argv[i + 1])];
					printf("p = %s \n", argv[i + j + 1]);
					strcpy(p[j], argv[i + j + 1]);
				}
			}
			else {
				strcpy(input, argv[i + 1]);
				printf("point_input = %s \n", input);
			}
		} 
	}

	unsigned int select = atoi(alg)+1;
	if (select == 1) {
		ex_Openholo(config, input);
	} else {
		ex_DepthMap(config, p);
		for (int i = 0; i < 3; i++)
		{
			delete[] p[i];
		}
	}

	

	printf("====== Openholo Test main V0.1 END ======\n");
}
