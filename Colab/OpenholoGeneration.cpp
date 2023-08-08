#include <ophPointCloud.h>
#include <ophDepthMap.h>
#include <ophTriMesh.h>
#include <ophLightField.h>
#include <ophWRP.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void ex_Openholo(const char* conf, const char* input, const char* encod, const char* mode, const char* pro)
{
	printf("====== ex_Openholo START ======\n");
	ophPointCloud* Hologram = new ophPointCloud();
	unsigned int encoding = -1;
	string openholo_enc = "";

	if (0 == strcmp(pro, "0")) {   // pro = 0 --> cpu
		printf("mode is CPU \n");
		bool is_CPU = true;
		Hologram->SetMode(is_CPU);
	}
	else if (0 == strcmp(pro, "1")) { // pro = 1 --> gpu
		if (0 == strcmp(mode, "0")) {  // mode = 0 --> single
			printf("mode is GPU and single \n");
			uint mode;
			mode |= MODE_GPU;
			mode |= MODE_FLOAT;
			mode |= MODE_FASTMATH;
			Hologram->SetMode(mode);
		}
		else if (0 == strcmp(mode, "1")) {  // mode = 1 --> double
			printf("mode is GPU and double \n");
			bool is_GPU = true;
			Hologram->SetMode(is_GPU);
		}
	}

	printf("PointCloud config path :  %s \n", conf);
	printf("PointCloud input path :  %s \n", input);

	if (0 == strcmp(encod, "0")) {   // encod = 0 --> ENCODE_PHASE
		encoding = Hologram->ENCODE_PHASE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_PHASE \n");
		openholo_enc = "Phase";
	}
	else if (0 == strcmp(encod, "1")) {  // encod = 1 --> ENCODE_AMPLITUDE
		encoding = Hologram->ENCODE_AMPLITUDE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_AMPLITUDE \n");
		openholo_enc = "Amplitude";
	}
	else if (0 == strcmp(encod, "2")) {   // encod = 2 --> ENCODE_REAL
		encoding = Hologram->ENCODE_REAL;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_REAL \n");
		openholo_enc = "Real";
	}
	else if (0 == strcmp(encod, "3")) {   // encod = 3 --> ENCODE_IMAGINARY
		encoding = Hologram->ENCODE_IMAGINARY;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_IMAGINEARY \n");
		openholo_enc = "Imaginary";
	}

	Hologram->readConfig(conf);
	Hologram->loadPointCloud(input);
	Hologram->generateHologram(ophPointCloud::PC_DIFF_RS);
	Hologram->encoding(encoding);
	Hologram->normalize();

	char buf[512] = { 0, };
	sprintf(buf, "%s%s%s", "Result/PointCloud_", openholo_enc.c_str(), ".bmp");
	Hologram->save(buf, 24);
	Hologram->release();

	printf("====== ex_Openholo END ======\n");
}

void ex_DepthMap(const char* conf, char* p[3], const char* encod, const char* mode, const char* pro)
{
	printf("====== ex_DepthMap START ======\n");
	ophDepthMap* Hologram = new ophDepthMap();
	unsigned int encoding = -1;
	string depthmap_enc = "";

	if (0 == strcmp(pro, "0")) {   // pro = 0 --> cpu
		printf("mode is CPU \n");
		bool is_CPU = true;
		Hologram->SetMode(is_CPU);
	}
	else if (0 == strcmp(pro, "1")) {   // pro = 1 --> gpu
		if (0 == strcmp(mode, "0")) {  // mode = 0 --> single
			printf("mode is GPU and single \n");
			uint mode;
			mode |= MODE_GPU;
			mode |= MODE_FLOAT;
			mode |= MODE_FASTMATH;

			Hologram->SetMode(mode);
		}
		else if (0 == strcmp(mode, "1")) { // mode = 1 --> double
			printf("mode is GPU and double \n");
			bool is_GPU = true;

			Hologram->SetMode(is_GPU);
		}
	}

	if (0 == strcmp(encod, "0")) {   // encod = 0 --> ENCODE_PHASE
		encoding = ophGen::ENCODE_PHASE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_PHASE \n");
		depthmap_enc = "Phase";
	}
	else if (0 == strcmp(encod, "1")) {   // encod = 1 --> ENCODE_AMPLITUDE
		encoding = ophGen::ENCODE_AMPLITUDE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_AMPLITUDE \n");
		depthmap_enc = "Amplitude";
	}
	else if (0 == strcmp(encod, "2")) {  // encod = 2 --> ENCODE_REAL
		encoding = ophGen::ENCODE_REAL;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_REAL \n");
		depthmap_enc = "Real";
	}
	else if (0 == strcmp(encod, "3")) {   // encod = 3 --> ENCODE_IMAGINARY
		encoding = ophGen::ENCODE_IMAGINARY;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_IMAGINEARY \n");
		depthmap_enc = "Imaginary";
	}

	Hologram->readConfig(conf);	 // Read Config Parameters for Depth Map CGH]
	Hologram->readImageDepth(p[0], p[1], p[2]);   // Load Depth and RGB image
	int nChannel = Hologram->getContext().waveNum;
	Hologram->generateHologram();
	Hologram->encoding(encoding);
	Hologram->normalize();

	char buf[512] = { 0, };
	sprintf(buf, "%s%s%s", "Result/DepthMap_", depthmap_enc.c_str(), ".bmp");
	Hologram->save(buf, nChannel * 8);

	Hologram->release();

	printf("====== ex_DepthMap END ======\n");
}

void ex_LightField(const char* conf, const char* input, const char* encod, const char* mode, const char* pro)
{
	printf("====== ex_LightField START ======\n");
	ophLF* Hologram = new ophLF();
	unsigned int encoding = -1;
	string lightfield_enc = "";

	if (0 == strcmp(pro, "0")) {   // pro = 0 --> cpu
		printf("mode is CPU \n");
		bool is_CPU = true;
		Hologram->SetMode(is_CPU);
	}
	else if (0 == strcmp(pro, "1")) { // pro = 1 --> gpu
		if (0 == strcmp(mode, "0")) {  // mode = 0 --> single
			printf("mode is GPU and single \n");
			uint mode;
			mode |= MODE_GPU;
			mode |= MODE_FLOAT;
			mode |= MODE_FASTMATH;
			Hologram->SetMode(mode);
		}
		else if (0 == strcmp(mode, "1")) {  // mode = 1 --> double
			printf("mode is GPU and double \n");
			bool is_GPU = true;
			Hologram->SetMode(is_GPU);
		}
	}

	printf("LightField config path :  %s \n", conf);
	printf("LightField input path :  %s \n", input);

	if (0 == strcmp(encod, "0")) {   // encod = 0 --> ENCODE_PHASE
		encoding = Hologram->ENCODE_PHASE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_PHASE \n");
		lightfield_enc = "Phase";
	}
	else if (0 == strcmp(encod, "1")) {  // encod = 1 --> ENCODE_AMPLITUDE
		encoding = Hologram->ENCODE_AMPLITUDE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_AMPLITUDE \n");
		lightfield_enc = "Amplitude";
	}
	else if (0 == strcmp(encod, "2")) {   // encod = 2 --> ENCODE_REAL
		encoding = Hologram->ENCODE_REAL;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_REAL \n");
		lightfield_enc = "Real";
	}
	else if (0 == strcmp(encod, "3")) {   // encod = 3 --> ENCODE_IMAGINARY
		encoding = Hologram->ENCODE_IMAGINARY;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_IMAGINEARY \n");
		lightfield_enc = "Imaginary";
	}

	Hologram->readConfig(conf); // "Config/Generation_PointCloud (RGB).xml"
	Hologram->loadLF(input, "bmp"); // "PointCloud & WRP/pointcloud_1470.ply"
	Hologram->generateHologram();
	((ophGen*)Hologram)->encoding(encoding);
	Hologram->normalize();
	ivec2 encode_size = Hologram->getEncodeSize();
	char buf[512] = { 0, };
	sprintf(buf, "%s%s%s", "Result/LightField_", lightfield_enc.c_str(), ".bmp");
	Hologram->save(buf, 8 * Hologram->getContext().waveNum, nullptr, encode_size[_X], encode_size[_Y]);
	Hologram->release();

	printf("====== ex_LightField END ======\n");
}

void ex_TriMesh(const char* conf, const char* input, const char* encod, const char* mode, const char* pro)
{
	printf("====== ex_TriMesh START ======\n");
	ophTri* Hologram = new ophTri();
	unsigned int encoding = -1;
	string trimesh_enc = "";

	if (0 == strcmp(pro, "0")) {   // pro = 0 --> cpu
		printf("mode is CPU \n");
		bool is_CPU = true;
		Hologram->SetMode(is_CPU);
	}
	else if (0 == strcmp(pro, "1")) { // pro = 1 --> gpu
		if (0 == strcmp(mode, "0")) {  // mode = 0 --> single
			printf("mode is GPU and single \n");
			uint mode;
			mode |= MODE_GPU;
			mode |= MODE_FLOAT;
			mode |= MODE_FASTMATH;
			Hologram->SetMode(mode);
		}
		else if (0 == strcmp(mode, "1")) {  // mode = 1 --> double
			printf("mode is GPU and double \n");
			bool is_GPU = true;
			Hologram->SetMode(is_GPU);
		}
	}

	printf("TriMesh config path :  %s \n", conf);
	printf("TriMesh input path :  %s \n", input);

	if (0 == strcmp(encod, "0")) {   // encod = 0 --> ENCODE_PHASE
		encoding = Hologram->ENCODE_PHASE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_PHASE \n");
		trimesh_enc = "Phase";
	}
	else if (0 == strcmp(encod, "1")) {  // encod = 1 --> ENCODE_AMPLITUDE
		encoding = Hologram->ENCODE_AMPLITUDE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_AMPLITUDE \n");
		trimesh_enc = "Amplitude";
	}
	else if (0 == strcmp(encod, "2")) {   // encod = 2 --> ENCODE_REAL
		encoding = Hologram->ENCODE_REAL;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_REAL \n");
		trimesh_enc = "Real";
	}
	else if (0 == strcmp(encod, "3")) {   // encod = 3 --> ENCODE_IMAGINARY
		encoding = Hologram->ENCODE_IMAGINARY;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_IMAGINEARY \n");
		trimesh_enc = "Imaginary";
	}

	Hologram->readConfig(conf);
	Hologram->loadMeshData(input, "ply");
	Hologram->generateHologram(Hologram->SHADING_FLAT);
	Hologram->encoding(encoding);
	Hologram->normalize();
	ivec2 encode_size = Hologram->getEncodeSize();

	char buf[512] = { 0, };
	sprintf(buf, "%s%s%s", "Result/TriMesh_", trimesh_enc.c_str(), ".bmp");
	Hologram->save(buf, 8 * Hologram->getContext().waveNum, nullptr, encode_size[_X], encode_size[_Y]);
	Hologram->release();

	printf("====== TriMesh END ======\n");
}

void ex_WRP(const char* conf, const char* input, const char* encod, const char* mode, const char* pro)
{
	printf("====== ex_WRP START ======\n");
	ophWRP* Hologram = new ophWRP();
	unsigned int encoding = -1;
	string wrp_enc = "";

	if (0 == strcmp(pro, "0")) {   // pro = 0 --> cpu
		printf("mode is CPU \n");
		bool is_CPU = true;
		Hologram->SetMode(is_CPU);
	}
	else if (0 == strcmp(pro, "1")) { // pro = 1 --> gpu
		if (0 == strcmp(mode, "0")) {  // mode = 0 --> single
			printf("mode is GPU and single \n");
			uint mode;
			mode |= MODE_GPU;
			mode |= MODE_FLOAT;
			mode |= MODE_FASTMATH;
			Hologram->SetMode(mode);
		}
		else if (0 == strcmp(mode, "1")) {  // mode = 1 --> double
			printf("mode is GPU and double \n");
			bool is_GPU = true;
			Hologram->SetMode(is_GPU);
		}
	}

	printf("WRP config path :  %s \n", conf);
	printf("WRP input path :  %s \n", input);

	if (0 == strcmp(encod, "0")) {   // encod = 0 --> ENCODE_PHASE
		encoding = Hologram->ENCODE_PHASE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_PHASE \n");
		wrp_enc = "Phase";
	}
	else if (0 == strcmp(encod, "1")) {  // encod = 1 --> ENCODE_AMPLITUDE
		encoding = Hologram->ENCODE_AMPLITUDE;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_AMPLITUDE \n");
		wrp_enc = "Amplitude";
	}
	else if (0 == strcmp(encod, "2")) {   // encod = 2 --> ENCODE_REAL
		encoding = Hologram->ENCODE_REAL;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_REAL \n");
		wrp_enc = "Real";
	}
	else if (0 == strcmp(encod, "3")) {   // encod = 3 --> ENCODE_IMAGINARY
		encoding = Hologram->ENCODE_IMAGINARY;
		printf("encoding :  %d \n", encoding);
		printf("ENCODE_IMAGINEARY \n");
		wrp_enc = "Imaginary";
	}

	Hologram->readConfig(conf); // "Config/Generation_PointCloud (RGB).xml"
	Hologram->loadPointCloud(input); // "PointCloud & WRP/pointcloud_1470.ply"
	Hologram->generateHologram();
	((ophGen*)Hologram)->encoding(encoding);
	Hologram->normalize();

	char buf[512] = { 0, };
	ivec2 encode_size = Hologram->getEncodeSize();
	sprintf(buf, "%s%s%s", "Result/WRP_", wrp_enc.c_str(), ".bmp");
	Hologram->save(buf, 8 * Hologram->getContext().waveNum, nullptr, encode_size[_X], encode_size[_Y]);
	Hologram->release();

	printf("====== ex_WRP END ======\n");
}

int main(int argc, char* argv[])
{
	printf("====== Openholo Test main V0.1 ======\n");
	string str1 = "-a";
	string str2 = "-c";
	string str3 = "-i";
	string str4 = "-e";
	string str5 = "-m";
	string str6 = "-p";

	char config[128] = { 0, };
	char input[128] = { 0, };
	const char* alg = "";
	char* p[3];
	const char* encod = "";
	const char* mode = "";
	const char* pro = "";


	printf("argc :  %d ", argc);

	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(str1.c_str(), argv[i])) {
			alg = argv[i + 1];
			printf("alg = %s \n", alg);
		}
		else if (0 == strcmp(str2.c_str(), argv[i])) {
			strcpy(config, argv[i + 1]);
			printf("config = %s \n", config);
		}
		else if (0 == strcmp(str3.c_str(), argv[i])) {
			if (0 == strcmp(alg, "1")) {
				for (int j = 0; j < 3; j++) {
					p[j] = new char[12 * sizeof(argv[i + 1])];
					printf("p = %s \n", argv[i + j + 1]);
					strcpy(p[j], argv[i + j + 1]);
				}
			}
			else if (0 == strcmp(alg, "0")) {
				strcpy(input, argv[i + 1]);
				printf("PointCloud_input = %s \n", input);
			}
			else if (0 == strcmp(alg, "2")) {
				strcpy(input, argv[i + 1]);
				printf("LightField_input = %s \n", input);
			}
			else if (0 == strcmp(alg, "3")) {
				strcpy(input, argv[i + 1]);
				printf("TriMesh_input = %s \n", input);
			}
			else {   // alg == 4
				strcpy(input, argv[i + 1]);
				printf("WRP_input = %s \n", input);
			}
		}
		else if (0 == strcmp(str4.c_str(), argv[i])) {
			encod = argv[i + 1];
			printf("encod = %s \n", encod);
		}
		else if (0 == strcmp(str5.c_str(), argv[i])) {
			mode = argv[i + 1];
			printf("mode = %s \n", mode);
		}
		else if (0 == strcmp(str6.c_str(), argv[i])) {
			pro = argv[i + 1];
			printf("pro = %s \n", pro);
		}
	}

	unsigned int select = atoi(alg);
	if (select == 0) {
		ex_Openholo(config, input, encod, mode, pro);
	}
	else if (select == 1) {
		ex_DepthMap(config, p, encod, mode, pro);
		for (int i = 0; i < 3; i++)
		{
			delete[] p[i];
		}
	}
	else if (select == 2) {
		ex_LightField(config, input, encod, mode, pro);
	}
	else if (select == 3) {
		ex_TriMesh(config, input, encod, mode, pro);
	}
	else if (select == 4) {
		ex_WRP(config, input, encod, mode, pro);
	}
	else {
		printf("Invalid algorithm selected. \n");
	}

	printf("====== Openholo Test main V0.1 END ======\n");
}
