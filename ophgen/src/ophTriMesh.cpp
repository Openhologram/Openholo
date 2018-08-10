#include "ophTriMesh.h"

#include "sys.h"
#include "tinyxml2.h"
#include "PLYparser.h"

#define for_i(iter, oper)	for(int i=0;i<iter;i++){oper}

#define _X1 0
#define _Y1 1
#define _Z1 2
#define _X2 3
#define _Y2 4
#define _Z2 5
#define _X3 6
#define _Y3 7
#define _Z3 8

uint ophTri::loadMeshText(const char* fileName) {

	cout << "Mesh Text File Load..." << endl;

	ifstream file;
	file.open(fileName);

	if (!file) {
		cout << "Open failed - no such file" << endl;
		cin.get();
		return 0;
	}

	triMeshArray = new Real[9 * 10000];

	Real data;
	uint num_data;

	num_data = 0;
	do {
		file >> data;
		triMeshArray[num_data] = data;
		num_data++;
	} while (file.get() != EOF);

	meshData->n_faces = num_data / 9;
	triMeshArray[meshData->n_faces * 9] = EOF;

	return 1;
}

void ophTri::loadMeshData(const char* fileName, const char* ext) {
	meshData = new OphMeshData;
	cout << "ext = " << ext << endl;

	if (!strcmp(ext,"txt")) {
		cout << "Text File.." << endl;
		if (loadMeshText(fileName))
			cout << "Mesh Data Load Finished.." << endl;
		else
			cout << "Mesh Data Load Failed.." << endl;
	}
	else if (!strcmp(ext,"ply")) {
		cout << "PLY File.." << endl;
		PLYparser meshPLY;
		if (meshPLY.loadPLY(fileName, meshData->n_faces, meshData->color_channels, &meshData->face_idx, &meshData->vertex, &meshData->color))
			cout << "Mesh Data Load Finished.." << endl;
		else
			cout << "Mesh Data Load Failed.." << endl;
	}
	else {
		cout << "Error: Mesh data must be .txt or .ply" << endl;
	}
	meshData->n_faces /= 3;
	triMeshArray = meshData->vertex;
}

int ophTri::readMeshConfig(const char* mesh_config) {
	LOG("Reading....%s...", mesh_config);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(mesh_config, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(mesh_config);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", mesh_config);
		return false;
	}

	xml_node = xml_doc.FirstChild();

#if REAL_IS_DOUBLE & true
	(xml_node->FirstChildElement("ObjectSize"))->QueryDoubleText(&objSize);
	(xml_node->FirstChildElement("ObjectShiftX"))->QueryDoubleText(&objShift[_X]);
	(xml_node->FirstChildElement("ObjectShiftY"))->QueryDoubleText(&objShift[_Y]);
	(xml_node->FirstChildElement("ObjectShiftZ"))->QueryDoubleText(&objShift[_Z]);
	//(xml_node->FirstChildElement("CarrierWaveVectorX"))->QueryDoubleText(&carrierWave[_X]);
	//(xml_node->FirstChildElement("CarrierWaveVectorY"))->QueryDoubleText(&carrierWave[_Y]);
	//(xml_node->FirstChildElement("CarrierWaveVectorZ"))->QueryDoubleText(&carrierWave[_Z]);
	(xml_node->FirstChildElement("LampDirectionX"))->QueryDoubleText(&illumination[_X]);
	(xml_node->FirstChildElement("LampDirectionY"))->QueryDoubleText(&illumination[_Y]);
	(xml_node->FirstChildElement("LampDirectionZ"))->QueryDoubleText(&illumination[_Z]);
	(xml_node->FirstChildElement("SLMPixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMPixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("WavelengthofLaser"))->QueryDoubleText(&context_.lambda);
#else
	(xml_node->FirstChildElement("ObjectSize"))->QueryFloatText(&objSize);
	(xml_node->FirstChildElement("ObjectShiftX"))->QueryFloatText(&objShift[_X]);
	(xml_node->FirstChildElement("ObjectShiftY"))->QueryFloatText(&objShift[_Y]);
	(xml_node->FirstChildElement("ObjectShiftZ"))->QueryFloatText(&objShift[_Z]);
	//(xml_node->FirstChildElement("CarrierWaveVectorX"))->QueryFloatText(&carrierWave[_X]);
	//(xml_node->FirstChildElement("CarrierWaveVectorY"))->QueryFloatText(&carrierWave[_Y]);
	//(xml_node->FirstChildElement("CarrierWaveVectorZ"))->QueryFloatText(&carrierWave[_Z]);
	(xml_node->FirstChildElement("LampDirectionX"))->QueryFloatText(&illumination[_X]);
	(xml_node->FirstChildElement("LampDirectionY"))->QueryFloatText(&illumination[_Y]);
	(xml_node->FirstChildElement("LampDirectionZ"))->QueryFloatText(&illumination[_Z]);
	(xml_node->FirstChildElement("SLMPixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMPixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("WavelengthofLaser"))->QueryFloatText(&context_.lambda);
#endif
	(xml_node->FirstChildElement("SLMPixelNumX"))->QueryIntText(&context_.pixel_number[_X]);
	(xml_node->FirstChildElement("SLMPixelNumY"))->QueryIntText(&context_.pixel_number[_Y]);
	//(xml_node->FirstChildElement("MeshShadingType"))->QueryIntText(&SHADING_TYPE);
	//(xml_node->FirstChildElement("EncodingMethod"))->QueryIntText(&ENCODE_METHOD);
	//if (ENCODE_METHOD == ENCODE_SSB || ENCODE_METHOD == ENCODE_OFFSSB)
	//	(xml_node->FirstChildElement("SingleSideBandPassBand"))->QueryIntText(&SSB_PASSBAND);

	context_.k = (2 * M_PI) / context_.lambda;
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];

	cout << "pixel num: " << context_.pixel_number[_X] << ", " << context_.pixel_number[_Y] << endl;
	cout << "pixel pit: " << context_.pixel_pitch[_X] << ", " << context_.pixel_pitch[_Y] << endl;
	cout << "lambda: " << context_.lambda << endl;
	cout << "illu: " << illumination[_X] << ", " << illumination[_Y] << ", " << illumination[_Z] << endl;
	cout << "size: " << objSize << endl;
	cout << "shift: " << objShift[_X] << ", " << objShift[_Y] << ", " << objShift[_Z] << endl;

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return true;
}


void ophTri::initializeAS() {
	angularSpectrum = new Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	memset(angularSpectrum, 0, context_.pixel_number[_X] * context_.pixel_number[_Y]);
}


void ophTri::objNormCenter() {

	normalizedMeshData = new Real[meshData->n_faces * 9];

	Real* x_point = new Real[meshData->n_faces * 3];
	Real* y_point = new Real[meshData->n_faces * 3];
	Real* z_point = new Real[meshData->n_faces * 3];

	for_i(meshData->n_faces * 3,
		*(x_point + i) = *(triMeshArray + 3 * i);
	*(y_point + i) = *(triMeshArray + 3 * i + 1);
	*(z_point + i) = *(triMeshArray + 3 * i + 2);
	);
	Real x_cen = (maxOfArr(x_point, meshData->n_faces * 3) + minOfArr(x_point, meshData->n_faces * 3)) / 2;
	Real y_cen = (maxOfArr(y_point, meshData->n_faces * 3) + minOfArr(y_point, meshData->n_faces * 3)) / 2;
	Real z_cen = (maxOfArr(z_point, meshData->n_faces * 3) + minOfArr(z_point, meshData->n_faces * 3)) / 2;

	Real* centered = new Real[meshData->n_faces * 9];

	for_i(meshData->n_faces * 3,
		*(centered + 3 * i) = *(x_point + i) - x_cen;
	*(centered + 3 * i + 1) = *(y_point + i) - y_cen;
	*(centered + 3 * i + 2) = *(z_point + i) - z_cen;
	);

	Real x_del = (maxOfArr(x_point, meshData->n_faces * 3) - minOfArr(x_point, meshData->n_faces * 3));
	Real y_del = (maxOfArr(y_point, meshData->n_faces * 3) - minOfArr(y_point, meshData->n_faces * 3));
	Real z_del = (maxOfArr(z_point, meshData->n_faces * 3) - minOfArr(z_point, meshData->n_faces * 3));

	Real del = maxOfArr({ x_del, y_del, z_del });

	for_i(meshData->n_faces * 9,
		*(normalizedMeshData + i) = *(centered + i) / del;
	);

	delete[] centered, x_point, y_point, z_point;

	cout << "Normalization Finished.." << endl;
}


void ophTri::objScaleShift() {
	scaledMeshData = new Real[meshData->n_faces * 9];
	
	objNormCenter();

	Real* x_point = new Real[meshData->n_faces * 3];
	Real* y_point = new Real[meshData->n_faces * 3];
	Real* z_point = new Real[meshData->n_faces * 3];

	for_i(meshData->n_faces * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(meshData->n_faces * 3,
		*(scaledMeshData + 3 * i) = *(x_point + i)*objSize + objShift[_X];
		*(scaledMeshData + 3 * i + 1) = *(y_point + i)*objSize + objShift[_Y];
		*(scaledMeshData + 3 * i + 2) = *(z_point + i)*objSize + objShift[_Z];
		);

	delete[] x_point, y_point, z_point, normalizedMeshData;

	cout << "Object Scaling and Shifting Finishied.." << endl;
}

void ophTri::objScaleShift(Real objSize_, vector<Real> objShift_) {
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[meshData->n_faces * 9];

	objNormCenter();

	Real* x_point = new Real[meshData->n_faces * 3];
	Real* y_point = new Real[meshData->n_faces * 3];
	Real* z_point = new Real[meshData->n_faces * 3];

	for_i(meshData->n_faces * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(meshData->n_faces * 3,
		*(scaledMeshData + 3 * i) = *(x_point + i)*objSize + objShift[_X];
		*(scaledMeshData + 3 * i + 1) = *(y_point + i)*objSize + objShift[_Y];
		*(scaledMeshData + 3 * i + 2) = *(z_point + i)*objSize + objShift[_Z];
	);

	delete[] x_point, y_point, z_point;

	cout << "Object Scaling and Shifting Finishied.." << endl;
}

void ophTri::objScaleShift(Real objSize_, Real objShift_[]) {
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[meshData->n_faces * 9];

	objNormCenter();

	Real* x_point = new Real[meshData->n_faces * 3];
	Real* y_point = new Real[meshData->n_faces * 3];
	Real* z_point = new Real[meshData->n_faces * 3];

	for_i(meshData->n_faces * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(meshData->n_faces * 3,
		*(scaledMeshData + 3 * i) = *(x_point + i)*objSize + objShift[_X];
	*(scaledMeshData + 3 * i + 1) = *(y_point + i)*objSize + objShift[_Y];
	*(scaledMeshData + 3 * i + 2) = *(z_point + i)*objSize + objShift[_Z];
	);

	delete[] x_point, y_point, z_point;

	cout << "Object Scaling and Shifting Finishied.." << endl;
}

vec3 vecCross(const vec3& a, const vec3& b)
{
	vec3 c;

	c(0) = a(0 + 1) * b(0 + 2) - a(0 + 2) * b(0 + 1);

	c(1) = a(1 + 1) * b(1 + 2) - a(1 + 2) * b(1 + 1);

	c(2) = a(2 + 1) * b(2 + 2) - a(2 + 2) * b(2 + 1);


	return c;
}


void ophTri::generateMeshHologram(uint SHADING_FLAG) {
	cout << "Hologram Generation ..." << endl;
	auto start = CUR_TIME;

	initialize();
	initializeAS();
	generateAS(SHADING_FLAG);

	cout << "< AS >" << endl;
	fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(angularSpectrum, holo_gen, context_.pixel_number[_X], context_.pixel_number[_Y], OPH_BACKWARD, false);
	
	auto end = CUR_TIME;
	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...hologram generated..\n", during);
}

void ophTri::generateMeshHologram() {
	cout << "Hologram Generation ..." << endl;
	auto start = CUR_TIME;

	initialize();
	initializeAS();
	generateAS(SHADING_TYPE);

	//holo_gen = angularSpectrum;

	fft2(context_.pixel_number, angularSpectrum, OPH_BACKWARD, OPH_ESTIMATE);
	fftExecute(holo_gen);

	auto end = CUR_TIME;
	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...hologram generated..\n", during);
}


void ophTri::generateAS(uint SHADING_FLAG) {

	Real* mesh = new Real[9];
	calGlobalFrequency();

	mesh_local = new Real[9];
	flx = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	fly = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	flz = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	freqTermX = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];
	freqTermY = new Real[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	refAS = new Complex<Real>[context_.pixel_number[_X] * context_.pixel_number[_Y]];

	findNormals(SHADING_FLAG);

	for (uint n = 0; n < meshData->n_faces; n++) {
		for_i(9,
			mesh[i] = scaledMeshData[9 * n + i];
			);

		if (checkValidity(mesh, *(no + n)) != 1)
			continue;
		
		if (findGeometricalRelations(mesh, *(no + n)) != 1)
			continue;

		if (calFrequencyTerm() != 1)
			continue;

		switch (SHADING_FLAG)
		{
		case SHADING_FLAT:
			refAS_Flat(*(no + n));
			break;
		case SHADING_CONTINUOUS:
			refAS_Continuous(n);
			break;
		default:
			cout << "error: WRONG SHADING_FLAG" << endl;
			cin.get();
		}
		refToGlobal();
		cout << n+1 << " / " << meshData->n_faces << endl;
	}

	cout << "Angular Spectrum Generated..." << endl;

	delete[] mesh, scaledMeshData, fx, fy, fz, mesh_local, flx, fly, flz, freqTermX, freqTermY, refAS;
}


uint ophTri::findNormals(uint SHADING_FLAG) {

	no = new vec3[meshData->n_faces];
	na = new vec3[meshData->n_faces];
	nv = new vec3[meshData->n_faces * 3];

	for (uint num = 0; num < meshData->n_faces; num++)
	{
		*(no + num) = vecCross({ scaledMeshData[num * 9 + _X1] - scaledMeshData[num * 9 + _X2],
			scaledMeshData[num * 9 + _Y1] - scaledMeshData[num * 9 + _Y2],
			scaledMeshData[num * 9 + _Z1] - scaledMeshData[num * 9 + _Z2] },
			{ scaledMeshData[num * 9 + _X3] - scaledMeshData[num * 9 + _X2],
			scaledMeshData[num * 9 + _Y3] - scaledMeshData[num * 9 + _Y2],
			scaledMeshData[num * 9 + _Z3] - scaledMeshData[num * 9 + _Z2] });
		// 'vec.h'의 cross함수가 전역으로 되어있어서 오류뜸.
		// 'vec.h'에 extern을 하라해서 했는데 그래도 안 됨.
		// 그래서그냥함수우선 가져옴.
	}
	Real normNo = 0;
	for (uint num = 0; num < meshData->n_faces; num++) {
		normNo = normNo + norm(no[num])*norm(no[num]);
	}
	normNo = sqrt(normNo);

	for (uint num = 0; num < meshData->n_faces; num++) {
		*(na + num) = no[num] / normNo;
	}

	if (SHADING_FLAG == SHADING_CONTINUOUS) {
		vec3* vertices = new vec3[meshData->n_faces * 3];
		vec3 zeros(0, 0, 0);

		for (uint idx = 0; idx < meshData->n_faces * 3; idx++) {
			*(vertices + idx) = { scaledMeshData[idx * 3 + 0], scaledMeshData[idx * 3 + 1], scaledMeshData[idx * 3 + 2] };
		}
		for (uint idx1 = 0; idx1 < meshData->n_faces * 3; idx1++) {
			if (*(vertices + idx1) == zeros)
				continue;
			vec3 sum = *(na + idx1 / 3);
			uint count = 1;
			uint* idxes = new uint[meshData->n_faces * 3];
			*(idxes) = idx1;
			for (uint idx2 = idx1 + 1; idx2 < meshData->n_faces * 3; idx2++) {
				if (*(vertices + idx2) == zeros)
					continue;
				if ((vertices[idx1][0] == vertices[idx2][0])
					& (vertices[idx1][1] == vertices[idx2][1])
					& (vertices[idx1][2] == vertices[idx2][2])) {
					sum += *(na + idx2 / 3);
					*(vertices + idx2) = zeros;
					*(idxes + count) = idx2;
					count++;
				}
			}
			*(vertices + idx1) = zeros;

			sum = sum / count;
			sum = sum / norm(sum);
			for (uint i = 0; i < count; i++)
				*(nv + *(idxes + i)) = sum;
		}
	}
	return 1;
}

uint ophTri::checkValidity(Real* mesh, vec3 no) {
	
	if (no[_Z] < 0 || (no[_X] == 0 && no[_Y] == 0 && no[_Z] == 0)) {
		return -1;
	}
	return 1;
}

uint ophTri::findGeometricalRelations(Real* mesh, vec3 no) {
	vec3 n = no / norm(no);

	Real th, ph;
	if (n[_X] == 0 && n[_Z] == 0)
		th = 0;
	else
		th = atan(n[_X] / n[_Z]);
	Real temp = n[_Y] / sqrt(n[_X] * n[_X] + n[_Z] * n[_Z]);
	ph = atan(temp);
	geom.glRot[0] = cos(th);			geom.glRot[1] = 0;			geom.glRot[2] = -sin(th);
	geom.glRot[3] = -sin(ph)*sin(th);	geom.glRot[4] = cos(ph);	geom.glRot[5] = -sin(ph)*cos(th);
	geom.glRot[6] = cos(ph)*sin(th);	geom.glRot[7] = sin(ph);	geom.glRot[8] = cos(ph)*cos(th);

	for_i(3,
		mesh_local[3 * i] = geom.glRot[0] * mesh[3 * i] + geom.glRot[1] * mesh[3 * i + 1] + geom.glRot[2] * mesh[3 * i + 2];
	mesh_local[3 * i + 1] = geom.glRot[3] * mesh[3 * i] + geom.glRot[4] * mesh[3 * i + 1] + geom.glRot[5] * mesh[3 * i + 2];
	mesh_local[3 * i + 2] = geom.glRot[6] * mesh[3 * i] + geom.glRot[7] * mesh[3 * i + 1] + geom.glRot[8] * mesh[3 * i + 2];
		)

	geom.glShift[_X] = -mesh_local[_X1];
	geom.glShift[_Y] = -mesh_local[_Y1];
	geom.glShift[_Z] = -mesh_local[_Z1];

	for_i(3,
		mesh_local[3 * i] += geom.glShift[_X];
		mesh_local[3 * i + 1] += geom.glShift[_Y];
		mesh_local[3 * i + 2] += geom.glShift[_Z];
	);

	if (mesh_local[_X2] * mesh_local[_Y3] == mesh_local[_X3] * mesh_local[_Y2])
		return -1;

	geom.loRot[0] = (refTri[_X3] * mesh_local[_Y2] - refTri[_X2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[1] = (refTri[_X3] * mesh_local[_X2] - refTri[_X2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[2] = (refTri[_Y3] * mesh_local[_Y2] - refTri[_Y2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[3] = (refTri[_Y3] * mesh_local[_X2] - refTri[_Y2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);

	/*
	cout << "global rotation" << endl;
	for_i(9,
		cout << geom.glRot[i] << ", ";
	);
	cout << endl << endl;

	cout << "global shift" << endl;
	for_i(3,
		cout << geom.glShift[i] << ", ";
	);
	cout << endl << endl;

	cout << "mesh local" << endl;
	for_i(9,
		cout << mesh_local[i] << ", ";
	);
	cout << endl << endl;

	cout << "local rotation" << endl;
	for_i(4,
		cout << geom.loRot[i] << ", ";
	);
	cout << endl << "." << endl << "." << endl << "." << endl << endl;

	cin.get();
	*/
	return 1;
}

void ophTri::calGlobalFrequency() {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];
	
	Real dfx = 1 / context_.pixel_pitch[_X] / Nx;
	Real dfy = 1 / context_.pixel_pitch[_Y] / Ny;
	fx = new Real[Nx*Ny];
	fy = new Real[Nx*Ny];
	fz = new Real[Nx*Ny];
	uint i = 0;
	for (int idxFy = Ny / 2; idxFy > -Ny / 2; idxFy--) {
		for (int idxFx = -Nx / 2; idxFx < Nx / 2; idxFx++) {
			fx[i] = idxFx*dfx;
			fy[i] = idxFy*dfy;
			fz[i] = sqrt((1 / context_.lambda)*(1 / context_.lambda) - fx[i] * fx[i] - fy[i] * fy[i]);

			i++;
		}
	}
}

uint ophTri::calFrequencyTerm() {
	
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	Real* flxShifted = new Real[Nx*Ny];
	Real* flyShifted = new Real[Nx*Ny];

	for_i(Nx*Ny,
		flx[i] = geom.glRot[0] * fx[i] + geom.glRot[1] * fy[i] + geom.glRot[2] * fz[i];
		fly[i] = geom.glRot[3] * fx[i] + geom.glRot[4] * fy[i] + geom.glRot[5] * fz[i];
		flz[i] = sqrt((1 / context_.lambda)*(1 / context_.lambda) - flx[i] * flx[i] - fly[i] * fly[i]);
		
		flxShifted[i] = flx[i] - (1 / context_.lambda)*(geom.glRot[0] * carrierWave[_X] + geom.glRot[1] * carrierWave[_Y] + geom.glRot[2] * carrierWave[_Z]);
		flyShifted[i] = fly[i] - (1 / context_.lambda)*(geom.glRot[3] * carrierWave[_X] + geom.glRot[4] * carrierWave[_Y] + geom.glRot[5] * carrierWave[_Z]);
		//cout << "fxfy: " << fx[i] << ", " << fy[i] << ", " << fz[i] << endl;
		//cout << "flxfly: " << flx[i] << ", " << fly[i] << ", " << flz[i] << endl;
		//cout << "fluxfluy: " << flxShifted[i] << ", " << flyShifted[i] << endl;
		);

	Real* invLoRot = new Real[4];
	invLoRot[0] = (1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[3];
	invLoRot[1] = -(1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[2];
	invLoRot[2] = -(1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[1];
	invLoRot[3] = (1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[0];

	//cout << "ta: " << invLoRot[0] << ", " << invLoRot[1] << ", " << invLoRot[2] << ", " << invLoRot[3] << endl;
	for_i(Nx*Ny,
		freqTermX[i] = invLoRot[0] * flxShifted[i] + invLoRot[1] * flyShifted[i];
		freqTermY[i] = invLoRot[2] * flxShifted[i] + invLoRot[3] * flyShifted[i];
		//cout << "freqTerm: " << freqTermX[i] << ", " << freqTermY[i] << endl;
		//cin.get();
		);
	
	delete[] flxShifted, flyShifted, invLoRot;
	return 1;
}

uint ophTri::refAS_Flat(vec3 no) {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	vec3 n = no / norm(no);

	Real shadingFactor;
	
	Complex<Real> temp1(0,0);
	Complex<Real> temp2(0,0);

	if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z] == 0) {
		shadingFactor = 1;
	}
	else {
		vec3 normIllu = illumination / norm(illumination);
		shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
		if (shadingFactor < 0)
			shadingFactor = 0;		
	}
	for (int i = 0; i < Nx*Ny; i++) {
		if (freqTermX[i] == -freqTermY[i] && freqTermY[i] != 0) {
			temp1[_IM] = 2 * M_PI*freqTermY[i];
			temp2[_IM] = 1;
			refAS[i] = shadingFactor*(((Complex<Real>)1 - exp(temp1)) / (4 * M_PI*M_PI*freqTermY[i] * freqTermY[i]) + temp2 / (2 * M_PI*freqTermY[i]));
		}
		else if (freqTermX[i] == freqTermY[i] && freqTermX[i] == 0) {
			refAS[i] = shadingFactor * 1 / 2;
		}
		else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
			temp1[_IM] = -2 * M_PI*freqTermX[i];
			temp2[_IM] = 1;
			refAS[i] = shadingFactor*((exp(temp1) - (Complex<Real>)1) / (2 * M_PI*freqTermX[i] * 2 * M_PI*freqTermX[i]) + (temp2 * exp(temp1)) / (2 * M_PI*freqTermX[i]));
		}
		else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
			temp1[_IM] = 2 * M_PI*freqTermY[i];
			temp2[_IM] = 1;
			refAS[i] = shadingFactor*(((Complex<Real>)1 - exp(temp1)) / (4 * M_PI*M_PI*freqTermY[i] * freqTermY[i]) - temp2 / (2 * M_PI*freqTermY[i]));
		}
		else {
			temp1[_IM] = -2 * M_PI*freqTermX[i];
			temp2[_IM] = -2 * M_PI*(freqTermX[i] + freqTermY[i]);
			refAS[i] = shadingFactor*((exp(temp1) - (Complex<Real>)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i]) + ((Complex<Real>)1 - exp(temp2)) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])));
		}
	}
	return 1;
}

uint ophTri::refAS_Continuous(uint n) {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	vec3 av(0, 0, 0);
	av[0] = nv[3 * n + 0][0] * illumination[0] + nv[3 * n + 0][1] * illumination[1] + nv[3 * n + 0][2] * illumination[2] + 0.1;
	av[2] = nv[3 * n + 1][0] * illumination[0] + nv[3 * n + 1][1] * illumination[1] + nv[3 * n + 1][2] * illumination[2] + 0.1;
	av[1] = nv[3 * n + 2][0] * illumination[0] + nv[3 * n + 2][1] * illumination[1] + nv[3 * n + 2][2] * illumination[2] + 0.1;

	//cout << nv[3 * n + 0][0] << ", " << nv[3 * n + 0][1] << ", " << nv[3 * n + 0][2] << endl;
	//cout << nv[3 * n + 1][0] << ", " << nv[3 * n + 1][1] << ", " << nv[3 * n + 1][2] << endl;
	//cout << nv[3 * n + 2][0] << ", " << nv[3 * n + 2][1] << ", " << nv[3 * n + 2][2] << endl;
	//cout << "illu: " << illumination[0] << ", " << illumination[1] << ", " << illumination[2] << endl;
	//cout << "av: " << av[0] << ", " << av[1] << ", " << av[2] << endl;
	//cout << endl;


	Complex<Real> D1(0, 0);
	Complex<Real> D2(0, 0);
	Complex<Real> D3(0, 0);

	Complex<Real> temp1(0, 0);
	Complex<Real> temp2(0, 0);
	Complex<Real> temp3(0, 0);
	
	for (int i = 0; i < Nx*Ny; i++) {
		if (freqTermX[i] == 0 && freqTermY[i] == 0) {
			D1((Real)1 / (Real)3, 0);
			D2((Real)1 / (Real)5, 0);
			D3((Real)1 / (Real)2, 0);
		}
		else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
			cout << i << endl;
			cout << "2" << endl;
			temp1[_IM] = -2 * M_PI*freqTermY[i];
			temp2[_IM] = 1;
			
			D1 = (temp1 - (Real)1)*temp1.exp() / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]) 
				- temp1 / (4 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D2 = -(M_PI*freqTermY[i] + temp2) / (4 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(temp1) 
				+ temp1 / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i]);
			D3 = exp(temp1) / (2 * M_PI*freqTermY[i]) + ((Real)1 - temp2) / (2 * M_PI*freqTermY[i]);
			cout << "D1: " << D1 << endl;
			cout << "D2: " << D2 << endl;
			cout << "D3: " << D3 << endl;
			cin.get();
		}
		else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
			cout << i << endl;
			cout << "3" << endl;
			temp1[_IM] = 4 * M_PI*M_PI*freqTermX[i] * freqTermX[i];
			temp2[_IM] = 1;
			temp3[_IM] = 2 * M_PI*freqTermX[i];

			D1 = (temp1 + 4 * M_PI*freqTermX[i] - (Real)2 * temp2) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * freqTermY[i])*exp(-temp3) 
				+ temp2 / (4 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = (Real)1 / (Real)2 * D1;
			D3 = ((temp3 + (Real)1)*exp(-temp3) - (Real)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
			cout << "D1: " << D1 << endl;
			cout << "D2: " << D2 << endl;
			cout << "D3: " << D3 << endl;
			cin.get();
		}
		else if (freqTermX[i] == -freqTermY[i]) {
			//cout << i << endl;
			//cout << "4" << endl;
			temp1[_IM] = 1;
			temp2[_IM] = 2 * M_PI*freqTermX[i];
			temp3[_IM] = 2 * M_PI*M_PI*freqTermX[i] * freqTermX[i];

			D1 = (-2 * M_PI*freqTermX[i] + temp1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-temp2) 
				- (temp3 + temp1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D2 = (-temp1) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i])*exp(-temp2) 
				+ (-temp3 + temp1 + 2 * M_PI*freqTermX[i]) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermX[i]);
			D3 = (-temp1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i])*exp(-temp2) 
				+ (-temp2 + (Real)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermX[i]);
		}
		else {
			//cout << i << endl;
			//cout << "5" << endl;
			//cout << freqTermX[i] << ", " << freqTermY[i] << endl;
			temp1[_IM] = -2 * M_PI*(freqTermX[i] + freqTermY[i]);
			temp2[_IM] = 1;
			temp3[_IM] = -2 * M_PI*freqTermX[i];
			//cout << temp1 << ", " << temp2 << ", " << temp3 << endl;
			//cout << exp(temp1) << endl;
			//cout << (temp2 - 2 * M_PI*(freqTermX[i] + freqTermY[i])) << endl;
			//cout << temp2 << endl;
			//cout << (8 * M_PI*M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i])) << endl;
			//cout << exp(temp1)*(temp2 - 2 * M_PI*(freqTermX[i] + freqTermY[i])) << endl;

			//cout << exp(temp1)*(temp2 - 2 * M_PI*(freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i])) << endl;
			//cout << exp(temp3)*(2 * M_PI*freqTermX[i] - temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermY[i]) << endl;
			//cout << ((2 * freqTermX[i] + freqTermY[i])*temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i])) << endl;
			//cout << exp(temp1)*(temp2*(freqTermX[i] + 2 * freqTermY[i]) - 2 * M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i])) << endl;
			//cout << exp(temp3)*(-temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermY[i] * freqTermY[i]) << endl;
			//cout << temp2 / (8 * M_PI*M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i])* (freqTermX[i] + freqTermY[i])) << endl;
			//cout << -exp(temp1) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])) << endl;
			//cout << exp(temp3) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i]) << endl;
			//cout << -(Real)1 / (4 * M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i])) << endl;

			D1 = exp(temp1)*(temp2 - 2 * M_PI*(freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(temp3)*(2 * M_PI*freqTermX[i] - temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * freqTermY[i])
				+ ((2 * freqTermX[i] + freqTermY[i])*temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermX[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]));
			D2 = exp(temp1)*(temp2*(freqTermX[i] + 2 * freqTermY[i]) - 2 * M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i])) / (8 * M_PI*M_PI*M_PI*freqTermY[i] * freqTermY[i] * (freqTermX[i] + freqTermY[i])*(freqTermX[i] + freqTermY[i]))
				+ exp(temp3)*(-temp2) / (8 * M_PI*M_PI*M_PI*freqTermX[i] * freqTermY[i] * freqTermY[i])
				+ temp2 / (8 * M_PI*M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i])* (freqTermX[i] + freqTermY[i]));
			D3 = -exp(temp1) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i]))
				+ exp(temp3) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i])
				- (Real)1 / (4 * M_PI*M_PI*freqTermX[i] * (freqTermX[i] + freqTermY[i]));
			//cout << "D1: " << D1 << endl;
			//cout << "D2: " << D2 << endl;
			//cout << "D3: " << D3 << endl;
			//cin.get();
		}
		refAS[i] = (av[1] - av[0])*D1 + (av[2] - av[1])*D2 + av[0] * D3;
		//cout << av[0] << ", " << av[1] << ", " << av[2] << endl;
		//cout << D1 << ", " << D2 << ", " << D3 << endl;
		//cout << i << ": " << refAS[i] << endl;
		//cin.get();
	}
	
	return 1;
}

void ophTri::refToGlobal() {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	Complex<Real> term1(0,0);
	Complex<Real> term2(0,0);

	term1[_IM] = -2 * M_PI / context_.lambda*(
		carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[3] * geom.glShift[_Y] + geom.glRot[6] * geom.glShift[_Z])
		+ carrierWave[_Y] * (geom.glRot[1] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[7] * geom.glShift[_Z])
		+ carrierWave[_Z] * (geom.glRot[2] * geom.glShift[_X] + geom.glRot[5] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z]));
	Complex<Real> temp(0,0);
	for (int i = 0; i < Nx*Ny; i++) {
		term2[_IM] = 2 * M_PI*(flx[i] * geom.glShift[_X] + fly[i] * geom.glShift[_Y] + flz[i] * geom.glShift[_Z]);
		temp = refAS[i] / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2])* exp(term1)* flz[i] / fz[i] * exp(term2);
		angularSpectrum[i] += temp;	
	}
}
