#include "ophTriMesh.h"

#define for_i(iter, oper)	for(uint i=0;i<iter;i++){oper}

#define _X1 0
#define _Y1 1
#define _Z1 2
#define _X2 3
#define _Y2 4
#define _Z2 5
#define _X3 6
#define _Y3 7
#define _Z3 8

void ophTri::loadMeshData(const char* fileName) {

	ifstream file;
	file.open(fileName);

	if (!file) {
		cout << "open failed - mesh data file" << endl;
		cin.get();
		return;
	}

	triMeshData = new Real[9 * 10000];

	Real data;
	uint num_data;

	num_data = 0;
	do {
		file >> data;
		triMeshData[num_data] = data;
		num_data++;
	} while (file.get() != EOF);

	num_mesh = num_data / 9;
	triMeshData[num_mesh*9] = EOF;

	cout << "Mesh Data Load Finished.." << endl;
}

void ophTri::objNormCenter() {
	normalizedMeshData = new Real[num_mesh * 9];

	Real* normalized = new Real[num_mesh * 9];
	oph::normalize<Real>(triMeshData, normalized, num_mesh * 9);
	
	Real* x_point = new Real[num_mesh * 3];
	Real* y_point = new Real[num_mesh * 3];
	Real* z_point = new Real[num_mesh * 3];
	
	for_i(num_mesh * 3,
		*(x_point + i) = *(normalized + 3 * i);
		*(y_point + i) = *(normalized + 3 * i + 1);
		*(z_point + i) = *(normalized + 3 * i + 2);
		);

	Real x_ave = average(x_point, num_mesh * 3);
	Real y_ave = average(y_point, num_mesh * 3);
	Real z_ave = average(z_point, num_mesh * 3);

	for_i(num_mesh * 3,
		*(normalizedMeshData + 3 * i) = *(normalized + 3 * i) - x_ave;
		*(normalizedMeshData + 3 * i + 1) = *(normalized + 3 * i + 1) - y_ave;
		*(normalizedMeshData + 3 * i + 2) = *(normalized + 3 * i + 2) - z_ave;
		);
	delete[] normalized, x_point, y_point, z_point;

	cout << "Normalization Finished.." << endl;
}

void ophTri::objScaleShift() {
	scaledMeshData = new Real[num_mesh * 9];

	objNormCenter();

	Real* x_point = new Real[num_mesh * 3];
	Real* y_point = new Real[num_mesh * 3];
	Real* z_point = new Real[num_mesh * 3];

	for_i(num_mesh * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(num_mesh * 3,
		*(scaledMeshData + 3 * i) = *(x_point + i)*objSize + objShift[_X];
		*(scaledMeshData + 3 * i + 1) = *(y_point + i)*objSize + objShift[_Y];
		*(scaledMeshData + 3 * i + 2) = *(z_point + i)*objSize + objShift[_Z];
		);

	delete[] x_point, y_point, z_point;

	cout << "Object Scaling and Shifting Finishied.." << endl;
}

void ophTri::objScaleShift(Real objSize_, vector<Real> objShift_) {
	setObjSize(objSize_);
	setObjShift(objShift_);

	scaledMeshData = new Real[num_mesh * 9];

	objNormCenter();

	Real* x_point = new Real[num_mesh * 3];
	Real* y_point = new Real[num_mesh * 3];
	Real* z_point = new Real[num_mesh * 3];

	for_i(num_mesh * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(num_mesh * 3,
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

	scaledMeshData = new Real[num_mesh * 9];

	objNormCenter();

	Real* x_point = new Real[num_mesh * 3];
	Real* y_point = new Real[num_mesh * 3];
	Real* z_point = new Real[num_mesh * 3];

	for_i(num_mesh * 3,
		*(x_point + i) = *(normalizedMeshData + 3 * i);
	*(y_point + i) = *(normalizedMeshData + 3 * i + 1);
	*(z_point + i) = *(normalizedMeshData + 3 * i + 2);
	);

	for_i(num_mesh * 3,
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

void ophTri::generateAS(uint SHADING_FLAG) {
	Real* mesh = new Real[9];
	calGlobalFrequency();

	for (uint n = 0; n < num_mesh; n++) {
		for_i(9,
			mesh[i] = scaledMeshData[9 * n + i];
			cout << mesh[i] << ", ";
			);
		cout << endl;
		
		vec3 no = vecCross({ mesh[_X2] - mesh[_X3], mesh[_Y2] - mesh[_Y3], mesh[_Z2] - mesh[_Z3] }, { mesh[_X3] - mesh[_X1], mesh[_Y3] - mesh[_Y1], mesh[_Z3] - mesh[_Z1] });
		// 'vec.h'의 cross함수가 전역으로 되어있어서 오류뜸.
		// 'vec.h'에 extern을 하라해서 했는데 그래도 안 됨.
		// 그래서그냥함수우선 가져옴.

		cout << "no: " << no[0] << ", " << no[1] << ", " << no[2] << endl;


		if (checkValidity(mesh, no) != 1)
			continue;
		
		if (findGeometricalRelations(mesh, no) != 1)
			continue;

		if (calFrequencyTerm() != 1)
			continue;

		switch (SHADING_FLAG)
		{
		case SHADING_FLAT:
			refAS_Flat(n);
			break;
		case SHADING_CONTINUOUS:
			refAS_Continuous();
			break;
		}
		refToGlobal();
	}

	cout << "Angular Spectrum Generation..." << endl;

	delete[] mesh;
}

uint ophTri::checkValidity(Real* mesh, vec3 no) {
	
	if (no[_Z] > 0 || (no[_X] == 0 && no[_Y] == 0 && no[_Z] == 0)) {
		return -1;
	}
	return 1;
}

uint ophTri::findGeometricalRelations(Real* mesh, vec3 no) {
	vec3 n = no / norm(no);

	//cout << "no: " << no[0] << ", " << no[1] << ", " << no[2] << endl;
	//cout << "n: " << n[0] << ", " << n[1] << ", " << n[2] << endl;

	Real th, ph;
	if (n[_X] == 0 && n[_Z] == 0)
		th = 0;
	else
		th = atan(n[_X] / n[_Z]);
	Real temp = n[_Y] / sqrt(n[_X] * n[_X] + n[_Z] * n[_Z]);
	cout << "temp: " << temp << endl;
	ph = atan(temp);
	cout << th << ", " << ph << endl;
	geom.glRot[0] = cos(th);			geom.glRot[1] = 0;			geom.glRot[2] = -sin(th);
	geom.glRot[3] = -sin(ph)*sin(th);	geom.glRot[4] = cos(ph);	geom.glRot[5] = -sin(ph)*cos(th);
	geom.glRot[6] = cos(ph)*sin(th);	geom.glRot[7] = sin(ph);	geom.glRot[8] = cos(ph)*cos(th);

	mesh_local = new Real[9];

	for_i(3,
		mesh_local[3 * i] = geom.glRot[0] * mesh[3 * i] + geom.glRot[1] * mesh[3 * i + 1] + geom.glRot[2] * mesh[3 * i + 2];
		mesh_local[3 * i + 1] = geom.glRot[3] * mesh[3 * i] + geom.glRot[4] * mesh[3 * i + 1] + geom.glRot[5] * mesh[3 * i + 2];
		mesh_local[3 * i + 2] = geom.glRot[6] * mesh[3 * i] + geom.glRot[7] * mesh[3 * i + 1] + geom.glRot[8] * mesh[3 * i + 2];
	);

	cout << "mesh local" << endl;
	for_i(9,
		cout << mesh_local[i] << ", ";
	);
	cout << endl << endl;

	geom.glShift[_X] = mesh_local[_X1];
	geom.glShift[_Y] = mesh_local[_Y1];
	geom.glShift[_Z] = mesh_local[_Z1];

	for_i(3,
		mesh_local[3 * i] -= geom.glShift[_X];
		mesh_local[3 * i + 1] -= geom.glShift[_Y];
		mesh_local[3 * i + 2] -= geom.glShift[_Z];
	);

	if (mesh_local[_X2] * mesh_local[_Y3] == mesh_local[_X3] * mesh_local[_Y2])
		return -1;

	geom.loRot[0] = (refTri[_X3] * mesh_local[_Y2] - refTri[_X2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[1] = (refTri[_X3] * mesh_local[_X2] - refTri[_X2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[2] = (refTri[_Y3] * mesh_local[_Y2] - refTri[_Y2] * mesh_local[_Y3]) / (mesh_local[_X3] * mesh_local[_Y2] - mesh_local[_Y3] * mesh_local[_X2]);
	geom.loRot[3] = (refTri[_Y3] * mesh_local[_X2] - refTri[_Y2] * mesh_local[_X3]) / (-mesh_local[_X3] * mesh_local[_Y2] + mesh_local[_Y3] * mesh_local[_X2]);


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


	return 1;
}

void ophTri::calGlobalFrequency() {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];
	
	Real dfx = 1 / context_.pixel_pitch[_X] / Nx;
	Real dfy = 1 / context_.pixel_pitch[_Y] / Ny;
	fx = new Real[Nx*Ny];
	fy = new Real[Ny*Ny];
	fz = new Real[Nx*Ny];
	uint i = 0;
	for (int idxFy = -Ny / 2; idxFy < Ny / 2; idxFy++) {
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

	flx = new Real[Nx*Ny];
	fly = new Real[Nx*Ny];
	flz = new Real[Nx*Ny];

	for_i(Nx*Ny,
		flx[i] = geom.glRot[0] * fx[i] + geom.glRot[1] * fy[i] + geom.glRot[2] * fz[i]
			- (1 / context_.lambda)*(geom.glRot[0] * carrierWave[_X] + geom.glRot[1] * carrierWave[_Y] + geom.glRot[2] * carrierWave[_Z]);
		fly[i] = geom.glRot[3] * fx[i] + geom.glRot[4] * fy[i] + geom.glRot[5] * fz[i]
			- (1 / context_.lambda)*(geom.glRot[3] * carrierWave[_X] + geom.glRot[4] * carrierWave[_Y] + geom.glRot[5] * carrierWave[_Z]);
		flz[i] = sqrt((1 / context_.lambda)*(1 / context_.lambda) - flx[i] * flx[i] - fly[i] * fly[i]);
		cout << flx[i] << ", " << fly[i] << ", " << flz[i] << endl;
		);
	cout << endl;

	freqTermX = new Real[Nx*Ny];
	freqTermY = new Real[Nx*Ny];

	Real* invLoRot = new Real[4];
	invLoRot[0] = (1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[3];
	invLoRot[1] = -(1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[2];
	invLoRot[2] = -(1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[1];
	invLoRot[3] = (1 / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2]))*geom.loRot[0];

	for_i(Nx*Ny,
		freqTermX[i] = invLoRot[0] * flx[i] + invLoRot[1] * fly[i];
		freqTermY[i] = invLoRot[2] * flx[i] + invLoRot[3] * fly[i];

		cout << freqTermX[i] << ", " << freqTermY[i] << endl;
		);
	cin.get();

	return 1;
}

void ophTri::refAS_Flat(vec3 no) {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	vec3 n = no / norm(no);
	Real shadingFactor;

	refAS = new Complex<Real>[Nx*Ny];
	
	Complex<Real> *temp1 = new Complex<Real>;
	Complex<Real> *temp2 = new Complex<Real>;
	
	for (uint i = 0; i < Nx*Ny; i++) {
		if (illumination[_X] == 0 && illumination[_Y] == 0 && illumination[_Z]) {
			shadingFactor = 1;
		}
		else {
			vec3 normIllu = illumination / norm(illumination);
			shadingFactor = 2 * (n[_X] * normIllu[_X] + n[_Y] * normIllu[_Y] + n[_Z] * normIllu[_Z]) + 0.3;
			if (shadingFactor < 0)
				shadingFactor = 0;
		}

		if (freqTermX[i] == -freqTermY[i] && freqTermY[i] != 0) {
			temp1->operator()(0, 2 * M_PI*freqTermY[i]);
			refAS[i] = shadingFactor*((Complex<Real>)1 - temp1->exp()) / (2 * M_PI*freqTermY[i] * freqTermY[i]) - (Complex<Real>)1 / *temp1;
			cout << "1st" << endl;
		}
		else if (freqTermX[i] == freqTermY[i] && freqTermX[i] == 0) {
			refAS[i] = shadingFactor * 1 / 2;
			cout << "2nd" << endl;
		}
		else if (freqTermX[i] != 0 && freqTermY[i] == 0) {
			temp1->operator()(0, -2 * M_PI*freqTermX[i]);
			temp2->operator()(0, 1);
			refAS[i] = shadingFactor*(temp1->exp() - (Complex<Real>)1) / (2 * M_PI*freqTermX[i] * 2 * M_PI*freqTermX[i]) - (*temp2 * temp1->exp()) / *temp1;
			cout << "3rd" << endl;
		}
		else if (freqTermX[i] == 0 && freqTermY[i] != 0) {
			temp1->operator()(0, -2 * M_PI*freqTermY[i]);
			temp2->operator()(0, 1);
			refAS[i] = shadingFactor*((Complex<Real>)1 - temp1->exp()) / (2 * M_PI*freqTermY[i] * freqTermY[i]) + *temp2 / *temp1;
			cout << "4th" << endl;
		}
		else {
			temp1->operator()(0, -2 * M_PI*freqTermX[i]);
			temp2->operator()(0, -2 * M_PI*(freqTermX[i] + freqTermY[i]));
			refAS[i] = shadingFactor*(temp1->exp() - (Complex<Real>)1) / (4 * M_PI*M_PI*freqTermX[i] * freqTermY[i]) + ((Complex<Real>)1 - temp2->exp()) / (4 * M_PI*M_PI*freqTermY[i] * (freqTermX[i] + freqTermY[i]));
			cout << "5th" << endl;
		}
	
	}

	delete temp1, temp2;

	for_i(20, cout << refAS[i] << endl;);
	
}

void ophTri::refAS_Continuous() {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	refAS = new Complex<Real>[Nx*Ny];

	Complex<Real> *temp1 = new Complex<Real>;
	Complex<Real> *temp2 = new Complex<Real>;

}

uint ophTri::findNormalForContinuous() {
	return 1;
}

//void ophTri::refToGlobal() {
//	Complex<Real>* term1;
//	Complex<Real>* term2;
//
//	for_i(context_.pixel_number[_X] * context_.pixel_number[_Y],
//		term1->operator()(0, -2 * M_PI / context_.lambda*(
//			carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[1] * geom.glShift[_Y] + geom.glRot[2] * geom.glShift[_Z])
//			+ carrierWave[_Y] * (geom.glRot[3] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[5] * geom.glShift[_Z])
//			+ carrierWave[_Z] * (geom.glRot[6] * geom.glShift[_X] + geom.glRot[7] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z])));
//		term2->operator()(0, 2 * M_PI*(flx[i] * geom.glShift[_X] + fly[i] * geom.glShift[_Y] + flz[i] * geom.glShift[_Z]));
//
//		angularSpectrum[i] = refAS[i] / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2])*flz[i] / fz[i] * term1->exp()*term2->exp();
//		)
//}

void ophTri::refToGlobal() {
	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	angularSpectrum = new Complex<Real>[Nx*Ny];
	Complex<Real> term1;
	Complex<Real> term2;

	for (uint i = 0; i < Nx*Ny; i++) {
		term1(0, -2 * M_PI / context_.lambda*(
			carrierWave[_X] * (geom.glRot[0] * geom.glShift[_X] + geom.glRot[1] * geom.glShift[_Y] + geom.glRot[2] * geom.glShift[_Z])
			+ carrierWave[_Y] * (geom.glRot[3] * geom.glShift[_X] + geom.glRot[4] * geom.glShift[_Y] + geom.glRot[5] * geom.glShift[_Z])
			+ carrierWave[_Z] * (geom.glRot[6] * geom.glShift[_X] + geom.glRot[7] * geom.glShift[_Y] + geom.glRot[8] * geom.glShift[_Z])));
		term2(0, 2 * M_PI*(flx[i] * geom.glShift[_X] + fly[i] * geom.glShift[_Y] + flz[i] * geom.glShift[_Z]));

		angularSpectrum[i] = refAS[i] / (geom.loRot[0] * geom.loRot[3] - geom.loRot[1] * geom.loRot[2])*flz[i] / fz[i] * term1.exp()*term2.exp();
	}

}