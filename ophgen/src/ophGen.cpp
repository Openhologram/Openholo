#include "ophGen.h"
#include <windows.h>
#include "sys.h"
#include "function.h"
#include <cuda_runtime.h>
#include <cufft.h>

#include "tinyxml2.h"
#include "PLYparser.h"

ophGen::ophGen(void)
	: Openholo()
	, holo_gen(nullptr)
	, holo_encoded(nullptr)
	, holo_normalized(nullptr)
{
	context_ = { 0 };
}

ophGen::~ophGen(void)
{
}

void ophGen::initialize(void)
{
	// Output Image Size
	int n_x = context_.pixel_number[_X];
	int n_y = context_.pixel_number[_Y];

	// Memory Location for Result Image
	if (holo_gen != nullptr) delete[] holo_gen;
	holo_gen = new oph::Complex<Real>[n_x * n_y];
	memset(holo_gen, 0, sizeof(Complex<Real>) * n_x * n_y);

	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[n_x * n_y];
	memset(holo_encoded, 0, sizeof(Real) * n_x * n_y);

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[n_x * n_y];
	memset(holo_normalized, 0, sizeof(uchar) * n_x * n_y);
}

int ophGen::loadPointCloud(const char* pc_file, OphPointCloudData *pc_data_)
{
	LOG("Reading....%s...\n", pc_file);

	auto start = CUR_TIME;

	PLYparser plyIO;
	if (!plyIO.loadPLY(pc_file, pc_data_->n_points, pc_data_->n_colors, &pc_data_->vertex, &pc_data_->color, &pc_data_->phase, pc_data_->isPhaseParse))
		return -1;

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return pc_data_->n_points;
}

bool ophGen::readConfig(const char* fname, OphPointCloudConfig& configdata)
{
	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(fname, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS )
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

#if REAL_IS_DOUBLE & true
	(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryDoubleText(&configdata.scale[_X]);
	(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryDoubleText(&configdata.scale[_Y]);
	(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryDoubleText(&configdata.scale[_Z]);
	(xml_node->FirstChildElement("OffsetInDepth"))->QueryDoubleText(&configdata.offset_depth);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("BandpassFilterWidthX"))->QueryDoubleText(&configdata.filter_width[_X]);
	(xml_node->FirstChildElement("BandpassFilterWidthY"))->QueryDoubleText(&configdata.filter_width[_Y]);
	(xml_node->FirstChildElement("FocalLengthofInputLens"))->QueryDoubleText(&configdata.focal_length_lens_in);
	(xml_node->FirstChildElement("FocalLengthofOutputLens"))->QueryDoubleText(&configdata.focal_length_lens_out);
	(xml_node->FirstChildElement("FocalLengthofEyepieceLens"))->QueryDoubleText(&configdata.focal_length_lens_eye_piece);
	(xml_node->FirstChildElement("WavelengthofLaser"))->QueryDoubleText(&context_.lambda);
	(xml_node->FirstChildElement("TiltAngleX"))->QueryDoubleText(&configdata.tilt_angle[_X]);
	(xml_node->FirstChildElement("TiltAngleY"))->QueryDoubleText(&configdata.tilt_angle[_Y]);
#else
	(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryFloatText(&configdata.scale[_X]);
	(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryFloatText(&configdata.scale[_Y]);
	(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryFloatText(&configdata.scale[_Z]);
	(xml_node->FirstChildElement("OffsetInDepth"))->QueryFloatText(&configdata.offset_depth);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("BandpassFilterWidthX"))->QueryFloatText(&configdata.filter_width[_X]);
	(xml_node->FirstChildElement("BandpassFilterWidthY"))->QueryFloatText(&configdata.filter_width[_Y]);
	(xml_node->FirstChildElement("FocalLengthofInputLens"))->QueryFloatText(&configdata.focal_length_lens_in);
	(xml_node->FirstChildElement("FocalLengthofOutputLens"))->QueryFloatText(&configdata.focal_length_lens_out);
	(xml_node->FirstChildElement("FocalLengthofEyepieceLens"))->QueryFloatText(&configdata.focal_length_lens_eye_piece);
	(xml_node->FirstChildElement("WavelengthofLaser"))->QueryFloatText(&context_.lambda);
	(xml_node->FirstChildElement("TiltAngleX"))->QueryFloatText(&configdata.tilt_angle[_X]);
	(xml_node->FirstChildElement("TiltAngleY"))->QueryFloatText(&configdata.tilt_angle[_Y]);
#endif
	(xml_node->FirstChildElement("SLMpixelNumX"))->QueryIntText(&context_.pixel_number[_X]);
	(xml_node->FirstChildElement("SLMpixelNumY"))->QueryIntText(&context_.pixel_number[_Y]);
	configdata.filter_shape_flag = (int8_t*)(xml_node->FirstChildElement("BandpassFilterShape"))->GetText();

	context_.k = (2 * M_PI) / context_.lambda;
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];
	/*XML parsing*/

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return true;
}

bool ophGen::readConfig(const char* fname, OphDepthMapConfig & config, OphDepthMapParams& params)
{
	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;
	/*XML parsing*/

	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(fname, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

	params.SOURCE_FOLDER = (xml_node->FirstChildElement("SourceFolder"))->GetText();
	params.IMAGE_PREFIX = (xml_node->FirstChildElement("ImagePrefix"))->GetText();
	params.DEPTH_PREFIX = (xml_node->FirstChildElement("DepthPrefix"))->GetText();
	params.RESULT_FOLDER = (xml_node->FirstChildElement("ResultFolder"))->GetText();
	params.RESULT_PREFIX = (xml_node->FirstChildElement("ResultPrefix"))->GetText();
	(xml_node->FirstChildElement("FlagStaticImage"))->QueryBoolText(&params.FLAG_STATIC_IMAGE);
	(xml_node->FirstChildElement("StartFrameNumber"))->QueryUnsignedText(&params.START_OF_FRAME_NUMBERING);
	(xml_node->FirstChildElement("NumberFrame"))->QueryUnsignedText(&params.NUMBER_OF_FRAME);
	(xml_node->FirstChildElement("NumberOfDigitOfFrameNumbering"))->QueryUnsignedText(&params.NUMBER_OF_DIGIT_OF_FRAME_NUMBERING);

	(xml_node->FirstChildElement("TransformMethod"))->QueryIntText(&params.Transform_Method_);
	(xml_node->FirstChildElement("PropagationMethod"))->QueryIntText(&params.Propagation_Method_);
	(xml_node->FirstChildElement("EncodingMethod"))->QueryIntText(&params.Encoding_Method_);

	(xml_node->FirstChildElement("SLMpixelNumX"))->QueryIntText(&context_.pixel_number[_X]);
	(xml_node->FirstChildElement("SLMpixelNumY"))->QueryIntText(&context_.pixel_number[_Y]);

	(xml_node->FirstChildElement("FlagChangeDepthQuantization"))->QueryBoolText(&params.FLAG_CHANGE_DEPTH_QUANTIZATION);
	(xml_node->FirstChildElement("DefaultDepthQuantization"))->QueryUnsignedText(&params.DEFAULT_DEPTH_QUANTIZATION);
	(xml_node->FirstChildElement("NumberOfDepthQuantization"))->QueryUnsignedText(&params.NUMBER_OF_DEPTH_QUANTIZATION);

	if (params.FLAG_CHANGE_DEPTH_QUANTIZATION == 0)
		config.num_of_depth = params.DEFAULT_DEPTH_QUANTIZATION;
	else
		config.num_of_depth = params.NUMBER_OF_DEPTH_QUANTIZATION;

	std::string render_depth = (xml_node->FirstChildElement("RenderDepth"))->GetText();

	std::size_t found = render_depth.find(':');
	if (found != std::string::npos)
	{
		std::string s = render_depth.substr(0, found);
		std::string e = render_depth.substr(found + 1);
		int start = std::stoi(s);
		int end = std::stoi(e);
		config.render_depth.clear();
		for (int k = start; k <= end; k++)
			config.render_depth.push_back(k);
	}
	else 
	{
		std::stringstream ss(render_depth);
		int render;

		while (ss >> render)
			config.render_depth.push_back(render);
	}

	if (config.render_depth.empty()) {
		LOG("not found Render Depth Parameter\n");
		return false;
	}

	(xml_node->FirstChildElement("RandomPahse"))->QueryBoolText(&params.RANDOM_PHASE);
	
#if REAL_IS_DOUBLE & true
	(xml_node->FirstChildElement("FieldLens"))->QueryDoubleText(&config.field_lens);
	(xml_node->FirstChildElement("WaveLength"))->QueryDoubleText(&context_.lambda);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("NearOfDepth"))->QueryDoubleText(&config.near_depthmap);
	(xml_node->FirstChildElement("FarOfDepth"))->QueryDoubleText(&config.far_depthmap);
#else
	(xml_node->FirstChildElement("FieldLens"))->QueryFloatText(&config.field_lens);
	(xml_node->FirstChildElement("WaveLength"))->QueryFloatText(&context_.lambda);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("NearOfDepth"))->QueryFloatText(&config.near_depthmap);
	(xml_node->FirstChildElement("FarOfDepth"))->QueryFloatText(&config.far_depthmap);
#endif

	context_.k = (2 * M_PI) / context_.lambda;
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];
	/*XML parsing*/

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);

	return true;
}

bool ophGen::readConfig(const char* fname, OphWRPConfig& configdata)
{
	/*	if (!ophGen::readConfig(cfg_file, pc_config_))
	return false;

	return true;*/
	LOG("Reading....%s...", fname);

	auto start = CUR_TIME;

	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;

	if (checkExtension(fname, ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fname);
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fname);
		return false;
	}

	xml_node = xml_doc.FirstChild();

#if REAL_IS_DOUBLE & true
	(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryDoubleText(&configdata.scale[_X]);
	(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryDoubleText(&configdata.scale[_Y]);
	(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryDoubleText(&configdata.scale[_Z]);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryDoubleText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryDoubleText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("Wavelength"))->QueryDoubleText(&context_.lambda);
	(xml_node->FirstChildElement("LocationOfWRP"))->QueryDoubleText(&configdata.wrp_location);
	(xml_node->FirstChildElement("PropagationDistance"))->QueryDoubleText(&configdata.propagation_distance);

#else
	(xml_node->FirstChildElement("ScalingXofPointCloud"))->QueryFloatText(&configdata.scale[_X]);
	(xml_node->FirstChildElement("ScalingYofPointCloud"))->QueryFloatText(&configdata.scale[_Y]);
	(xml_node->FirstChildElement("ScalingZofPointCloud"))->QueryFloatText(&configdata.scale[_Z]);
	(xml_node->FirstChildElement("SLMpixelPitchX"))->QueryFloatText(&context_.pixel_pitch[_X]);
	(xml_node->FirstChildElement("SLMpixelPitchY"))->QueryFloatText(&context_.pixel_pitch[_Y]);
	(xml_node->FirstChildElement("Wavelength"))->QueryFloatText(&context_.lambda);
	(xml_node->FirstChildElement("LocationOfWRP"))->QueryFloatText(&configdata.wrp_location);
	(xml_node->FirstChildElement("PropagationDistance"))->QueryFloatText(&configdata.propagation_distance);
#endif
	(xml_node->FirstChildElement("SLMpixelNumX"))->QueryIntText(&context_.pixel_number[_X]);
	(xml_node->FirstChildElement("SLMpixelNumY"))->QueryIntText(&context_.pixel_number[_Y]);
	(xml_node->FirstChildElement("NumberOfWRP"))->QueryIntText(&configdata.num_wrp);


	context_.k = (2 * M_PI) / context_.lambda;
	context_.ss[_X] = context_.pixel_number[_X] * context_.pixel_pitch[_X];
	context_.ss[_Y] = context_.pixel_number[_Y] * context_.pixel_pitch[_Y];
	/*XML parsing*/

	auto end = CUR_TIME;

	auto during = ((std::chrono::duration<Real>)(end - start)).count();

	LOG("%.5lfsec...done\n", during);
	return true;
}

void ophGen::normalize(void)
{
	oph::normalize((Real*)holo_encoded, holo_normalized, context_.pixel_number[_X], context_.pixel_number[_Y]);
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uchar* src, uint px, uint py)
{
	if (fname == nullptr) return -1;

	uchar* source = src;
	ivec2 p(px, py);

	if (src == nullptr)
		source = holo_normalized;
	if (px == 0 && py == 0)
		p = ivec2(context_.pixel_number[_X], context_.pixel_number[_Y]);

	if (checkExtension(fname, ".bmp")) 	// when the extension is bmp
		return Openholo::saveAsImg(fname, bitsperpixel, source, p[_X], p[_Y]);
	else {									// when extension is not .ohf, .bmp - force bmp
		char buf[256];
		memset(buf, 0x00, sizeof(char) * 256);
		sprintf_s(buf, "%s.bmp", fname);

		return Openholo::saveAsImg(buf, bitsperpixel, source, p[_X], p[_Y]);
	}
}

int ophGen::save(const char * fname, uint8_t bitsperpixel, uint px, uint py, uint fnum, uchar* args ...)
{
	std::string file = fname;
	std::string name;
	std::string ext;

	size_t ex = file.rfind(".");
	if (ex == -1) ex = file.length();
	 
	name = file.substr(0, ex);
	ext = file.substr(ex, file.length() - 1);

	va_list ap;
	__crt_va_start(ap, args);

	for (uint i = 0; i < fnum; i++) {
		name.append(std::to_string(i)).append(ext);
		if (i == 0) {
			save(name.c_str(), bitsperpixel, args, px, py);
			continue;
		}
		uchar* data = __crt_va_arg(ap, uchar*);
		save(name.c_str(), bitsperpixel, data, px, py);
	}

	__crt_va_end(ap);

	return 0;
}

void* ophGen::load(const char * fname)
{
	if (checkExtension(fname, ".bmp")) {
		return Openholo::loadAsImg(fname);
	}
	else if (checkExtension(fname, ".ohf")) {
		return nullptr;
	}
	else {			// when extension is not .ohf, .bmp
		return nullptr;
	}

	return nullptr;
}

#define for_i(itr, oper) for(uint i=0; i<itr; i++){ oper }

void ophGen::loadComplex(char* real_file, char* imag_file, int n_x, int n_y) {

	context_.pixel_number.v[_X] = n_x;
	context_.pixel_number.v[_Y] = n_y;

	ifstream freal, fimag;
	freal.open(real_file);
	fimag.open(imag_file);
	if (!freal) {
		cout << "open failed - real" << endl;
		cin.get();
		return;
	}
	if (!fimag) {
		cout << "open failed - imag" << endl;
		cin.get();
		return;
	}

	if (holo_gen != nullptr) delete[] holo_gen;
	holo_gen = new oph::Complex<Real>[n_x * n_y];
	memset(holo_gen, 0, sizeof(Complex<Real>) * n_x * n_y);

	Real realVal, imagVal;
	
	int i;
	i = 0;
	for (int i = 0; i < n_x * n_y; i++) {
		freal >> realVal;
		fimag >> imagVal;

		Complex<Real> compVal;
		compVal(realVal, imagVal);

		holo_gen[i] = compVal;
		if (realVal == EOF || imagVal == EOF)
			break;
	}
}

void ophGen::normalizeEncoded() {
	oph::normalize((Real*)holo_encoded, holo_normalized, encode_size.v[_X], encode_size.v[_Y]);
}

void ophGen::encoding(unsigned int ENCODE_FLAG) {

	const int size = context_.pixel_number.v[_X] * context_.pixel_number.v[_Y];

	if (ENCODE_FLAG == ENCODE_BURCKHARDT)	{
		encode_size[_X] = context_.pixel_number[_X] * 3;
		encode_size[_Y] = context_.pixel_number[_Y];
	}
	else if (ENCODE_FLAG == ENCODE_TWOPHASE)	{
		encode_size[_X] = context_.pixel_number[_X] * 2;
		encode_size[_Y] = context_.pixel_number[_Y];
	}
	else	{
		encode_size[_X] = context_.pixel_number[_X];
		encode_size[_Y] = context_.pixel_number[_Y];
	}

	/*	initialize	*/
	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[encode_size[_X] * encode_size[_Y]];
	memset(holo_encoded, 0, sizeof(Real) * encode_size[_X] * encode_size[_Y]);
	
	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[encode_size[_X] * encode_size[_Y]];
	memset(holo_normalized, 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);


	switch (ENCODE_FLAG)
	{
	case ENCODE_SIMPLENI:
		cout << "Simple Numerical Interference Encoding.." << endl;
		numericalInterference(holo_gen, holo_encoded, size);
		break;
	case ENCODE_REAL:
		cout << "Real Part Encoding.." << endl;
		realPart<Real>(holo_gen, holo_encoded, size);
		break;
	case ENCODE_BURCKHARDT:
		cout << "Burckhardt Encoding.." << endl;
		burckhardt(holo_gen, holo_encoded, size);
		break;
	case ENCODE_TWOPHASE:
		cout << "Two Phase Encoding.." << endl;
		twoPhaseEncoding(holo_gen, holo_encoded, size);
		break;
	case ENCODE_PHASE:
		cout << "Phase Encoding.." << endl;
		getPhase(holo_gen, holo_encoded, size);
		break;
	case ENCODE_AMPLITUDE:
		cout << "Amplitude Encoding.." << endl;
		getAmplitude(holo_gen, holo_encoded, size);
		break;
	case ENCODE_SSB:
	case ENCODE_OFFSSB:
		cout << "error: PUT PASSBAND" << endl;
		cin.get();
		return;
	default:
		cout << "error: WRONG ENCODE_FLAG" << endl;
		cin.get();
		return;
	}
}

void ophGen::encoding(unsigned int ENCODE_FLAG, unsigned int passband) {
	
	const int size = context_.pixel_number.v[_X] * context_.pixel_number.v[_Y];
	
	encode_size.v[_X] = context_.pixel_number.v[_X];
	encode_size.v[_Y] = context_.pixel_number.v[_Y];

	/*	initialize	*/
	int encode_size = size;
	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[encode_size];
	memset(holo_encoded, 0, sizeof(Real) * encode_size);

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[encode_size];
	memset(holo_normalized, 0, sizeof(uchar) * encode_size);

	switch (ENCODE_FLAG)
	{
	case ENCODE_SSB:
		cout << "Single Side Band Encoding.." << endl;
		singleSideBand(holo_gen, holo_encoded, context_.pixel_number, passband);
		break;
	case ENCODE_OFFSSB:
		cout << "Off-axis Single Side Band Encoding.." << endl;
		freqShift(holo_gen, holo_gen, context_.pixel_number, 0, 100);
		singleSideBand(holo_gen, holo_encoded, context_.pixel_number, passband);
		break;
	default:
		cout << "error: WRONG ENCODE_FLAG" << endl;
		cin.get();
		return;
	}
}

void ophGen::encoding() {

	const int size = context_.pixel_number.v[_X] * context_.pixel_number.v[_Y];

	if (ENCODE_METHOD == ENCODE_BURCKHARDT) {
		encode_size[_X] = context_.pixel_number[_X] * 3;
		encode_size[_Y] = context_.pixel_number[_Y];
	}
	else if (ENCODE_METHOD == ENCODE_TWOPHASE) {
		encode_size[_X] = context_.pixel_number[_X] * 2;
		encode_size[_Y] = context_.pixel_number[_Y];
	}
	else {
		encode_size[_X] = context_.pixel_number[_X];
		encode_size[_Y] = context_.pixel_number[_Y];
	}

	/*	initialize	*/
	if (holo_encoded != nullptr) delete[] holo_encoded;
	holo_encoded = new Real[encode_size[_X] * encode_size[_Y]];
	memset(holo_encoded, 0, sizeof(Real) * encode_size[_X] * encode_size[_Y]);

	if (holo_normalized != nullptr) delete[] holo_normalized;
	holo_normalized = new uchar[encode_size[_X] * encode_size[_Y]];
	memset(holo_normalized, 0, sizeof(uchar) * encode_size[_X] * encode_size[_Y]);


	switch (ENCODE_METHOD)
	{
	case ENCODE_SIMPLENI:
		cout << "Simple Numerical Interference Encoding.." << endl;
		numericalInterference(holo_gen, holo_encoded, size);
		break;
	case ENCODE_REAL:
		cout << "Real Part Encoding.." << endl;
		realPart<Real>(holo_gen, holo_encoded, size);
		break;
	case ENCODE_BURCKHARDT:
		cout << "Burckhardt Encoding.." << endl;
		burckhardt(holo_gen, holo_encoded, size);
		break;
	case ENCODE_TWOPHASE:
		cout << "Two Phase Encoding.." << endl;
		twoPhaseEncoding(holo_gen, holo_encoded, size);
		break;
	case ENCODE_PHASE:
		cout << "Phase Encoding.." << endl;
		getPhase(holo_gen, holo_encoded, size);
		break;
	case ENCODE_AMPLITUDE:
		cout << "Amplitude Encoding.." << endl;
		getAmplitude(holo_gen, holo_encoded, size);
		break;
	case ENCODE_SSB:
		cout << "Single Side Band Encoding.." << endl;
		singleSideBand(holo_gen, holo_encoded, context_.pixel_number, SSB_PASSBAND);
		break;
	case ENCODE_OFFSSB:
		cout << "Off-axis Single Side Band Encoding.." << endl;
		freqShift(holo_gen, holo_gen, context_.pixel_number, 0, 100);
		singleSideBand(holo_gen, holo_encoded, context_.pixel_number, SSB_PASSBAND);
		break;
	default:
		cout << "error: WRONG ENCODE_FLAG" << endl;
		cin.get();
		return;
	}
}

void ophGen::numericalInterference(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Real* temp1 = new Real[size];
	oph::absCplxArr<Real>(holo, temp1, size);

	Real* ref = new Real;
	*ref = oph::maxOfArr(temp1, size);

	oph::Complex<Real>* temp2 = new oph::Complex<Real>[size];
	temp2 = holo;
	for_i(size,
		temp2[i][_RE] += *ref;
	);

	Real* temp3 = new Real[size];
	oph::absCplxArr<Real>(temp2, temp3, size);

	for_i(size,
		encoded[i] = temp3[i] * temp3[i];
	);

	delete[] temp1, temp2, temp3;
	delete ref;
}

void ophGen::twoPhaseEncoding(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* normCplx = new Complex<Real>[size];
	oph::normalize<Real>(holo, normCplx, size);

	Real* amp = new Real[size];
	oph::getAmplitude(normCplx, encoded, size);

	Real* pha = new Real[size];
	oph::getPhase(normCplx, pha, size);

	for_i(size, *(pha + i) += M_PI;);

	Real* delPhase = new Real[size];
	for_i(size, *(delPhase + i) = acos(*(amp + i)););

	for_i(size,
		*(encoded + i * 2) = *(pha + i) + *(delPhase + i);
	*(encoded + i * 2 + 1) = *(pha + i) - *(delPhase + i);
	);

	delete[] normCplx, amp, pha, delPhase;
}

void ophGen::burckhardt(oph::Complex<Real>* holo, Real* encoded, const int size)
{
	Complex<Real>* norm = new Complex<Real>[size];
	oph::normalize(holo, norm, size);

	Real* phase = new Real[size];
	oph::getPhase(norm, phase, size);

	Real* ampl = new Real[size];
	oph::getAmplitude(norm, ampl, size);

	Real* A1 = new Real[size];
	memsetArr<Real>(A1, 0, 0, size - 1);
	Real* A2 = new Real[size];
	memsetArr<Real>(A2, 0, 0, size - 1);
	Real* A3 = new Real[size];
	memsetArr<Real>(A3, 0, 0, size - 1);

	for_i(size,
		if (*(phase + i) >= 0 && *(phase + i) < (2 * M_PI / 3))
		{
			*(A1 + i) = *(ampl + i)*(cos(*(phase + i)) + sin(*(phase + i)) / sqrt(3));
			*(A2 + i) = 2 * sin(*(phase + i)) / sqrt(3);
		}
		else if (*(phase + i) >= (2 * M_PI / 3) && *(phase + i) < (4 * M_PI / 3))
		{
			*(A2 + i) = *(ampl + i)*(cos(*(phase + i) - (2 * M_PI / 3)) + sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3));
			*(A3 + i) = 2 * sin(*(phase + i) - (2 * M_PI / 3)) / sqrt(3);
		}
		else if (*(phase + i) >= (4 * M_PI / 3) && *(phase + i) < (2 * M_PI))
		{
			*(A3 + i) = *(ampl + i)*(cos(*(phase + i) - (4 * M_PI / 3)) + sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3));
			*(A1 + i) = 2 * sin(*(phase + i) - (4 * M_PI / 3)) / sqrt(3);
		}
	);

	for_i(size,
		*(encoded + (3 * i)) = *(A1 + i);
	*(encoded + (3 * i + 1)) = *(A2 + i);
	*(encoded + (3 * i + 2)) = *(A3 + i);
	);

}

void ophGen::singleSideBand(oph::Complex<Real>* holo, Real* encoded, const ivec2 holosize, int SSB_PASSBAND)
{
	int size = holosize[_X] * holosize[_Y];

	oph::Complex<Real>* AS = new oph::Complex<Real>[size];
	fft2(holosize, holo, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(holo, AS, holosize[_X], holosize[_Y], OPH_FORWARD, false);
	//fftExecute(temp);

	switch (SSB_PASSBAND)
	{
	case SSB_LEFT:
		for (int i = 0; i < holosize[_Y]; i++)
		{
			for (int j = holosize[_X] / 2; j < holosize[_X]; j++)
			{
				AS[i*holosize[_X] + j] = 0;
			}
		}
		break;
	case SSB_RIGHT:
		for (int i = 0; i < holosize[_Y]; i++)
		{
			for (int j = 0; j < holosize[_X] / 2; j++)
			{
				AS[i*holosize[_X] + j] = 0;
			}
		}
		break;
	case SSB_TOP:
		for (int i = size / 2; i < size; i++)
		{
			AS[i] = 0;
		}
		break;
		
	case SSB_BOTTOM:
		for (int i = 0; i < size / 2; i++)
		{
			AS[i] = 0;
		}
		break;
	}

	oph::Complex<Real>* filtered = new oph::Complex<Real>[size];
	fft2(holosize, AS, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(AS, filtered, holosize[_X], holosize[_Y], OPH_BACKWARD, false);

	//fftExecute(filtered);


	Real* realFiltered = new Real[size];
	oph::realPart<Real>(filtered, realFiltered, size);

	oph::normalize(realFiltered, encoded, size);

	delete[] AS, filtered , realFiltered;
}


void ophGen::freqShift(oph::Complex<Real>* src, Complex<Real>* dst, const ivec2 holosize, int shift_x, int shift_y)
{
	int size = holosize[_X] * holosize[_Y];

	oph::Complex<Real>* AS = new oph::Complex<Real>[size];
	fft2(holosize, src, OPH_FORWARD, OPH_ESTIMATE);
	fftExecute(AS);

	oph::Complex<Real>* shifted = new oph::Complex<Real>[size];
	oph::circShift<Complex<Real>>(AS, shifted, shift_x, shift_y, holosize.v[_X], holosize.v[_Y]);

	fft2(holosize, shifted, OPH_BACKWARD, OPH_ESTIMATE);
	fftExecute(dst);
}


void ophGen::fresnelPropagation(OphContext context, Complex<Real>* in, Complex<Real>* out, Real distance) {

	int Nx = context.pixel_number[_X];
	int Ny = context.pixel_number[_Y];

	Complex<Real>* in2x = new Complex<Real>[Nx*Ny * 4];
	Complex<Real> zero(0, 0);
	//oph::memsetArr<Complex<Real>>(in2x, zero, 0, Nx*Ny*4-2);

	uint idxIn = 0;

	for (uint idxNy = Ny / 2; idxNy < Ny + (Ny / 2); idxNy++) {
		for (uint idxNx = Nx / 2; idxNx < Nx + (Nx / 2); idxNx++) {

			in2x[idxNy*Nx * 2 + idxNx] = in[idxIn];
			idxIn++;
		}
	}

	Complex<Real>* temp1 = new Complex<Real>[Nx*Ny * 4];

	ivec2 n(context.pixel_number[_X] * 2, context.pixel_number[_Y] * 2);

	fft2(n, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(in2x, temp1, n[_X], n[_Y], OPH_FORWARD, false);

	Real* fx = new Real[Nx*Ny * 4];
	Real* fy = new Real[Nx*Ny * 4];

	uint i = 0;
	for (uint idxFy = (1 - Ny); idxFy < (1 + Ny); idxFy++) {
		for (uint idxFx = (1 - Nx); idxFx < (1 + Nx); idxFx++) {
			fx[i] = idxFx;
			fy[i] = idxFy;
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[Nx*Ny * 4];
	Complex<Real> sqrtPart;

	Complex<Real>* temp2 = new Complex<Real>[Nx*Ny * 4];

	for (uint i = 0; i < Nx*Ny * 4; i++) {
		sqrtPart._Val[_RE] = sqrt(1 / (context.lambda*context.lambda) - fx[i] * fx[i] - fy[i] * fy[i]);
		prop[i] = 2 * M_PI*distance;
		prop[i] *= sqrtPart;
		temp2[i] = temp1[i] * exp(prop[i]);
	}

	Complex<Real>* temp3 = new Complex<Real>[Nx*Ny * 4];
	fft2((Nx * 2, Ny * 2), temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(temp2, temp3, Nx * 2, Ny * 2, OPH_BACKWARD, false);

	uint idxOut = 0;

	for (uint idxNy = Ny / 2; idxNy < Ny + (Ny / 2); idxNy++) {
		for (uint idxNx = Nx / 2; idxNx < Nx + (Nx / 2); idxNx++) {

			out[idxOut] = temp3[idxNy*Nx * 2 + idxNx];
			idxOut++;
		}
	}

	delete[] in2x, temp1, fx, fy, prop, temp2, temp3;
}

void ophGen::fresnelPropagation(Complex<Real>* in, Complex<Real>* out, Real distance) {

	int Nx = context_.pixel_number[_X];
	int Ny = context_.pixel_number[_Y];

	Complex<Real>* in2x = new Complex<Real>[Nx*Ny * 4];
	Complex<Real> zero(0, 0);
	oph::memsetArr<Complex<Real>>(in2x, zero, 0, Nx*Ny * 4 - 1);

	uint idxIn = 0;

	for (uint idxNy = Ny / 2; idxNy < Ny + (Ny / 2); idxNy++) {
		for (uint idxNx = Nx / 2; idxNx < Nx + (Nx / 2); idxNx++) {

			in2x[idxNy*Nx * 2 + idxNx] = in[idxIn];
			idxIn++;
		}
	}

	Complex<Real>* temp1 = new Complex<Real>[Nx*Ny * 4];

	fft2({ Nx * 2, Ny * 2 }, in2x, OPH_FORWARD, OPH_ESTIMATE);
	fftwShift(in2x, temp1, Nx*2, Ny*2, OPH_FORWARD, false);

	Real* fx = new Real[Nx*Ny * 4];
	Real* fy = new Real[Nx*Ny * 4];

	uint i = 0;
	for (uint idxFy = (1 - Ny); idxFy < (1 + Ny); idxFy++) {
		for (uint idxFx = (1 - Nx); idxFx < (1 + Nx); idxFx++) {
			fx[i] = idxFx;
			fy[i] = idxFy;
			i++;
		}
	}

	Complex<Real>* prop = new Complex<Real>[Nx*Ny * 4];
	Complex<Real> sqrtPart;

	Complex<Real>* temp2 = new Complex<Real>[Nx*Ny * 4];

	for (uint i = 0; i < Nx*Ny * 4; i++) {
		sqrtPart._Val[_RE] = sqrt(1 / (context_.lambda*context_.lambda) - fx[i] * fx[i] - fy[i] * fy[i]);
		prop[i] = 2 * M_PI * distance;
		prop[i] *= sqrtPart;
		temp2[i] = temp1[i] * exp(prop[i]);
	}

	Complex<Real>* temp3 = new Complex<Real>[Nx*Ny * 4];
	fft2({ Nx * 2, Ny * 2 }, temp2, OPH_BACKWARD, OPH_ESTIMATE);
	fftwShift(temp2, temp3, Nx * 2, Ny * 2, OPH_BACKWARD, false);

	uint idxOut = 0;

	for (uint idxNy = Ny / 2; idxNy < Ny + (Ny / 2); idxNy++) {
		for (uint idxNx = Nx / 2; idxNx < Nx + (Nx / 2); idxNx++) {

			out[idxOut] = temp3[idxNy*Nx * 2 + idxNx];
			idxOut++;
		}
	}

	delete[] in2x, temp1, fx, fy, prop, temp2, temp3;
}


void ophGen::encodeSideBand(bool bCPU, ivec2 sig_location)
{
	if (holo_gen == nullptr) {
		LOG("Not found diffracted data.");
		return;
	}

	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	int cropx1, cropx2, cropx, cropy1, cropy2, cropy;
	if (sig_location[1] == 0) { //Left or right half
		cropy1 = 1;
		cropy2 = pny;
	}
	else {
		cropy = (int)floor(((Real)pny) / 2);
		cropy1 = cropy - (int)floor(((Real)cropy) / 2);
		cropy2 = cropy1 + cropy - 1;
	}

	if (sig_location[0] == 0) { // Upper or lower half
		cropx1 = 1;
		cropx2 = pnx;
	}
	else {
		cropx = (int)floor(((Real)pnx) / 2);
		cropx1 = cropx - (int)floor(((Real)cropx) / 2);
		cropx2 = cropx1 + cropx - 1;
	}

	cropx1 -= 1;
	cropx2 -= 1;
	cropy1 -= 1;
	cropy2 -= 1;

	if (bCPU)
		encodeSideBand_CPU(cropx1, cropx2, cropy1, cropy2, sig_location);
	else
		encodeSideBand_GPU(cropx1, cropx2, cropy1, cropy2, sig_location);
}

void ophGen::encodeSideBand_CPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[_X];
	int pny = context_.pixel_number[_Y];

	oph::Complex<Real>* h_crop = new oph::Complex<Real>[pnx*pny];
	memset(h_crop, 0.0, sizeof(oph::Complex<Real>)*pnx*pny);

	int p = 0;
#pragma omp parallel for private(p)
	for (p = 0; p < pnx*pny; p++)
	{
		int x = p % pnx;
		int y = p / pnx;
		if (x >= cropx1 && x <= cropx2 && y >= cropy1 && y <= cropy2)
			h_crop[p] = holo_gen[p];
	}

	oph::Complex<Real> *in = nullptr;

	fft2(oph::ivec2(pnx, pny), in, OPH_BACKWARD);
	fftwShift(h_crop, h_crop, pnx, pny, OPH_BACKWARD, true);

	memset(holo_encoded, 0.0, sizeof(Real)*pnx*pny);
	int i = 0;
#pragma omp parallel for private(i)	
	for (i = 0; i < pnx*pny; i++) {
		oph::Complex<Real> shift_phase(1, 0);
		getShiftPhaseValue(shift_phase, i, sig_location);

		holo_encoded[i] = (h_crop[i] * shift_phase).real();
	}

	delete[] h_crop;
}

extern "C"
{
	/**
	* \defgroup gpu_model GPU Modules
	* @{
	*/
	/**
	* @brief Convert data from the spatial domain to the frequency domain using 2D FFT on GPU.
	* @details call CUDA Kernel - fftShift and CUFFT Library.
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param direction : If direction == -1, forward FFT, if type == 1, inverse FFT.
	* @param bNomarlized : If bNomarlized == true, normalize the result after FFT.
	* @see propagation_AngularSpectrum_GPU, encoding_GPU
	*/
	void cudaFFT(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_filed, cufftDoubleComplex* output_field, int direction, bool bNormailized = false);	
	
	/**
	* @brief Crop input data according to x, y coordinates on GPU.
	* @details call CUDA Kernel - cropFringe. 
	* @param stream : CUDA Stream
	* @param nx : the number of column of the input data
	* @param ny : the number of row of the input data
	* @param in_field : input complex data variable
	* @param output_field : output complex data variable
	* @param cropx1 : the start x-coordinate to crop.
	* @param cropx2 : the end x-coordinate to crop.
	* @param cropy1 : the start y-coordinate to crop.
	* @param cropy2 : the end y-coordinate to crop.
	* @see encoding_GPU
	*/
	void cudaCropFringe(CUstream_st* stream, int nx, int ny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int cropx1, int cropx2, int cropy1, int cropy2);

	/**
	* @brief Encode the CGH according to a signal location parameter on the GPU.
	* @details The variable, ((Real*)p_hologram) has the final result.
	* @param stream : CUDA Stream
	* @param pnx : the number of column of the input data
	* @param pny : the number of row of the input data
	* @param in_field : input data
	* @param out_field : output data
	* @param sig_locationx : signal location of x-axis, left or right half
	* @param sig_locationy : signal location of y-axis, upper or lower half
	* @param ssx : pnx * ppx
	* @param ssy : pny * ppy
	* @param ppx : pixel pitch of x-axis
	* @param ppy : pixel pitch of y-axis
	* @param PI : Pi
	* @see encoding_GPU
	*/
	void cudaGetFringe(CUstream_st* stream, int pnx, int pny, cufftDoubleComplex* in_field, cufftDoubleComplex* out_field, int sig_locationx, int sig_locationy,
		Real ssx, Real ssy, Real ppx, Real ppy, Real PI);
}

void ophGen::encodeSideBand_GPU(int cropx1, int cropx2, int cropy1, int cropy2, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	Real ppx = context_.pixel_pitch[0];
	Real ppy = context_.pixel_pitch[1];
	Real ssx = context_.ss[0];
	Real ssy = context_.ss[1];

	cufftDoubleComplex *k_temp_d_, *u_complex_gpu_;
	cudaStream_t stream_;
	cudaStreamCreate(&stream_);

	cudaMalloc((void**)&u_complex_gpu_, sizeof(cufftDoubleComplex) * pnx * pny);
	cudaMalloc((void**)&k_temp_d_, sizeof(cufftDoubleComplex) * pnx * pny);
	cudaMemcpy(u_complex_gpu_, holo_gen, sizeof(cufftDoubleComplex) * pnx * pny, cudaMemcpyHostToDevice);

	cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaCropFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, cropx1, cropx2, cropy1, cropy2);

	cudaMemsetAsync(u_complex_gpu_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaFFT(stream_, pnx, pny, k_temp_d_, u_complex_gpu_, 1, true);

	cudaMemsetAsync(k_temp_d_, 0, sizeof(cufftDoubleComplex)*pnx*pny, stream_);
	cudaGetFringe(stream_, pnx, pny, u_complex_gpu_, k_temp_d_, sig_location[0], sig_location[1], ssx, ssy, ppx, ppy, M_PI);

	cufftDoubleComplex* sample_fd = (cufftDoubleComplex*)malloc(sizeof(cufftDoubleComplex)*pnx*pny);
	memset(sample_fd, 0.0, sizeof(cufftDoubleComplex)*pnx*pny);

	cudaMemcpyAsync(sample_fd, k_temp_d_, sizeof(cufftDoubleComplex)*pnx*pny, cudaMemcpyDeviceToHost), stream_;
	memset(holo_encoded, 0.0, sizeof(Real)*pnx*pny);

	for (int i = 0; i < pnx * pny; i++)
		holo_encoded[i] = sample_fd[i].x;

	delete[] sample_fd;
	cudaStreamDestroy(stream_);
}

void ophGen::getShiftPhaseValue(oph::Complex<Real>& shift_phase_val, int idx, oph::ivec2 sig_location)
{
	int pnx = context_.pixel_number[0];
	int pny = context_.pixel_number[1];
	Real ppx = context_.pixel_pitch[0];
	Real ppy = context_.pixel_pitch[1];
	Real ssx = context_.ss[0];
	Real ssy = context_.ss[1];

	if (sig_location[1] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		Real yy = (ssy / 2.0) - (ppy)*r - ppy;

		oph::Complex<Real> val;
		if (sig_location[1] == 1)
			val[_IM] = 2 * M_PI * (yy / (4 * ppy));
		else
			val[_IM] = 2 * M_PI * (-yy / (4 * ppy));

		val.exp();
		shift_phase_val *= val;
	}

	if (sig_location[0] != 0)
	{
		int r = idx / pnx;
		int c = idx % pnx;
		Real xx = (-ssx / 2.0) - (ppx)*c - ppx;

		oph::Complex<Real> val;
		if (sig_location[0] == -1)
			val[_IM] = 2 * M_PI * (-xx / (4 * ppx));
		else
			val[_IM] = 2 * M_PI * (xx / (4 * ppx));

		val.exp();
		shift_phase_val *= val;
	}
}

void ophGen::getRandPhaseValue(oph::Complex<Real>& rand_phase_val, bool rand_phase)
{
	if (rand_phase)
	{
		rand_phase_val[_RE] = 0.0;
		Real min, max;
#if REAL_IS_DOUBLE & true
		min = 0.0;
		max = 1.0;
#else
		min = 0.f;
		max = 1.f;
#endif
		rand_phase_val[_IM] = 2 * M_PI * oph::rand(min, max);
		rand_phase_val.exp();

	}
	else {
		rand_phase_val[_RE] = 1.0;
		rand_phase_val[_IM] = 0.0;
	}
}

void ophGen::ophFree(void)
{
	if (holo_gen) delete[] holo_gen;
	if (holo_encoded) delete[] holo_encoded;
	if (holo_normalized) delete[] holo_normalized;
}