#include "ImgCodecOhc.h"

#define NOMINMAX // using std::numeric_limits<DataType>::max(), min() of <limits> instead of <minwindef.h>

#include "sys.h"
#include <limits> // limit value of each data types


//hot key for call by this pointer
#define FHeader this->Header->fileHeader
#define FldInfo this->Header->fieldInfo
#define WavLeng this->Header->wavlenTable


/************************ OHC CODEC *****************************/

oph::ImgCodecOhc::ImgCodecOhc() {
	this->initOHCheader();
}

oph::ImgCodecOhc::~ImgCodecOhc() {
	this->releaseOHCheader();
	this->releaseFldData();
	this->releaseCodeBuffer();
}

oph::ImgCodecOhc::ImgCodecOhc(const std::string &_fname) {
	this->initOHCheader();
	this->setFileName(_fname);
}

oph::ImgCodecOhc::ImgCodecOhc(const std::string &_fname, const OHCheader &_Header) {
	this->initOHCheader();
	this->setFileName(_fname);
	this->setOHCheader(_Header);
}

void oph::ImgCodecOhc::initOHCheader() {
	if (this->Header != nullptr)
		delete this->Header;

	this->Header = new OHCheader();
}

bool oph::ImgCodecOhc::setFileName(const std::string &_fname) {
	this->fname = _fname;

	return true;
}

bool oph::ImgCodecOhc::setOHCheader(const OHCheader &_Header) {
	if (this->Header != nullptr)
		delete this->Header;

	this->Header = new OHCheader(_Header);

	return true;
}

void oph::ImgCodecOhc::getOHCheader(OHCheader &_Header) {
	if (this->Header == nullptr)
		LOG("OHC CODEC : No Header Data.");
	else
		_Header = *(this->Header);
}

void oph::ImgCodecOhc::getFieldInfo(OHCFIELDINFOHEADER &_FieldInfo, std::vector<double_t> &_wavlenTable) {
	if (this->Header == nullptr)
		LOG("OHC CODEC : No Header Data.");
	else {
		_FieldInfo = this->Header->fieldInfo;
		_wavlenTable = this->Header->wavlenTable;
	}
}

void oph::ImgCodecOhc::releaseOHCheader() {
	delete this->Header;
}

void oph::ImgCodecOhc::releaseCodeBuffer() {
	//delete[] this->buf;
	delete[] this->buf_f32;
	delete[] this->buf_f64;
}

void oph::ImgCodecOhc::releaseFldData() {
	for (int i = 0; i < field_cmplx.size(); ++i) {
		this->field_cmplx[i].release();
	}
	this->field_cmplx.clear();
}


/************************ OHC Decoder *****************************/

oph::ImgDecoderOhc::ImgDecoderOhc()
	: ImgCodecOhc()
{
}

oph::ImgDecoderOhc::ImgDecoderOhc(const std::string &_fname)
	: ImgCodecOhc(_fname)
{
}

oph::ImgDecoderOhc::ImgDecoderOhc(const std::string &_fname, const OHCheader &_Header)
	: ImgCodecOhc(_fname, _Header)
{
}

oph::ImgDecoderOhc::~ImgDecoderOhc()
{
	this->releaseOHCheader();
	this->releaseFldData();
	this->releaseCodeBuffer();
}

void oph::ImgDecoderOhc::releaseFldData() {
	this->ImgCodecOhc::releaseFldData();

	for (int i = 0; i < field_ampli.size(); ++i) {
		this->field_ampli[i].release();
	}
	this->field_ampli.clear();

	for (int i = 0; i < field_phase.size(); ++i) {
		this->field_phase[i].release();
	}
	this->field_phase.clear();

	this->bLoadFile = false;
}

ivec2 oph::ImgDecoderOhc::getNumOfPixel() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return ivec2(-1, -1);
	}
	else
		return ivec2(FldInfo.pxNumX, FldInfo.pxNumY);
}

vec2 oph::ImgDecoderOhc::getPixelPitch() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return vec2(-1., -1.);
	}
	else
		return vec2(FldInfo.pxPitchX, FldInfo.pxPitchY);
}

LenUnit oph::ImgDecoderOhc::getPixelPitchUnit() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (LenUnit)-1;
	}
	else
		return FldInfo.pitchUnit;
}

uint oph::ImgDecoderOhc::getNumOfWavlen() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (uint)-1;
	}
	else
		return FldInfo.wavlenNum;
}

ColorType oph::ImgDecoderOhc::getColorType() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (ColorType)-1;
	}
	else
		return FldInfo.clrType;
}

ColorArran oph::ImgDecoderOhc::getColorArrange() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (ColorArran)-1;
	}
	else
		return FldInfo.clrArrange;
}

LenUnit oph::ImgDecoderOhc::getUnitOfWavlen() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (LenUnit)-1;
	}
	else
		return FldInfo.wavlenUnit;
}

CompresType oph::ImgDecoderOhc::getCompressedFormatType() {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return (CompresType)-1;
	}
	else
		return FldInfo.comprsType;
}

void oph::ImgDecoderOhc::getWavelength(std::vector<double_t> &wavlen_array) {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return;
	}
	else
		wavlen_array = WavLeng;
}

void oph::ImgDecoderOhc::getLinkFilePath(std::vector<std::string> &linkFilePath_array) {
	if ((this->Header == nullptr) || !this->bLoadFile) {
		LOG("OHC CODEC Error : No loaded data.");
		return;
	}
	else
		linkFilePath_array = this->linkFilePath;
}

bool oph::ImgDecoderOhc::load() {
	this->File.open(this->fname/*, std::ios::in | std::ios::trunc*/);

	bool isOpen = File.is_open();
	if (this->File.is_open()) {
		if (this->Header == nullptr)
			this->Header = new OHCheader();


		// Read OHC File Header
		File >> FHeader.fileSignature[0] >> FHeader.fileSignature[1];
		if ((FHeader.fileSignature[0] != FMT_SIGN_OHC[0]) || (FHeader.fileSignature[1] != FMT_SIGN_OHC[1])) {
			LOG("Not OHC File");
			return false;
		}
		else {
			File >> FHeader.fileSize;
			File >> FHeader.fileVersionMajor;
			File >> FHeader.fileVersionMinor;
			File >> FHeader.fileReserved1;
			File >> FHeader.fileReserved2;
			File >> FHeader.fileOffBytes;
			std::cout << "Reading Openholo Complex Field File..." << std::endl << fname << std::endl;
			std::cout << "OHC File was made on OpenHolo version " << (int)FHeader.fileVersionMajor << "." << (int)FHeader.fileVersionMinor << "..." << std::endl;
		}
		//this->File.read((char*)&FHeader, sizeof(OHCheader));
		//if ((FHeader.fileSignature[0] != FMT_SIGN_OHC[0]) || (FHeader.fileSignature[1] != FMT_SIGN_OHC[1])) {
		//	LOG("Not OHC File");
		//	return false;
		//}
		//else {
		//	std::cout << "Reading Openholo Complex Field File..." << std::endl << fname << std::endl;
		//	std::cout << "OHC File was made on OpenHolo version " << (int)FHeader.fileVersionMajor << "." << (int)FHeader.fileVersionMinor << "..." << std::endl;
		//}

		// Read Complex Field Info Header
		int pitchUnit, clrType, clrArrange, wavlenUnit, cmplxFldType, fldStore, fldCodeType, bPhaseCode, comprsType;
		File >> FldInfo.headerSize;
		File >> FldInfo.pxNumX;
		File >> FldInfo.pxNumY;
		File >> FldInfo.pxPitchX;
		File >> FldInfo.pxPitchY;
		File >> pitchUnit;
		File >> FldInfo.wavlenNum;
		File >> clrType;
		File >> clrArrange;
		File >> wavlenUnit;
		File >> cmplxFldType;
		File >> fldStore;
		File >> fldCodeType;
		File >> bPhaseCode;
		File >> FldInfo.phaseCodeMin;
		File >> FldInfo.phaseCodeMax;
		File >> FldInfo.fldSize;
		File >> comprsType;

		FldInfo.pitchUnit = (LenUnit)pitchUnit;
		FldInfo.clrType = (ColorType)clrType;
		FldInfo.clrArrange = (ColorArran)clrArrange;
		FldInfo.wavlenUnit = (LenUnit)wavlenUnit;
		FldInfo.cmplxFldType = (DataType)cmplxFldType;
		FldInfo.fldStore = (FldStore)fldStore;
		FldInfo.fldCodeType = (FldCodeType)fldCodeType;
		FldInfo.bPhaseCode = (BPhaseCode)bPhaseCode;
		FldInfo.comprsType = (CompresType)comprsType;

		//this->File.read((char*)&FldInfo, sizeof(OHCFIELDINFOHEADER));
		if (FldInfo.fldSize == 0) {
			LOG("Error : No Field Data");
			this->File.close();
			return false;
		}

		// Read Wavelength Table
		for (uint n = 0; n < FldInfo.wavlenNum; ++n) {
			double_t waveLength = 0.0;
			File >> waveLength;
			WavLeng.push_back(waveLength);
		}
		//for (uint n = 0; n < FldInfo.wavlenNum; ++n) {
		//	double_t wavelength = 0.;
		//	this->File.read((char*)&wavelength, sizeof(double_t));
		//	WavLeng.push_back(wavelength);
		//}

		// Decoding Field Data
		bool ok = false;
		switch (FldInfo.cmplxFldType) {
		case DataType::Float64:
		case DataType::Float32:
			ok = decodeFieldData();
			break;
		case DataType::CmprFmt: //파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
			LOG("Error : Compressed Image Format Decoding is Not Yet supported...");
			this->File.close();
			return false;
			break;
		default:
			LOG("Error : Invalid Decoding Complex Field Data Type...");
			this->File.close();
			return false;
			break;
		}
		//switch (FldInfo.cmplxFldType) {
		//case DataType::Float64:
		//	ok = decodeFieldData<double_t>();
		//	break;
		//case DataType::Float32:
		//	ok = decodeFieldData<float_t>();
		//	break;
		//case DataType::Int8:
		//	ok = decodeFieldData<int8_t>();
		//	break;
		//case DataType::Int16:
		//	ok = decodeFieldData<int16_t>();
		//	break;
		//case DataType::Int32:
		//	ok = decodeFieldData<int32_t>();
		//	break;
		//case DataType::Int64:
		//	ok = decodeFieldData<int64_t>();
		//	break;
		//case DataType::Uint8:
		//	ok = decodeFieldData<uint8_t>();
		//	break;
		//case DataType::Uint16:
		//	ok = decodeFieldData<uint16_t>();
		//	break;
		//case DataType::Uint32:
		//	ok = decodeFieldData<uint32_t>();
		//	break;
		//case DataType::Uint64:
		//	ok = decodeFieldData<uint64_t>();
		//	break;
		//case DataType::CmprFmt: //파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
		//	LOG("Error : Compressed Image Format Decoding is Not Yet supported...");
		//	this->File.close();
		//	return false;
		//	break;
		//default:
		//	LOG("Error : Invalid Decoding Complex Field Data Type...");
		//	this->File.close();
		//	return false;
		//	break;
		//}

		this->bLoadFile = true;
		this->File.close();
		return true;
	}
	else {
		LOG("Error : Failed loading OHC file...");
		return false;
	}
}

bool oph::ImgDecoderOhc::decodeFieldData()
{
	int n_wavlens = FldInfo.wavlenNum;
	int cols = FldInfo.pxNumX;
	int rows = FldInfo.pxNumY;
	int n_pixels = cols * rows;
	ulonglong n_fields = n_pixels * n_wavlens;

	int n_cmplxChnl = 0; // Is a data value Dual data(2) or Single data(1) ?

	switch (FldInfo.fldCodeType) {
	case FldCodeType::RI: {
		n_cmplxChnl = 2;
		for (int w = 0; w < n_wavlens; ++w) {
			OphComplexField data_field(cols, rows);
			data_field.zeros();
			this->field_cmplx.push_back(data_field);
		}
		break;
	}
	case FldCodeType::AP: {
		n_cmplxChnl = 2;
		for (int w = 0; w < n_wavlens; ++w) {
			OphRealField data_field(cols, rows);
			data_field.zeros();
			this->field_ampli.push_back(data_field);
			this->field_phase.push_back(data_field);
		}
		break;
	}
	case FldCodeType::AE: {
		n_cmplxChnl = 1;
		for (int w = 0; w < n_wavlens; ++w) {
			OphRealField data_field(cols, rows);
			data_field.zeros();
			this->field_ampli.push_back(data_field);
		}
		break;
	}
	case FldCodeType::PE: {
		n_cmplxChnl = 1;
		for (int w = 0; w < n_wavlens; ++w) {
			OphRealField data_field(cols, rows);
			data_field.zeros();
			this->field_phase.push_back(data_field);
		}
		break;
	}
	default: {
		LOG("Error : Invalid Complex Field Encoding Type...");
		return false;
		break;
	}
	}

	if (FldInfo.fldStore == FldStore::Directly) {
		if (FldInfo.cmplxFldType == DataType::Float32) {
			this->buf_f32 = new float[n_fields * n_cmplxChnl];
			for (ulonglong i = 0; i < n_fields * n_cmplxChnl; i++)
				File >> buf_f64[i];
			//this->File.read((char*)&this->buf_f32, FldInfo.fldSize);
		}
		else if (FldInfo.cmplxFldType == DataType::Float64) {
			this->buf_f64 = new double[n_fields * n_cmplxChnl];
			for (ulonglong i = 0; i < n_fields * n_cmplxChnl; i++)
				File >> buf_f64[i];
			//this->File.read((char*)&this->buf_f64, FldInfo.fldSize);
		}

		for (int x = 0; x < cols; ++x) {
			for (int y = 0; y < rows; ++y) {
				int idx = x * rows + y;

				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
					ulonglong idx_sqtlChnl = n_wavlens * idx + clrChnl;

					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								this->field_cmplx[clrChnl][x][y][_RE] = *(this->buf_f32 + idx_sqtlChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = *(this->buf_f32 + idx_sqtlChnl + 1 * n_fields);
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								this->field_cmplx[clrChnl][x][y][_RE] = *(this->buf_f64 + idx_sqtlChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = *(this->buf_f64 + idx_sqtlChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AP: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								this->field_ampli[clrChnl][x][y] = *(this->buf_f32 + idx_sqtlChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = *(this->buf_f32 + idx_sqtlChnl + 1 * n_fields);
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								this->field_ampli[clrChnl][x][y] = *(this->buf_f64 + idx_sqtlChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = *(this->buf_f64 + idx_sqtlChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AE: {
							if (FldInfo.cmplxFldType == DataType::Float32)
								this->field_ampli[clrChnl][x][y] = *(this->buf_f32 + idx_sqtlChnl + 0 * n_fields);
							else if (FldInfo.cmplxFldType == DataType::Float64)
								this->field_ampli[clrChnl][x][y] = *(this->buf_f64 + idx_sqtlChnl + 0 * n_fields);
							break;
						}
						case FldCodeType::PE: {
							if (FldInfo.cmplxFldType == DataType::Float32)
								this->field_phase[clrChnl][x][y] = *(this->buf_f32 + idx_sqtlChnl + 0 * n_fields);
							else if (FldInfo.cmplxFldType == DataType::Float64)
								this->field_phase[clrChnl][x][y] = *(this->buf_f64 + idx_sqtlChnl + 0 * n_fields);
							break;
						}
						}
					}
					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
						ulonglong idx_eachChnl = idx + clrChnl * n_pixels;

						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								this->field_cmplx[clrChnl][x][y][_RE] = *(this->buf_f32 + idx_eachChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = *(this->buf_f32 + idx_eachChnl + 1 * n_fields);
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								this->field_cmplx[clrChnl][x][y][_RE] = *(this->buf_f64 + idx_eachChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = *(this->buf_f64 + idx_eachChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AP: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								this->field_ampli[clrChnl][x][y] = *(this->buf_f32 + idx_eachChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = *(this->buf_f32 + idx_eachChnl + 1 * n_fields);
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								this->field_ampli[clrChnl][x][y] = *(this->buf_f64 + idx_eachChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = *(this->buf_f64 + idx_eachChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AE: {
							if (FldInfo.cmplxFldType == DataType::Float32)
								this->field_ampli[clrChnl][x][y] = *(this->buf_f32 + idx_eachChnl + 0 * n_fields);
							else if (FldInfo.cmplxFldType == DataType::Float64)
								this->field_ampli[clrChnl][x][y] = *(this->buf_f64 + idx_eachChnl + 0 * n_fields);
							break;
						}
						case FldCodeType::PE: {
							if (FldInfo.cmplxFldType == DataType::Float32)
								this->field_phase[clrChnl][x][y] = *(this->buf_f32 + idx_eachChnl + 0 * n_fields);
							else if (FldInfo.cmplxFldType == DataType::Float64)
								this->field_phase[clrChnl][x][y] = *(this->buf_f64 + idx_eachChnl + 0 * n_fields);
							break;
						}
						}
					}
				}
			}
		}
		return true;
	}
	else if (FldInfo.fldStore == FldStore::LinkFile) { // 데이터 타입으로 파일 직접 저장 말고, 이미지 포맷 링크 방식
		LOG("Error : Link Image File Decoding is Not Yet supported...");
		return false;
	}
	else {
		LOG("Error : Invalid Field Data Store Type...");
		return false;
	}
}

//template<typename T>
//bool oph::ImgDecoderOhc::decodeFieldData() {
//	// Data Type Info for Decoding
//	bool bIsInteger = std::numeric_limits<T>::is_integer; // only float, double, long double is false
//	//bool bIsSigned = std::numeric_limits<T>::is_signed; // unsigned type is false, bool is too false.
//	double max_T = (double)std::numeric_limits<T>::max();
//	double min_T = (double)std::numeric_limits<T>::min();
//
//	int n_wavlens = FldInfo.wavlenNum;
//	int cols = FldInfo.pxNumX;
//	int rows = FldInfo.pxNumY;
//	int n_pixels = cols * rows;
//	ulonglong n_fields = n_pixels * n_wavlens;
//
//	int n_cmplxChnl = 0; // Is a data value Dual data(2) or Single data(1) ?
//
//	switch (FldInfo.fldCodeType) {
//	case FldCodeType::RI: {
//		n_cmplxChnl = 2;
//		for (int w = 0; w < n_wavlens; ++w) {
//			OphComplexField data_field(cols, rows);
//			data_field.zeros();
//			this->field_cmplx.push_back(data_field);
//		}
//		break;
//	}
//	case FldCodeType::AP: {
//		n_cmplxChnl = 2;
//		for (int w = 0; w < n_wavlens; ++w) {
//			OphRealField data_field(cols, rows);
//			data_field.zeros();
//			this->field_ampli.push_back(data_field);
//			this->field_phase.push_back(data_field);
//		}
//		break;
//	}
//	case FldCodeType::AE: {
//		n_cmplxChnl = 1;
//		for (int w = 0; w < n_wavlens; ++w) {
//			OphRealField data_field(cols, rows);
//			data_field.zeros();
//			this->field_ampli.push_back(data_field);
//		}
//		break;
//	}
//	case FldCodeType::PE: {
//		n_cmplxChnl = 1;
//		for (int w = 0; w < n_wavlens; ++w) {
//			OphRealField data_field(cols, rows);
//			data_field.zeros();
//			this->field_phase.push_back(data_field);
//		}
//		break;
//	}
//	default: {
//		LOG("Error : Invalid Complex Field Encoding Type...");
//		return false;
//		break;
//	}
//	}
//
//	if (FldInfo.fldStore == FldStore::Directly) {
//		this->buf = new T[n_fields * n_cmplxChnl];
//		this->File.read((char*)&this->buf, FldInfo.fldSize);
//
//		for (int y = 0; y < rows; ++y) {
//			for (int x = 0; x < cols; ++x) {
//				int idx = y * cols + x;
//
//				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
//					ulonglong idx_sqtlChnl = n_wavlens * idx + clrChnl;
//
//					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
//						switch (FldInfo.fldCodeType) {
//						case FldCodeType::RI: {
//							if (!bIsInteger) { // floating type
//								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
//							}
//							else if (bIsInteger) { // integer type
//								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
//							}
//							break;
//						}
//						case FldCodeType::AP: {
//							if (!bIsInteger) {
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
//							}
//							else if (bIsInteger) {
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
//									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real phase = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
//									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						case FldCodeType::AE: {
//							if (!bIsInteger)
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//							else if (bIsInteger)
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//							break;
//						}
//						case FldCodeType::PE: {
//							if (!bIsInteger)
//								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//							else if (bIsInteger) {
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
//									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real phase = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
//									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						}
//					}
//					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
//						ulonglong idx_eachChnl = idx + clrChnl * n_pixels;
//
//						switch (FldInfo.fldCodeType) {
//						case FldCodeType::RI: {
//							if (!bIsInteger) { // floating type
//								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
//							}
//							else if (bIsInteger) { // integer type
//								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
//							}
//							break;
//						}
//						case FldCodeType::AP: {
//							if (!bIsInteger) {
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
//							}
//							else if (bIsInteger) {
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
//									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real phase = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
//									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						case FldCodeType::AE: {
//							if (!bIsInteger)
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//							else if (bIsInteger)
//								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//							break;
//						}
//						case FldCodeType::PE: {
//							if (!bIsInteger)
//								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//							else if (bIsInteger) {
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
//									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real phase = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
//									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						}
//					}
//				}
//			}
//		}
//		return true;
//	}
//	else if (FldInfo.fldStore == FldStore::LinkFile) { // 데이터 타입으로 파일 직접 저장 말고, 이미지 포맷 링크 방식
//		LOG("Error : Link Image File Decoding is Not Yet supported...");
//		return false;
//	}
//	else {
//		LOG("Error : Invalid Field Data Store Type...");
//		return false;
//	}
//}

//template<typename T>
//Real oph::ImgDecoderOhc::decodePhase(const T phase, const Real min_p, const Real max_p, const double min_T, const double max_T) {
//	// Normalize phase data type range to (0.0, 1.0)
//	Real _phase = ((double)phase - min_T) / (max_T - min_T);
//
//	// Mapping to (phaseCodeMin, phaseCodeMax)
//	if (std::is_same<double, Real>::value)
//		return (Real)(_phase*(max_p - min_p) + min_p)*M_PI;
//	else if (std::is_same<float, Real>::value)
//		return (Real)(_phase*(max_p - min_p) + min_p)*M_PI_F;
//}


/************************ OHC Encoder *****************************/

oph::ImgEncoderOhc::ImgEncoderOhc()
	: ImgCodecOhc()
{
	initOHCheader();
}

oph::ImgEncoderOhc::ImgEncoderOhc(const std::string &_fname, const OHCheader &_Header)
	: ImgCodecOhc(_fname, _Header)
{
	initOHCheader();
}

oph::ImgEncoderOhc::ImgEncoderOhc(const std::string &_fname)
	: ImgCodecOhc(_fname)
{
	initOHCheader();
}

oph::ImgEncoderOhc::~ImgEncoderOhc()
{
	this->releaseOHCheader();
	this->releaseFldData();
	this->releaseCodeBuffer();
}

void oph::ImgEncoderOhc::initOHCheader() {
	if (this->Header != nullptr)
		delete this->Header;

	this->Header = new OHCheader();

	//Set Initial Header of Encoder
	FHeader.fileSignature[0] = FMT_SIGN_OHC[0];
	FHeader.fileSignature[1] = FMT_SIGN_OHC[1];
	FHeader.fileVersionMajor = _OPH_LIB_VERSION_MAJOR_;
	FHeader.fileVersionMinor = _OPH_LIB_VERSION_MINOR_;
	FHeader.fileReserved1 = 0;
	FHeader.fileReserved2 = 0;

	//Set Initial Complex Field Information for Encoder
	FldInfo.headerSize = 0;
	FldInfo.pitchUnit = LenUnit::m;
	FldInfo.wavlenNum = 0;
	FldInfo.clrType = ColorType::MLT;
	FldInfo.clrArrange = ColorArran::EachChannel;
	FldInfo.wavlenUnit = LenUnit::m;
	FldInfo.fldStore = FldStore::Directly;
	FldInfo.fldCodeType = FldCodeType::RI;
	FldInfo.bPhaseCode = BPhaseCode::NotEncoded;
	FldInfo.phaseCodeMin = -1.0;
	FldInfo.phaseCodeMax = 1.0;
	FldInfo.comprsType = (CompresType)-1;

	if (std::is_same<double, Real>::value)
		FldInfo.cmplxFldType = DataType::Float64;
	else if (std::is_same<float, Real>::value)
		FldInfo.cmplxFldType = DataType::Float32;
}

void oph::ImgEncoderOhc::setNumOfPixel(const uint _pxNumX, const uint _pxNumY) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.pxNumX = _pxNumX;
		FldInfo.pxNumY = _pxNumY;
	}
}

void oph::ImgEncoderOhc::setNumOfPixel(const ivec2 _pxNum) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.pxNumX = _pxNum[_X];
		FldInfo.pxNumY = _pxNum[_Y];
	}
}

void oph::ImgEncoderOhc::setPixelPitch(const double _pxPitchX, const double _pxPitchY, const LenUnit unit) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.pxPitchX = _pxPitchX;
		FldInfo.pxPitchY = _pxPitchY;
		FldInfo.pitchUnit = unit;
	}
}

void oph::ImgEncoderOhc::setPixelPitch(const vec2 _pxPitch, const LenUnit unit) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.pxPitchX = _pxPitch[_X];
		FldInfo.pxPitchY = _pxPitch[_Y];
		FldInfo.pitchUnit = unit;
	}
}

void oph::ImgEncoderOhc::setNumOfWavlen(const uint n_wavlens) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.wavlenNum = n_wavlens;
	}
}

void oph::ImgEncoderOhc::setColorType(const ColorType _clrType) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.clrType = _clrType;
	}
}

void oph::ImgEncoderOhc::setColorArrange(const ColorArran _clrArrange) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.clrArrange = _clrArrange;
	}
}

void oph::ImgEncoderOhc::setUnitOfWavlen(const LenUnit unit) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.wavlenUnit = unit;
	}
}

//void oph::ImgEncoderOhc::setFieldEncoding(const FldStore _fldStore, const FldCodeType _fldCodeType, const DataType _cmplxFldType) {
void oph::ImgEncoderOhc::setFieldEncoding(const FldStore _fldStore, const FldCodeType _fldCodeType) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.fldStore = _fldStore;
		FldInfo.fldCodeType = _fldCodeType;
		//FldInfo.cmplxFldType = _cmplxFldType;
	}
}

void oph::ImgEncoderOhc::setPhaseEncoding(const BPhaseCode _bPhaseCode, const double _phaseCodeMin, const double _phaseCodeMax) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.bPhaseCode = _bPhaseCode;
		FldInfo.phaseCodeMin = _phaseCodeMin;
		FldInfo.phaseCodeMax = _phaseCodeMax;
	}
}


void oph::ImgEncoderOhc::setPhaseEncoding(const BPhaseCode _bPhaseCode, const vec2 _phaseCodeRange) {
	if (this->Header == nullptr) {
		LOG("OHC CODEC Error : No header data.");
		return;
	}
	else {
		FldInfo.bPhaseCode = _bPhaseCode;
		FldInfo.phaseCodeMin = _phaseCodeRange[0];
		FldInfo.phaseCodeMax = _phaseCodeRange[1];
	}
}

//void oph::ImgEncoderOhc::setCompressedFormatType(const CompresType _comprsType) {
//	if (this->Header == nullptr) {
//		LOG("OHC CODEC Error : No header data.");
//		return;
//	}
//	else {
//		FldInfo.comprsType = _comprsType;
//	}	
//}

void oph::ImgEncoderOhc::setWavelength(const Real _wavlen, const LenUnit _unit) {
	this->addWavelength(_wavlen);
	this->setUnitOfWavlen(_unit);
}

void oph::ImgEncoderOhc::addWavelengthNComplexFieldData(const Real wavlen, const OphComplexField &data) {
	this->addWavelength(wavlen);
	this->addComplexFieldData(data);
}

void oph::ImgEncoderOhc::addComplexFieldData(const OphComplexField &data) {
	this->field_cmplx.push_back(data);
}

void oph::ImgEncoderOhc::addComplexFieldData(const Complex<Real>* data)
{
	if (data == nullptr) {
		LOG("not found Complex data");
		return;
	}

	ivec2 buffer_size = ivec2(this->Header->fieldInfo.pxNumX, this->Header->fieldInfo.pxNumY);

	OphComplexField complexField(buffer_size[_X], buffer_size[_Y]);
	Buffer2Field(data, complexField, buffer_size);

	this->field_cmplx.push_back(complexField);
}

void oph::ImgEncoderOhc::addWavelength(const Real wavlen) {
	WavLeng.push_back(wavlen);
	this->setNumOfWavlen((uint32_t)WavLeng.size());
}

//void oph::ImgEncoderOhc::addLinkFilePath(const std::string &path) {
//	this->linkFilePath.push_back(path);
//}

bool oph::ImgEncoderOhc::save() {
	this->File.open(this->fname, std::ios::out | std::ios::trunc);

	//FILE *fp;
	//fopen_s(&fp, this->fname.c_str(), "w");
	//if (fp == nullptr) return false;

	//if (fp) {
	if (this->File.is_open()) {
		if (this->Header == nullptr) {
			this->Header = new OHCheader();
			this->initOHCheader();
		}

		// Encoding Field Data
		uint64_t dataSize = 0;
		switch (FldInfo.cmplxFldType) {
		case DataType::Float64:
		case DataType::Float32:
			dataSize = encodeFieldData();
			break;
		case DataType::CmprFmt: // 파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
			LOG("Error : Compressed Image Format Encoding is Not Yet supported...");
			//fclose(fp);
			this->File.close();
			return false;
			break;
		default:
			LOG("Error : Invalid Encoding Complex Field Data Type...");
			//fclose(fp);
			this->File.close();
			return false;
			break;
		}
		//switch (FldInfo.cmplxFldType) {
		//case DataType::Float64:
		//	dataSize = encodeFieldData<double_t>();
		//	break;
		//case DataType::Float32:
		//	dataSize = encodeFieldData<float_t>();
		//	break;
		//case DataType::Int8:
		//	dataSize = encodeFieldData<int8_t>();
		//	break;
		//case DataType::Int16:
		//	dataSize = encodeFieldData<int16_t>();
		//	break;
		//case DataType::Int32:
		//	dataSize = encodeFieldData<int32_t>();
		//	break;
		//case DataType::Int64:
		//	dataSize = encodeFieldData<int64_t>();
		//	break;
		//case DataType::Uint8:
		//	dataSize = encodeFieldData<uint8_t>();
		//	break;
		//case DataType::Uint16:
		//	dataSize = encodeFieldData<uint16_t>();
		//	break;
		//case DataType::Uint32:
		//	dataSize = encodeFieldData<uint32_t>();
		//	break;
		//case DataType::Uint64:
		//	dataSize = encodeFieldData<uint64_t>();
		//	break;
		//case DataType::CmprFmt: // 파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
		//	LOG("Error : Compressed Image Format Encoding is Not Yet supported...");
		//	//fclose(fp);
		//	this->File.close();
		//	return false;
		//	break;
		//default:
		//	LOG("Error : Invalid Encoding Complex Field Data Type...");
		//	//fclose(fp);
		//	this->File.close();
		//	return false;
		//	break;
		//}

		// Set data for Field Size
		uint64_t wavlenTableSize = FldInfo.wavlenNum * sizeof(double_t);

		if (dataSize == 0) {
			LOG("Error : No Field Data");
			//fclose(fp);
			this->File.close();
			return false;
		}
		else {
			if (FldInfo.cmplxFldType != DataType::CmprFmt)
				FldInfo.comprsType = (CompresType)-1;

			FldInfo.headerSize = (uint32_t)(sizeof(OHCFIELDINFOHEADER) + wavlenTableSize);
			FldInfo.fldSize = dataSize;
			FHeader.fileSize = sizeof(OHCFILEHEADER) + FldInfo.headerSize + FldInfo.fldSize;
			FHeader.fileOffBytes = sizeof(OHCFILEHEADER) + FldInfo.headerSize;
		}

		// write File Header
		//fwrite(&FHeader, 1, sizeof(OHCheader), fp);
		//this->File.write((char*)&FHeader, sizeof(OHCheader));
		File << FHeader.fileSignature[0] << FHeader.fileSignature[1] << "\n";
		File << FHeader.fileSize << "\n";
		File << FHeader.fileVersionMajor << "\n";
		File << FHeader.fileVersionMinor << "\n";
		File << FHeader.fileReserved1 << "\n";
		File << FHeader.fileReserved2 << "\n";
		File << FHeader.fileOffBytes << "\n";

		// write Field Info Header
		//fwrite(&FldInfo, 1, sizeof(OHCFIELDINFOHEADER), fp);
		//this->File.write((char*)&FldInfo, sizeof(OHCFIELDINFOHEADER));
		File << FldInfo.headerSize << "\n";
		File << FldInfo.pxNumX << "\n";
		File << FldInfo.pxNumY << "\n";
		File << FldInfo.pxPitchX << "\n";
		File << FldInfo.pxPitchY << "\n";
		File << (int)FldInfo.pitchUnit << "\n";
		File << FldInfo.wavlenNum << "\n";
		File << (int)FldInfo.clrType << "\n";
		File << (int)FldInfo.clrArrange << "\n";
		File << (int)FldInfo.wavlenUnit << "\n";
		File << (int)FldInfo.cmplxFldType << "\n";
		File << (int)FldInfo.fldStore << "\n";
		File << (int)FldInfo.fldCodeType << "\n";
		File << (int)FldInfo.bPhaseCode << "\n";
		File << FldInfo.phaseCodeMin << "\n";
		File << FldInfo.phaseCodeMax << "\n";
		File << FldInfo.fldSize << "\n";
		File << (int)FldInfo.comprsType << "\n";

		// write Wavelength Table
		for (uint n = 0; n < FldInfo.wavlenNum; ++n) {
			double_t waveLength = WavLeng[n];
			//fwrite(&waveLength, 1, sizeof(double_t), fp);
			//this->File.write((char*)&waveLength, sizeof(double_t));
			File << waveLength << "\n";
		}

		// write Complex Field Data
		//fwrite(this->buf, 1, sizeof(dataSize), fp);
		if (FldInfo.cmplxFldType == DataType::Float32) 
		{
			size_t dataTypeSize = sizeof(float);
			ulonglong maxIdx = dataSize / dataTypeSize;
			for (ulonglong i = 0; i < maxIdx; i++)
				File << buf_f32[i] << " ";

			//this->File.write((char*)this->buf_f32, sizeof(dataSize));
		}
		else if (FldInfo.cmplxFldType == DataType::Float64) 
		{
			size_t dataTypeSize = sizeof(double);
			ulonglong maxIdx = dataSize / dataTypeSize;
			for (ulonglong i = 0; i < maxIdx; i++)
				File << buf_f64[i] << " ";

			//this->File.write((char*)this->buf_f64, sizeof(dataSize));
		}
		//this->File.write((char*)this->buf, sizeof(dataSize));

		//fclose(fp);
		this->File.close();
		return true;
	}
	else {
		LOG("Error : Failed saving OHC file...");
		return false;
	}
}

uint64_t oph::ImgEncoderOhc::encodeFieldData()
{
	ulonglong dataSizeBytes = 0;
	int n_wavlens = FldInfo.wavlenNum;
	int cols = FldInfo.pxNumX;
	int rows = FldInfo.pxNumY;
	int n_pixels = cols * rows;
	ulonglong n_fields = n_pixels * n_wavlens;

	int n_cmplxChnl = 0; // Is a data value Dual data(2) or Single data(1) ?
	if ((FldInfo.fldCodeType == FldCodeType::AP) || (FldInfo.fldCodeType == FldCodeType::RI))
		n_cmplxChnl = 2;
	else if ((FldInfo.fldCodeType == FldCodeType::AE) || (FldInfo.fldCodeType == FldCodeType::PE))
		n_cmplxChnl = 1;

	if (FldInfo.fldStore == FldStore::Directly) {
		if (FldInfo.cmplxFldType == DataType::Float32) {
			dataSizeBytes = sizeof(float) * n_fields * n_cmplxChnl;
			this->buf_f32 = new float[n_fields * n_cmplxChnl];
			std::memset(this->buf_f32, NULL, dataSizeBytes);
		}
		else if (FldInfo.cmplxFldType == DataType::Float64) {
			dataSizeBytes = sizeof(double) * n_fields * n_cmplxChnl;
			this->buf_f64 = new double[n_fields * n_cmplxChnl];
			std::memset(this->buf_f64, NULL, dataSizeBytes);
		}

		for (int x = 0; x < cols; ++x) {
			for (int y = 0; y < rows; ++y) {
				int idx = x * rows + y;

				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
					ulonglong idx_sqtlChnl = n_wavlens * idx + clrChnl;

					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_sqtlChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y][_RE];
								*(this->buf_f32 + idx_sqtlChnl + 1 * n_fields) = (float)this->field_cmplx[clrChnl][x][y][_IM];
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_sqtlChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y][_RE];
								*(this->buf_f64 + idx_sqtlChnl + 1 * n_fields) = (double)this->field_cmplx[clrChnl][x][y][_IM];
							}
							break;
						}
						case FldCodeType::AP: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_sqtlChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].mag();
								*(this->buf_f32 + idx_sqtlChnl + 1 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_sqtlChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].mag();
								*(this->buf_f64 + idx_sqtlChnl + 1 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].angle();
							}
							break;
						}
						case FldCodeType::AE: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_sqtlChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].mag();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_sqtlChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].mag();
							}
							break;
						}
						case FldCodeType::PE: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_sqtlChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_sqtlChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].angle();
							}
							break;
						}
						}
					}
					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
						ulonglong idx_eachChnl = idx + clrChnl * n_pixels;

						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_eachChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y][_RE];
								*(this->buf_f32 + idx_eachChnl + 1 * n_fields) = (float)this->field_cmplx[clrChnl][x][y][_IM];
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_eachChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y][_RE];
								*(this->buf_f64 + idx_eachChnl + 1 * n_fields) = (double)this->field_cmplx[clrChnl][x][y][_IM];
							}
							break;
						}
						case FldCodeType::AP: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_eachChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].mag();
								*(this->buf_f32 + idx_eachChnl + 1 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_eachChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].mag();
								*(this->buf_f64 + idx_eachChnl + 1 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].angle();
							}
							break;
						}
						case FldCodeType::AE: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_eachChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].mag();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_eachChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].mag();
							}
							break;
						}
						case FldCodeType::PE: {
							if (FldInfo.cmplxFldType == DataType::Float32) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f32 + idx_eachChnl + 0 * n_fields) = (float)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (FldInfo.cmplxFldType == DataType::Float64) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*(this->buf_f64 + idx_eachChnl + 0 * n_fields) = (double)this->field_cmplx[clrChnl][x][y].angle();
							}
							break;
						}
						}
					}
				}
			}
		}
		return dataSizeBytes;
	}
	else if (FldInfo.fldStore == FldStore::LinkFile) { // 데이터 타입으로 파일 직접 저장 말고, 이미지 포맷 링크 방식
		LOG("Error : Link Image File Encoding is Not Yet supported...");
		return dataSizeBytes;
	}
	else {
		LOG("Error : Invalid Field Data Store Type...");
		return 0;
	}
}

//template<typename T>
//uint64_t oph::ImgEncoderOhc::encodeFieldData() {
//	// Data Type Info for Encoding
//	bool bIsInteger = std::numeric_limits<T>::is_integer; // only float, double, long double is false
//	//bool bIsSigned = std::numeric_limits<T>::is_signed; // unsigned type is false, bool is too false.
//	double max_T = (double)std::numeric_limits<T>::max();
//	double min_T = (double)std::numeric_limits<T>::min();
//
//	ulonglong dataSizeBytes = 0;
//	int n_wavlens = FldInfo.wavlenNum;
//	int cols = FldInfo.pxNumX;
//	int rows = FldInfo.pxNumY;
//	int n_pixels = cols * rows;
//	ulonglong n_fields = n_pixels * n_wavlens;
//
//	int n_cmplxChnl = 0; // Is a data value Dual data(2) or Single data(1) ?
//	if ((FldInfo.fldCodeType == FldCodeType::AP) || (FldInfo.fldCodeType == FldCodeType::RI))
//		n_cmplxChnl = 2;
//	else if ((FldInfo.fldCodeType == FldCodeType::AE) || (FldInfo.fldCodeType == FldCodeType::PE))
//		n_cmplxChnl = 1;
//
//	if (FldInfo.fldStore == FldStore::Directly) {
//		dataSizeBytes = sizeof(T) * n_fields * n_cmplxChnl;
//		this->buf = new T[n_fields * n_cmplxChnl];
//		std::memset(this->buf, NULL, dataSizeBytes);
//
//		for (int y = 0; y < rows; ++y) {
//			for (int x = 0; x < cols; ++x) {
//				int idx = y * cols + x;
//
//				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
//					ulonglong idx_sqtlChnl = n_wavlens * idx + clrChnl;
//
//					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
//						switch (FldInfo.fldCodeType) {
//						case FldCodeType::RI: {
//							if (!bIsInteger) { // floating type
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*(((T*)this->buf) + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
//								*(((T*)this->buf) + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
//							}
//							else if (bIsInteger) { // integer type
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*(((T*)this->buf) + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
//								*(((T*)this->buf) + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
//							}
//							break;
//						}
//						case FldCodeType::AP: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//								*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//							}
//							else if (bIsInteger) {
//								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
//									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//									*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//								}
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
//									*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						case FldCodeType::AE: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//							}
//							else if (bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//							}
//							break;
//						}
//						case FldCodeType::PE: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//							}
//							else if (bIsInteger) {
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
//									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//									*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//								}
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
//									*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						}
//					}
//					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
//						ulonglong idx_eachChnl = idx + clrChnl * n_pixels;
//
//						switch (FldInfo.fldCodeType) {
//						case FldCodeType::RI: {
//							if (!bIsInteger) { // floating type
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*(((T*)this->buf) + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
//								*(((T*)this->buf) + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
//							}
//							else if (bIsInteger) { // integer type
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
//								*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
//							}
//							break;
//						}
//						case FldCodeType::AP: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//								*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//							}
//							else if (bIsInteger) {
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
//									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//									*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//								}
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
//									*((T*)this->buf + idx_eachChnl + 1 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						case FldCodeType::AE: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//							}
//							else if (bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
//							}
//							break;
//						}
//						case FldCodeType::PE: {
//							if (!bIsInteger) {
//								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//							}
//							else if (bIsInteger) {
//								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
//									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
//									*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
//								}
//								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
//									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
//									*((T*)this->buf + idx_eachChnl + 0 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
//								}
//							}
//							break;
//						}
//						}
//					}
//				}
//			}
//		}
//		return dataSizeBytes;
//	}
//	else if (FldInfo.fldStore == FldStore::LinkFile) { // 데이터 타입으로 파일 직접 저장 말고, 이미지 포맷 링크 방식
//		LOG("Error : Link Image File Encoding is Not Yet supported...");
//		return dataSizeBytes;
//	}
//	else {
//		LOG("Error : Invalid Field Data Store Type...");
//		return 0;
//	}
//}
//
//template<typename T>
//T oph::ImgEncoderOhc::encodePhase(const Real phase_angle, const Real min_p, const Real max_p, const double min_T, const double max_T) {
//	// Normalize phase (phaseCodeMin, phaseCodeMax) to (0.0, 1.0)
//	Real _phase;
//	if (std::is_same<double, Real>::value)
//		_phase = (phase_angle - min_p * M_PI) / ((max_p - min_p) * M_PI);
//	else if (std::is_same<float, Real>::value)
//		_phase = (phase_angle - min_p * M_PI_F) / ((max_p - min_p) * M_PI_F);
//
//	// Mapping to data type range
//	return (T)(_phase * (max_T - min_T) + min_T);
//}