#include "ImgCodecOhc.h"

#define NOMINMAX // using std::numeric_limits<DataType>::max(), min() of <limits> instead of <minwindef.h>

#include "sys.h"
#include <limits> // limit value of each data types


#define FHeader this->Header->fileHeader
#define FldInfo this->Header->fieldInfo
#define WavLeng this->Header->wavlenTable


#define FMT_SIGN_OHC "OH" // File Format Signature : 0x484F
#define LINK_IMG_PATH_SIZE 4 * 1024 * sizeof(BYTE) // 4KB


/************************ OHC CODEC *****************************/

oph::ImgCodecOhc::ImgCodecOhc() {
}

oph::ImgCodecOhc::~ImgCodecOhc() {
	this->releaseOHCheader();
	this->releaseFldData();
	this->releaseCodeBuffer();
}

oph::ImgCodecOhc::ImgCodecOhc(const std::string &_fname) {
	this->setFileName(_fname);
}

oph::ImgCodecOhc::ImgCodecOhc(const std::string &_fname, const OHCheader &_Header) {
	this->setFileName(_fname);
	this->setOHCheader(_Header);
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
	delete[] this->buf;
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
	: ImgCodecOhc(_fname) {
}

oph::ImgDecoderOhc::ImgDecoderOhc(const std::string &_fname, const OHCheader &_Header)
	: ImgCodecOhc(_fname, _Header) {
}

oph::ImgDecoderOhc::~ImgDecoderOhc() {
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
}

bool oph::ImgDecoderOhc::load() {
	this->File.open(this->fname, std::ios::in | std::ios::trunc);

	if (this->File.is_open()) {
		// Read OHC File Header
		this->File.read((char*)&FHeader, sizeof(OHCheader));
		if ((FHeader.fileSignature[0] != FMT_SIGN_OHC[0]) || (FHeader.fileSignature[1] != FMT_SIGN_OHC[1])) {
			LOG("Not OHC File");
			return false;
		}
		else {
			if (this->Header != nullptr) delete this->Header;

			this->Header = new OHCheader;
			this->Header->fileHeader = FHeader;

			std::cout << "Reading Openholo Complex Field File..." << std::endl << fname << std::endl;
			std::cout << "OHC File was made on OpenHolo version " << (int)FHeader.fileVersionMajor << "." << (int)FHeader.fileVersionMinor << "..." << std::endl;
		}

		// Read Complex Field Info Header
		this->File.read((char*)&FldInfo, sizeof(OHCFIELDINFOHEADER));
		if (FldInfo.fldSize == 0) {
			LOG("Error : No Field Data");
			this->File.close();
			return false;
		}

		// Read Wavelength Table
		for (uint n = 0; n < FldInfo.wavlenNum; ++n) {
			double_t wavelength = 0.;
			this->File.read((char*)&wavelength, sizeof(double_t));
			WavLeng.push_back(wavelength);
		}

		// Decoding Field Data
		bool ok = false;
		switch (FldInfo.cmplxFldType) {
		case DataType::Float64:
			ok = decodeFieldData<double_t>();
			break;
		case DataType::Float32:
			ok = decodeFieldData<float_t>();
			break;
		case DataType::Int8:
			ok = decodeFieldData<int8_t>();
			break;
		case DataType::Int16:
			ok = decodeFieldData<int16_t>();
			break;
		case DataType::Int32:
			ok = decodeFieldData<int32_t>();
			break;
		case DataType::Int64:
			ok = decodeFieldData<int64_t>();
			break;
		case DataType::Uint8:
			ok = decodeFieldData<uint8_t>();
			break;
		case DataType::Uint16:
			ok = decodeFieldData<uint16_t>();
			break;
		case DataType::Uint32:
			ok = decodeFieldData<uint32_t>();
			break;
		case DataType::Uint64:
			ok = decodeFieldData<uint64_t>();
			break;
		case DataType::ImgFmt: //파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
			LOG("Error : Image Format Decoding is Not Yet supported...");
			this->File.close();
			return false;
			break;
		default:
			LOG("Error : Invalid Decoding Complex Field Data Type...");
			this->File.close();
			return false;
			break;
		}

		this->File.close();
		this->releaseCodeBuffer();
		return true;
	}
	else {
		LOG("Error : Failed loading OHC file...");
		return false;
	}
}

template<typename T>
bool oph::ImgDecoderOhc::decodeFieldData() {
	// Data Type Info for Decoding
	bool bIsInteger = std::numeric_limits<T>::is_integer; // only float, double, long double is false
	//bool bIsSigned = std::numeric_limits<T>::is_signed; // unsigned type is false, bool is too false.
	double max_T = (double)std::numeric_limits<T>::max();
	double min_T = (double)std::numeric_limits<T>::min();

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
		this->buf = new T[n_fields * n_cmplxChnl];
		this->File.read((char*)&this->buf, FldInfo.fldSize);

		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				int idx = y * cols + x;

				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
					int idx_sqtlChnl = n_wavlens * idx + clrChnl;

					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (!bIsInteger) { // floating type
								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
							}
							else if (bIsInteger) { // integer type
								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AP: {
							if (!bIsInteger) {
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
							}
							else if (bIsInteger) {
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);

								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real phase = (Real)*((T*)this->buf + idx_sqtlChnl + 1 * n_fields);
									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						case FldCodeType::AE: {
							if (!bIsInteger)
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
							else if (bIsInteger)
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
							break;
						}
						case FldCodeType::PE: {
							if (!bIsInteger)
								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
							else if (bIsInteger) {
								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real phase = (Real)*((T*)this->buf + idx_sqtlChnl + 0 * n_fields);
									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						}
					}
					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
						int idx_eachChnl = idx + clrChnl * n_pixels;

						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (!bIsInteger) { // floating type
								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
							}
							else if (bIsInteger) { // integer type
								this->field_cmplx[clrChnl][x][y][_RE] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
								this->field_cmplx[clrChnl][x][y][_IM] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
							}
							break;
						}
						case FldCodeType::AP: {
							if (!bIsInteger) {
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
							}
							else if (bIsInteger) {
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);

								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real phase = (Real)*((T*)this->buf + idx_eachChnl + 1 * n_fields);
									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						case FldCodeType::AE: {
							if (!bIsInteger)
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
							else if (bIsInteger)
								this->field_ampli[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
							break;
						}
						case FldCodeType::PE: {
							if (!bIsInteger)
								this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
							else if (bIsInteger) {
								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded)
									this->field_phase[clrChnl][x][y] = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real phase = (Real)*((T*)this->buf + idx_eachChnl + 0 * n_fields);
									this->field_phase[clrChnl][x][y] = this->decodePhase<T>(phase, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
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

template<typename T>
Real oph::ImgDecoderOhc::decodePhase(const T phase, const Real min_p, const Real max_p, const double min_T, const double max_T) {
	// Normalize phase data type range to (0.0, 1.0)
	Real _phase = ((double)phase - min_T) / (max_T - min_T);

	// Mapping to (phaseCodeMin, phaseCodeMax)
	if (std::is_same<double, Real>::value)
		return (Real)(_phase*(max_p - min_p) + min_p)*M_PI;
	else if (std::is_same<float, Real>::value)
		return (Real)(_phase*(max_p - min_p) + min_p)*M_PI_F;
}


/************************ OHC Encoder *****************************/

oph::ImgEncoderOhc::ImgEncoderOhc()
	: ImgCodecOhc()
{
}

oph::ImgEncoderOhc::ImgEncoderOhc(const std::string &_fname, const OHCheader &_Header)
	: ImgCodecOhc(_fname, _Header) {
}

oph::ImgEncoderOhc::ImgEncoderOhc(const std::string &_fname)
	: ImgCodecOhc(_fname) {
}

oph::ImgEncoderOhc::~ImgEncoderOhc() {
}

void oph::ImgEncoderOhc::setNumOfPixel(const uint _pxNumX, const uint _pxNumY) {
	this->Header->fieldInfo.pxNumX = _pxNumX;
	this->Header->fieldInfo.pxNumY = _pxNumY;
}

void oph::ImgEncoderOhc::setNumOfPixel(const ivec2 _pxNum) {
	this->Header->fieldInfo.pxNumX = _pxNum[_X];
	this->Header->fieldInfo.pxNumX = _pxNum[_Y];
}

void oph::ImgEncoderOhc::setPixelPitch(const double _pxPitchX, const double _pxPitchY, const LenUnit unit) {
	this->Header->fieldInfo.pxPitchX = _pxPitchX;
	this->Header->fieldInfo.pxPitchY = _pxPitchY;
	this->Header->fieldInfo.pitchUnit = unit;
}

void oph::ImgEncoderOhc::setPixelPitch(const vec2 _pxPitch, const LenUnit unit) {
	this->Header->fieldInfo.pxPitchX = _pxPitch[_X];
	this->Header->fieldInfo.pxPitchY = _pxPitch[_Y];
	this->Header->fieldInfo.pitchUnit = unit;
}

void oph::ImgEncoderOhc::setNumOfWavlen(const uint n_wavlens) {
	this->Header->fieldInfo.wavlenNum = n_wavlens;
}

void oph::ImgEncoderOhc::setColorType(const ColorType _clrType) {
	this->Header->fieldInfo.clrType = _clrType;
}

void oph::ImgEncoderOhc::setColorArrange(const ColorArran _clrArrange) {
	this->Header->fieldInfo.clrArrange = _clrArrange;
}

void oph::ImgEncoderOhc::setUnitOfWavlen(const LenUnit unit) {
	this->Header->fieldInfo.wavlenUnit = unit;
}

void oph::ImgEncoderOhc::setFieldEncoding(const FldStore _fldStore, const FldCodeType _fldCodeType, const DataType _cmplxFldType) {
	this->Header->fieldInfo.fldStore = _fldStore;
	this->Header->fieldInfo.fldCodeType = _fldCodeType;
	this->Header->fieldInfo.cmplxFldType = _cmplxFldType;
}

void oph::ImgEncoderOhc::setPhaseEncoding(const BPhaseCode _bPhaseCode, const double _phaseCodeMin, const double _phaseCodeMax) {
	this->Header->fieldInfo.bPhaseCode = _bPhaseCode;
	this->Header->fieldInfo.phaseCodeMin = _phaseCodeMin;
	this->Header->fieldInfo.phaseCodeMax = _phaseCodeMax;
}


void oph::ImgEncoderOhc::setPhaseEncoding(const BPhaseCode _bPhaseCode, const vec2 _phaseCodeRange) {
	this->Header->fieldInfo.bPhaseCode = _bPhaseCode;
	this->Header->fieldInfo.phaseCodeMin = _phaseCodeRange[0];
	this->Header->fieldInfo.phaseCodeMax = _phaseCodeRange[1];;
}

void oph::ImgEncoderOhc::setImageFormat(const ImageFormat _imgFmt) {
	this->Header->fieldInfo.imgFmt = _imgFmt;
}

void oph::ImgEncoderOhc::setWavelength(const Real _wavlen, const LenUnit _unit) {
	push_back_Wavlen(_wavlen);
	setUnitOfWavlen(_unit);
}

void oph::ImgEncoderOhc::push_back_WaveFld(const Real wavlen, const OphComplexField &data) {
	push_back_Wavlen(wavlen);
	push_back_FldData(data);
}

void oph::ImgEncoderOhc::push_back_FldData(const OphComplexField &data) {
	this->field_cmplx.push_back(data);
}

void oph::ImgEncoderOhc::push_back_Wavlen(const Real wavlen) {
	this->Header->wavlenTable.push_back(wavlen);
	this->Header->fieldInfo.wavlenNum = (uint32_t)this->Header->wavlenTable.size();
}

void oph::ImgEncoderOhc::push_back_LinkFilePath(const std::string &path) {
	this->linkFilePath.push_back(path);
}

void oph::ImgEncoderOhc::getLinkFilePath(const int idx, std::string &path) {
	path = this->linkFilePath[idx];
}

bool oph::ImgEncoderOhc::save() {
	this->File.open(this->fname, std::ios::out | std::ios::trunc);

	if (this->File.is_open()) {
		// Encoding Field Data
		uint64_t dataSize = 0;
		switch (FldInfo.cmplxFldType) {
		case DataType::Float64:
			dataSize = encodeFieldData<double_t>();
			break;
		case DataType::Float32:
			dataSize = encodeFieldData<float_t>();
			break;
		case DataType::Int8:
			dataSize = encodeFieldData<int8_t>();
			break;
		case DataType::Int16:
			dataSize = encodeFieldData<int16_t>();
			break;
		case DataType::Int32:
			dataSize = encodeFieldData<int32_t>();
			break;
		case DataType::Int64:
			dataSize = encodeFieldData<int64_t>();
			break;
		case DataType::Uint8:
			dataSize = encodeFieldData<uint8_t>();
			break;
		case DataType::Uint16:
			dataSize = encodeFieldData<uint16_t>();
			break;
		case DataType::Uint32:
			dataSize = encodeFieldData<uint32_t>();
			break;
		case DataType::Uint64:
			dataSize = encodeFieldData<uint64_t>();
			break;
		case DataType::ImgFmt: // 파일 링크 아님, 이미지 코덱을 직접 저장하는 방식
			LOG("Error : Image Format Encoding is Not Yet supported...");
			this->File.close();
			return false;
			break;
		default:
			LOG("Error : Invalid Encoding Complex Field Data Type...");
			this->File.close();
			return false;
			break;
		}

		// Set data for Field Size
		uint64_t wavlenTableSize = FldInfo.wavlenNum * sizeof(double_t);

		if (dataSize == 0) {
			LOG("Error : No Field Data");
			this->File.close();
			return false;
		}
		else {
			FldInfo.headerSize = (uint32_t)(sizeof(OHCFIELDINFOHEADER) + wavlenTableSize);
			FldInfo.fldSize = dataSize;

			if (FldInfo.cmplxFldType != DataType::ImgFmt)
				FldInfo.imgFmt = ImageFormat::RAW;
		}

		// Set File Header
		FHeader.fileSignature[0] = FMT_SIGN_OHC[0];
		FHeader.fileSignature[1] = FMT_SIGN_OHC[1];
		FHeader.fileSize = sizeof(OHCFILEHEADER) + FldInfo.headerSize + FldInfo.fldSize;
		FHeader.fileVersionMajor = _OPH_LIB_VERSION_MAJOR_;
		FHeader.fileVersionMinor = _OPH_LIB_VERSION_MINOR_;
		FHeader.fileReserved1 = 0;
		FHeader.fileReserved2 = 0;
		FHeader.fileOffBytes = sizeof(OHCFILEHEADER) + FldInfo.headerSize;

		// write File Header
		this->File.write((char*)&FHeader, sizeof(OHCheader));

		// write Field Info Header
		this->File.write((char*)&FldInfo, sizeof(OHCFIELDINFOHEADER));

		// write Wavelength Table
		for (uint n = 0; n < FldInfo.wavlenNum; ++n) {
			double_t waveLength = WavLeng[n];
			this->File.write((char*)&waveLength, sizeof(double_t));
		}

		// write Complex Field Data
		this->File.write((char*)this->buf, sizeof(dataSize));

		this->File.close();
		this->releaseCodeBuffer();
		return true;
	}
	else {
		LOG("Error : Failed saving OHC file...");
		return false;
	}
}

template<typename T>
uint64_t oph::ImgEncoderOhc::encodeFieldData() {
	// Data Type Info for Encoding
	bool bIsInteger = std::numeric_limits<T>::is_integer; // only float, double, long double is false
	//bool bIsSigned = std::numeric_limits<T>::is_signed; // unsigned type is false, bool is too false.
	double max_T = (double)std::numeric_limits<T>::max();
	double min_T = (double)std::numeric_limits<T>::min();

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
		dataSizeBytes = sizeof(T) * n_fields * n_cmplxChnl;
		this->buf = new T[n_fields * n_cmplxChnl];
		std::memset(this->buf, NULL, dataSizeBytes);

		for (int y = 0; y < rows; ++y) {
			for (int x = 0; x < cols; ++x) {
				int idx = y * cols + x;

				for (int clrChnl = 0; clrChnl < n_wavlens; ++clrChnl) { // RGB is wavlenNum == 3
					int idx_sqtlChnl = n_wavlens * idx + clrChnl;

					if (FldInfo.clrArrange == ColorArran::SequentialRGB) {
						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (!bIsInteger) { // floating type
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
								*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
							}
							else if (bIsInteger) { // integer type
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
								*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
							}
							break;
						}
						case FldCodeType::AP: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
								*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (bIsInteger) {
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();

								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
									*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
								}
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
									*((T*)this->buf + idx_sqtlChnl + 1 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						case FldCodeType::AE: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
							}
							else if (bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
							}
							break;
						}
						case FldCodeType::PE: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (bIsInteger) {
								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
									*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
								}
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
									*((T*)this->buf + idx_sqtlChnl + 0 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						}
					}
					else if (FldInfo.clrArrange == ColorArran::EachChannel) {
						int idx_eachChnl = idx + clrChnl * n_pixels;

						switch (FldInfo.fldCodeType) {
						case FldCodeType::RI: {
							if (!bIsInteger) { // floating type
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
								*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
							}
							else if (bIsInteger) { // integer type
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_RE];
								*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y][_IM];
							}
							break;
						}
						case FldCodeType::AP: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
								*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (bIsInteger) {
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();

								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
									*((T*)this->buf + idx_eachChnl + 1 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
								}
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
									*((T*)this->buf + idx_eachChnl + 1 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
							}
							break;
						}
						case FldCodeType::AE: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
							}
							else if (bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].mag();
							}
							break;
						}
						case FldCodeType::PE: {
							if (!bIsInteger) {
								setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
								*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
							}
							else if (bIsInteger) {
								if (FldInfo.bPhaseCode == BPhaseCode::NotEncoded) {
									setPhaseEncoding(BPhaseCode::NotEncoded, -1.0, 1.0);
									*((T*)this->buf + idx_eachChnl + 0 * n_fields) = (T)this->field_cmplx[clrChnl][x][y].angle();
								}
								else if (FldInfo.bPhaseCode == BPhaseCode::Encoded) {
									Real angle = this->field_cmplx[clrChnl][x][y].angle(); //atan2 : return -3.141592(-1.*PI) ~ 3.141592(1.*PI)
									*((T*)this->buf + idx_eachChnl + 0 * n_fields) = this->encodePhase<T>(angle, FldInfo.phaseCodeMin, FldInfo.phaseCodeMax, min_T, max_T);
								}
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

template<typename T>
T oph::ImgEncoderOhc::encodePhase(const Real phase_angle, const Real min_p, const Real max_p, const double min_T, const double max_T) {
	// Normalize phase (phaseCodeMin, phaseCodeMax) to (0.0, 1.0)
	Real _phase;
	if (std::is_same<double, Real>::value)
		_phase = (phase_angle - min_p * M_PI) / ((max_p - min_p) * M_PI);
	else if (std::is_same<float, Real>::value)
		_phase = (phase_angle - min_p * M_PI_F) / ((max_p - min_p) * M_PI_F);

	// Mapping to data type range
	return (T)(_phase * (max_T - min_T) + min_T);
}