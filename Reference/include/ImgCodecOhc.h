#ifndef __ImgCodecOhc_h
#define __ImgCodecOhc_h

#include <fstream>
#include "include.h"
#include "mat.h"
#include "vec.h"

#include "ImgCodecDefine.h"

#ifdef OPH_EXPORT
#define OPH_DLL __declspec(dllexport)
#else
#define OPH_DLL __declspec(dllimport)
#endif

namespace oph
{
	class OPH_DLL ImgCodecOhc {
	public: /* method */
		ImgCodecOhc();
		ImgCodecOhc(const std::string &_fname);
		ImgCodecOhc(const std::string &_fname, const OHCheader &_Header);
		virtual ~ImgCodecOhc() = 0;

		bool setFileName(const std::string &_fname);
		bool setOHCheader(const OHCheader &_Header);

		void getOHCheader(OHCheader &_Header);
		void getFieldInfo(OHCFIELDINFOHEADER &_FieldInfo, std::vector<double_t> &_wavlenTable);

		void releaseOHCheader();
		void releaseCodeBuffer();
		void releaseFldData();

	protected: /* field */
		std::string fname;
		void* buf = nullptr; //coded data
		std::vector<OphComplexField> field_cmplx; //Real & Imagine data
		std::vector<std::string> linkFilePath;

		OHCheader* Header = nullptr;
	};


	/* Load *.ohc file format to Complex field data */
	class OPH_DLL ImgDecoderOhc : public ImgCodecOhc {
	public:
		ImgDecoderOhc();
		ImgDecoderOhc(const std::string &_fname);
		ImgDecoderOhc(const std::string &_fname, const OHCheader &_Header);
		virtual ~ImgDecoderOhc();

		void releaseFldData(); //override

		bool load();

	protected:
		template<typename T> bool decodeFieldData();
		template<typename T> Real decodePhase(const T phase, const Real min_p, const Real max_p, const double min_T, const double max_T);

		//Only Amplitude Encoding or Only Phase Encoding or Amplitude & Phase data
		std::vector<OphRealField> field_ampli;
		std::vector<OphRealField> field_phase;
		std::ifstream File;
	};


	/* Save Complex field data to *.ohc file format */
	class OPH_DLL ImgEncoderOhc : public ImgCodecOhc {
	public:
		ImgEncoderOhc();
		ImgEncoderOhc(const std::string &_fname);
		ImgEncoderOhc(const std::string &_fname, const OHCheader &_Header);
		virtual ~ImgEncoderOhc();

		//Set field Info parameters functions
		void setNumOfPixel(const uint _pxNumX, const uint _pxNumY);
		void setNumOfPixel(const ivec2 _pxNum);
		void setPixelPitch(const double _pxPitchX, const double _pxPitchY, const LenUnit unit = LenUnit::m);
		void setPixelPitch(const vec2 _pxPitch, const LenUnit unit = LenUnit::m);
		void setNumOfWavlen(const uint n_wavlens);
		void setColorType(const ColorType _clrType);
		void setColorArrange(const ColorArran _clrArrange);
		void setUnitOfWavlen(const LenUnit unit);
		void setFieldEncoding(const FldStore _fldStore, const FldCodeType _fldCodeType, const DataType _cmplxFldType = DataType::Float64);
		void setPhaseEncoding(const BPhaseCode _bPhaseCode, const double _phaseCodeMin, const double _phaseCodeMax);
		void setPhaseEncoding(const BPhaseCode _bPhaseCode, const vec2 _phaseCodeRange);
		void setImageFormat(const ImageFormat _imgFmt);
		void setWavelength(const Real _wavlen, const LenUnit _unit = LenUnit::nm);
		void push_back_WaveFld(const Real wavlen, const OphComplexField &data);
		void push_back_FldData(const OphComplexField &data);
		void push_back_Wavlen(const Real wavlen);
		void push_back_LinkFilePath(const std::string &path);
		void getLinkFilePath(const int idx, std::string &path);

		bool save();

	protected:
		template<typename T> uint64_t encodeFieldData();
		template<typename T> T encodePhase(const Real phase_angle, const Real min_p, const Real max_p, const double min_T, const double max_T);

		std::ofstream File;
	};
}

#endif // !__ImgCodecOhc_h