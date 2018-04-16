#ifndef __struct_h
#define __struct_h

namespace oph
{
	typedef struct OPH_DLL PointCloudData {
		PointCloudData(const float x, const float y, const float z)
		: x(x), y(y), z(z), amplitude(0.f), phase(0.f) {}
		PointCloudData(const float x, const float y, const float z, const float amplitude, const float phase)
		: x(x), y(y), z(z), amplitude(amplitude), phase(phase) {}

		float x;
		float y;
		float z;
		float amplitude;
		float phase;
	}OphPointCloudData;

	typedef struct OPH_DLL ConfigParams {
		ConfigParams(void) 
		: pointCloudScaleX(0), pointCloudScaleY(0), pointCloudScaleZ(0), offsetDepth(0), samplingPitchX(0), samplingPitchY(0)
		, nx(0), ny(0), filterShapeFlag(0), filterXwidth(0), filterYwidth(0), focalLengthLensIn(0), focalLengthLensOut(0)
		, focalLengthLensEyePiece(0), lambda(0), tiltAngleX(0), tiltAngleY(0) {}

		ConfigParams(const std::string InputConfigFile)
		: pointCloudScaleX(0), pointCloudScaleY(0), pointCloudScaleZ(0), offsetDepth(0), samplingPitchX(0), samplingPitchY(0)
		, nx(0), ny(0), filterShapeFlag(0), filterXwidth(0), filterYwidth(0), focalLengthLensIn(0), focalLengthLensOut(0)
		, focalLengthLensEyePiece(0), lambda(0), tiltAngleX(0), tiltAngleY(0) {
			std::ifstream File(InputConfigFile, std::ios::in);
			if (!File.is_open()) {
				std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
				File.close();
				return;
			}

			std::vector<std::string> Title;
			std::vector<std::string> Value;
			std::string Line;
			std::stringstream LineStream;

			int i = 0;
			while (std::getline(File, Line)) {
				std::string _Title;
				std::string _Value;
				std::string _Equal; // " = "
				LineStream << Line;
				LineStream >> _Title >> _Equal >> _Value;
				LineStream.clear();

				Title.push_back(_Title);
				Value.push_back(_Value);
				++i;
			}

			if (i != 17) {
				std::cerr << "OpenHolo Error : Failed to load Config Specification Data File(*.config)" << std::endl;
				File.close();
				return;
			}

			this->pointCloudScaleX = static_cast<float>(atof(Value[0].c_str()));
			this->pointCloudScaleY = static_cast<float>(atof(Value[1].c_str()));
			this->pointCloudScaleZ = static_cast<float>(atof(Value[2].c_str()));
			this->offsetDepth = static_cast<float>(atof(Value[3].c_str()));
			this->samplingPitchX = static_cast<float>(atof(Value[4].c_str()));
			this->samplingPitchY = static_cast<float>(atof(Value[5].c_str()));
			this->nx = atoi(Value[6].c_str());
			this->ny = atoi(Value[7].c_str());
			this->filterShapeFlag = (char*)Value[8].c_str();
			this->filterXwidth = static_cast<float>(atof(Value[9].c_str()));
			this->filterYwidth = static_cast<float>(atof(Value[10].c_str()));
			this->focalLengthLensIn = static_cast<float>(atof(Value[11].c_str()));
			this->focalLengthLensOut = static_cast<float>(atof(Value[12].c_str()));
			this->focalLengthLensEyePiece = static_cast<float>(atof(Value[13].c_str()));
			this->lambda = static_cast<float>(atof(Value[14].c_str()));
			this->tiltAngleX = static_cast<float>(atof(Value[15].c_str()));
			this->tiltAngleY = static_cast<float>(atof(Value[16].c_str()));
			File.close();
		}

		float pointCloudScaleX;	/// Scaling factor of x coordinate of point cloud
		float pointCloudScaleY;	/// Scaling factor of y coordinate of point cloud
		float pointCloudScaleZ;	/// Scaling factor of z coordinate of point cloud

		float offsetDepth;		/// Offset value of point cloud in z direction

		float samplingPitchX;	/// Pixel pitch of SLM in x direction
		float samplingPitchY;	/// Pixel pitch of SLM in y direction

		int nx;	/// Number of pixel of SLM in x direction
		int ny;	/// Number of pixel of SLM in y direction

		char *filterShapeFlag;	/// Shape of spatial bandpass filter ("Circle" or "Rect" for now)
		float filterXwidth;		/// Width of spatial bandpass filter in x direction (For "Circle," only this is used)
		float filterYwidth;		/// Width of spatial bandpass filter in y direction

		float focalLengthLensIn;		/// Focal length of input lens of Telecentric
		float focalLengthLensOut;		/// Focal length of output lens of Telecentric
		float focalLengthLensEyePiece;	/// Focal length of eyepiece lens				

		float lambda;		/// Wavelength of laser

		float tiltAngleX;	/// Tilt angle in x direction for spatial filtering
		float tiltAngleY;	/// Tilt angle in y direction for spatial filtering
	} OphConfigParams;

	// for PointCloud
	typedef struct OPH_DLL KernelConst {
		int n_points;	///number of point cloud

		float scaleX;		/// Scaling factor of x coordinate of point cloud
		float scaleY;		/// Scaling factor of y coordinate of point cloud
		float scaleZ;		/// Scaling factor of z coordinate of point cloud

		float offsetDepth;	/// Offset value of point cloud in z direction

		int Nx;		/// Number of pixel of SLM in x direction
		int Ny;		/// Number of pixel of SLM in y direction

		float sin_thetaX; ///sin(tiltAngleX)
		float sin_thetaY; ///sin(tiltAngleY)
		float k;		  ///Wave Number = (2 * PI) / lambda;

		float pixel_x; /// Pixel pitch of SLM in x direction
		float pixel_y; /// Pixel pitch of SLM in y direction
		float halfLength_x; /// (pixel_x * nx) / 2
		float halfLength_y; /// (pixel_y * ny) / 2
	} GpuConst;
}

#define WIDTHBYTES(bits) (((bits)+31)/32*4)

//#define _bitsperpixel 8 //24 // 3바이트=24 
#define _planes 1
#define _compression 0
#define _xpixelpermeter 0x130B //2835 , 72 DPI
#define _ypixelpermeter 0x130B //2835 , 72 DPI
//#define pixel 0xFF

#pragma pack(push,1)
typedef struct OPH_DLL {
	uint8_t signature[2];
	uint32_t filesize;
	uint32_t reserved;
	uint32_t fileoffset_to_pixelarray;
} fileheader;
typedef struct OPH_DLL {
	uint32_t dibheadersize;
	uint32_t width;
	uint32_t height;
	uint16_t planes;
	uint16_t bitsperpixel;
	uint32_t compression;
	uint32_t imagesize;
	uint32_t ypixelpermeter;
	uint32_t xpixelpermeter;
	uint32_t numcolorspallette;
	uint32_t mostimpcolor;
} bitmapinfoheader;
// 24비트 미만 칼라는 팔레트가 필요함
typedef struct OPH_DLL {
	uint8_t rgbBlue;
	uint8_t rgbGreen;
	uint8_t rgbRed;
	uint8_t rgbReserved;
} rgbquad;
typedef struct {
	fileheader fileheader;
	bitmapinfoheader bitmapinfoheader;
	rgbquad rgbquad[256]; // 8비트 256칼라(흑백)
} bitmap;
#pragma pack(pop)

#endif // !__struct_h