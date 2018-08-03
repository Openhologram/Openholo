#include "ophCascadedPropagation.h"
#include "sys.h"
#include "tinyxml2.h"
#include <string>



ophCascadedPropagation::ophCascadedPropagation()
	: ready_to_propagate(false),
	hologram_path(L"")
{
}

ophCascadedPropagation::ophCascadedPropagation(const wchar_t* configfilepath)
	: ready_to_propagate(false),
	hologram_path(L"")
{
	if (readConfig(configfilepath) && allocateMem() && loadInput())
		ready_to_propagate = true;
}

ophCascadedPropagation::~ophCascadedPropagation()
{
}

void ophCascadedPropagation::ophFree()
{
	deallocateMem();
}

bool ophCascadedPropagation::propagate()
{
	if (!ready_to_propagate)
	{
		PRINT_ERROR("module not initialized");
		return false;
	}

	if (!propagateSlmToPupil())
	{
		PRINT_ERROR("failed to propagate to pupil plane");
		return false;
	}

	if (!propagatePupilToRetina())
	{
		PRINT_ERROR("failed to propagate to retina plane");
		return false;
	}

	return true;
}

bool ophCascadedPropagation::saveIntensityAsImg(const wchar_t* pathname, uint8_t bitsperpixel)
{
	wstring bufw(pathname);
	string bufs;
	bufs.assign(bufw.begin(), bufw.end());
	oph::uchar* src = getIntensityfields(getRetinaWavefieldAll());
	return (1 == saveAsImg(bufs.c_str(), bitsperpixel, src, getResX(), getResY()));
}


bool ophCascadedPropagation::allocateMem()
{
	wavefield_SLM.resize(getNumColors());
	wavefield_pupil.resize(getNumColors());
	wavefield_retina.resize(getNumColors());
	oph::uint nx = getResX();
	oph::uint ny = getResY();
	for (oph::uint i = 0; i < getNumColors(); i++)
	{
		wavefield_SLM[i] = new oph::Complex<Real>[nx * ny];
		wavefield_pupil[i] = new oph::Complex<Real>[nx * ny];
		wavefield_retina[i] = new oph::Complex<Real>[nx * ny];
	}

	return true;
}

void ophCascadedPropagation::deallocateMem()
{
	for each (auto e in wavefield_SLM)
		delete[] e;
	wavefield_SLM.clear();

	for each (auto e in wavefield_pupil)
		delete[] e;
	wavefield_pupil.clear();
	
	for each (auto e in wavefield_retina)
		delete[] e;
	wavefield_retina.clear();
}

// read in hologram data
bool ophCascadedPropagation::loadInput()
{
	string buf;
	buf.assign(hologram_path.begin(), hologram_path.end());
	if (checkExtension(buf.c_str(), ".bmp") == 0)
	{
		PRINT_ERROR("input file format not supported");
		return false;
	}
	oph::uint nx = getResX();
	oph::uint ny = getResY();
	oph::uchar* data = new oph::uchar[nx * ny * getNumColors()];
	if (loadAsImgUpSideDown(buf.c_str(), data) == 0)	// loadAsImg() keeps to fail
	{
		PRINT_ERROR("input file not found");
		delete[] data;
		return false;
	}

	// copy data to wavefield
	try {
		oph::uint numColors = getNumColors();
		for (oph::uint row = 0; row < ny; row++)
		{
			for (oph::uint col = 0; col < nx; col++)
			{
				for (oph::uint color = 0; color < numColors; color++)
				{
					// BGR to RGB & upside-down
					wavefield_SLM[numColors - 1 - color][(ny - 1 - row) * nx+ col] = oph::Complex<Real>((Real)data[(row * nx + col) * numColors + color], 0);
				}
			}
		}
	}

	catch (...) {
		PRINT_ERROR("failed to generate wavefield from bmp");
		delete[] data;
		return false;
	}

	delete[] data;
	return true;
}

oph::uchar* ophCascadedPropagation::getIntensityfields(vector<oph::Complex<Real>*> waveFields)
{
	oph::uint numColors = getNumColors();
	if (numColors != 1 && numColors != 3)
	{
		PRINT_ERROR("invalid number of color channels");
		return nullptr;
	}

	oph::uint nx = getResX();
	oph::uint ny = getResY();
	oph::uchar* intensityField = new oph::uchar[nx * ny * numColors];
	for (oph::uint color = 0; color < numColors; color++)
	{
		Real* intensityFieldUnnormalized = new Real[nx * ny];

		// find minmax
		Real maxIntensity = 0.0;
		Real minIntensity = REAL_IS_DOUBLE ? DBL_MAX : FLT_MAX;
		for (oph::uint row = 0; row < ny; row++)
		{
			for (oph::uint col = 0; col < nx; col++)
			{
				intensityFieldUnnormalized[row * nx + col] = waveFields[color][row * nx + col].mag2();
				maxIntensity = max(maxIntensity, intensityFieldUnnormalized[row * nx + col]);
				minIntensity = min(minIntensity, intensityFieldUnnormalized[row * nx + col]);
			}
		}

		maxIntensity *= getNor();			// IS IT REALLY NEEDED?
		if (maxIntensity <= minIntensity)
		{
			for (oph::uint row = 0; row < ny; row++)
			{
				for (oph::uint col = 0; col < nx; col++)
				{
					intensityField[(row * nx + col) * numColors + (numColors - 1 - color)] = 0;	// flip RGB order
				}
			}
		}
		else
		{
			for (oph::uint row = 0; row < ny; row++)
			{
				for (oph::uint col = 0; col < nx; col++)
				{
					Real normalizedVal = (intensityFieldUnnormalized[row * nx + col] - minIntensity) / (maxIntensity - minIntensity);
					normalizedVal = min(1.0, normalizedVal);

					// rotate 180 & RGB flip
					intensityField[((ny - 1 - row) * nx + (nx - 1 - col)) * numColors + (numColors - 1 - color)] = (oph::uchar)(normalizedVal * 255);
				}
			}
		}
		delete[] intensityFieldUnnormalized;
	}

	return intensityField;
}

bool ophCascadedPropagation::readConfig(const wchar_t* fname)
{
	/*XML parsing*/
	tinyxml2::XMLDocument xml_doc;
	tinyxml2::XMLNode *xml_node;
	wstring fnamew(fname);
	string fnames;
	fnames.assign(fnamew.begin(), fnamew.end());

	if (checkExtension(fnames.c_str(), ".xml") == 0)
	{
		LOG("file's extension is not 'xml'\n");
		return false;
	}
	auto ret = xml_doc.LoadFile(fnames.c_str());
	if (ret != tinyxml2::XML_SUCCESS)
	{
		LOG("Failed to load file \"%s\"\n", fnames.c_str());
		return false;
	}

	xml_node = xml_doc.FirstChild();
	auto next = xml_node->FirstChildElement("NumColors");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&config_.num_colors))
		return false;
	if (getNumColors() == 0 || getNumColors() > 3)
		return false;
	
	if (config_.num_colors >= 1)
	{
		next = xml_node->FirstChildElement("WavelengthR");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.wavelengths[0]))
			return false;
	}
	if (config_.num_colors >= 2)
	{
		next = xml_node->FirstChildElement("WavelengthG");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.wavelengths[1]))
			return false;
	}
	if (config_.num_colors == 3)
	{
		next = xml_node->FirstChildElement("WavelengthB");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.wavelengths[2]))
			return false;
	}

	next = xml_node->FirstChildElement("PixelPitchHor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.dx))
		return false;
	next = xml_node->FirstChildElement("PixelPitchVer");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.dy))
		return false;
	if (getPixelPitchX() != getPixelPitchY())
	{
		PRINT_ERROR("current implementation assumes pixel pitches are same for X and Y axes");
		return false;
	}

	next = xml_node->FirstChildElement("ResolutionHor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&config_.nx))
		return false;
	next = xml_node->FirstChildElement("ResolutionVer");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&config_.ny))
		return false;

	next = xml_node->FirstChildElement("FieldLensFocalLength");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.field_lens_focal_length))
		return false;
	next = xml_node->FirstChildElement("DistReconstructionPlaneToPupil");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.dist_reconstruction_plane_to_pupil))
		return false;
	next = xml_node->FirstChildElement("DistPupilToRetina");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.dist_pupil_to_retina))
		return false;
	next = xml_node->FirstChildElement("PupilDiameter");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.pupil_diameter))
		return false;
	next = xml_node->FirstChildElement("Nor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&config_.nor))
		return false;

	next = xml_node->FirstChildElement("HologramPath");
	if (!next || !(next->GetText()))
		return false;
	string holopaths = (xml_node->FirstChildElement("HologramPath"))->GetText();
	hologram_path.assign(holopaths.begin(), holopaths.end());

	return true;
}

bool ophCascadedPropagation::propagateSlmToPupil()
{
	oph::uint numColors = getNumColors();
	oph::uint nx = getResX();
	oph::uint ny = getResY();
	oph::Complex<Real>* buf = new oph::Complex<Real>[nx * ny];
	for (oph::uint color = 0; color < numColors; color++)
	{
		fftwShift(getSlmWavefield(color), buf, nx, ny, OPH_FORWARD, false);

		Real k = 2 * M_PI / getWavelengths()[color];
		Real vw = getWavelengths()[color] * getFieldLensFocalLength() / getPixelPitchX();
		Real dx1 = vw / (Real)nx;
		Real dy1 = vw / (Real)ny;
		for (oph::uint row = 0; row < ny; row++)
		{
			Real Y1 = ((Real)row - ((Real)ny - 1) * 0.5f) * dy1;
			for (oph::uint col = 0; col < nx; col++)
			{
				Real X1 = ((Real)col - ((Real)nx - 1) * 0.5f) * dx1;

				// 1st propagation
				oph::Complex<Real> t1 = oph::Complex<Real>(0, k / 2 / getFieldLensFocalLength() * (X1 * X1 + Y1 * Y1)).exp();
				oph::Complex<Real> t2(0, getWavelengths()[color] * getFieldLensFocalLength());
				buf[row * nx + col] = t1 / t2 * buf[row * nx + col];

				// applying aperture: need some optimization later
				if ((sqrt(X1 * X1 + Y1 * Y1) >= getPupilDiameter() / 2) || (row >= ny / 2 - 1))
					buf[row * nx + col] = 0;

				Real f_eye = (getFieldLensFocalLength() - getDistObjectToPupil()) * getDistPupilToRetina() / (getFieldLensFocalLength() - getDistObjectToPupil() + getDistPupilToRetina());
				oph::Complex<Real> t3 = oph::Complex<Real>(0, -k / 2 / f_eye * (X1 * X1 + Y1 * Y1)).exp();
				buf[row * nx + col] *= t3;
			}
		}

		memcpy(getPupilWavefield(color), buf, sizeof(oph::Complex<Real>) * nx * ny);
	}
	delete[] buf;
	return true;
}

bool ophCascadedPropagation::propagatePupilToRetina()
{
	oph::uint numColors = getNumColors();
	oph::uint nx = getResX();
	oph::uint ny = getResY();
	oph::Complex<Real>* buf = new oph::Complex<Real>[nx * ny];
	for (oph::uint color = 0; color < numColors; color++)
	{
		memcpy(buf, getPupilWavefield(color), sizeof(oph::Complex<Real>) * nx * ny);

		Real k = 2 * M_PI / getWavelengths()[color];
		Real vw = getWavelengths()[color] * getFieldLensFocalLength() / getPixelPitchX();
		Real dx1 = vw / (Real)nx;
		Real dy1 = vw / (Real)ny;
		for (oph::uint row = 0; row < ny; row++)
		{
			Real Y1 = ((Real)row - ((Real)ny - 1) * 0.5f) * dy1;
			for (oph::uint col = 0; col < nx; col++)
			{
				Real X1 = ((Real)col - ((Real)nx - 1) * 0.5f) * dx1;

				// 2nd propagation
				oph::Complex<Real> t1 = oph::Complex<Real>(0, k / 2 / getDistPupilToRetina() * (X1 * X1 + Y1 * Y1)).exp();
				buf[row * nx + col] *= t1;
			}
		}

		fftwShift(buf, getRetinaWavefield(color), nx, ny, OPH_FORWARD, false);
	}

	delete[] buf;
	return true;
}

oph::Complex<Real>* ophCascadedPropagation::getSlmWavefield(oph::uint id)
{
	if (wavefield_SLM.size() <= (size_t)id)
		return nullptr;
	return wavefield_SLM[id];
}

oph::Complex<Real>* ophCascadedPropagation::getPupilWavefield(oph::uint id)
{
	if (wavefield_pupil.size() <= (size_t)id)
		return nullptr;
	return wavefield_pupil[id];
}

oph::Complex<Real>* ophCascadedPropagation::getRetinaWavefield(oph::uint id)
{
	if (wavefield_retina.size() <= (size_t)id)
		return nullptr;
	return wavefield_retina[id];
}

vector<oph::Complex<Real>*> ophCascadedPropagation::getRetinaWavefieldAll()
{
	return wavefield_retina;
}


