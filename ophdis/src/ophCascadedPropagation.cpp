#include "ophCascadedPropagation.h"
#include "sys.h"
#include "tinyxml2.h"
#include <string>



OphCascadedPropagation::OphCascadedPropagation()
	: m_ReadyToPropagate(false),
	m_HologramPath(L"")
{
}

OphCascadedPropagation::OphCascadedPropagation(const wchar_t* configfilepath)
	: m_ReadyToPropagate(false),
	m_HologramPath(L"")
{
	if (readConfig(configfilepath) && allocateMem() && loadInput())
		m_ReadyToPropagate = true;
}

OphCascadedPropagation::~OphCascadedPropagation()
{
}

void OphCascadedPropagation::ophFree()
{
	deallocateMem();
}

bool OphCascadedPropagation::propagate()
{
	if (!m_ReadyToPropagate)
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

bool OphCascadedPropagation::saveIntensityAsImg(const wchar_t* pathname, uint8_t bitsperpixel)
{
	wstring bufw(pathname);
	string bufs;
	bufs.assign(bufw.begin(), bufw.end());
	oph::uchar* src = getIntensityfield(getRetinaWavefieldAll());
	return (1 == saveAsImg(bufs.c_str(), bitsperpixel, src, GetResX(), GetResY()));
}


bool OphCascadedPropagation::allocateMem()
{
	m_WFSlm.resize(GetNumColors());
	m_WFPupil.resize(GetNumColors());
	m_WFRetina.resize(GetNumColors());
	oph::uint nx = GetResX();
	oph::uint ny = GetResY();
	for (uint i = 0; i < GetNumColors(); i++)
	{
		m_WFSlm[i] = new oph::Complex<Real>[nx * ny];
		m_WFPupil[i] = new oph::Complex<Real>[nx * ny];
		m_WFRetina[i] = new oph::Complex<Real>[nx * ny];
	}

	return true;
}

void OphCascadedPropagation::deallocateMem()
{
	for each (auto e in m_WFSlm)
		delete[] e;
	m_WFSlm.clear();

	for each (auto e in m_WFPupil)
		delete[] e;
	m_WFPupil.clear();
	
	for each (auto e in m_WFRetina)
		delete[] e;
	m_WFRetina.clear();
}

// read in hologram data
bool OphCascadedPropagation::loadInput()
{
	string buf;
	buf.assign(m_HologramPath.begin(), m_HologramPath.end());
	if (checkExtension(buf.c_str(), ".bmp") == 0)
	{
		PRINT_ERROR("input file format not supported");
		return false;
	}
	oph::uint nx = GetResX();
	oph::uint ny = GetResY();
	oph::uchar* data = new oph::uchar[nx * ny * GetNumColors()];
	if (loadAsImgUpSideDown(buf.c_str(), data) == 0)	// loadAsImg() keeps to fail
	{
		PRINT_ERROR("input file not found");
		delete[] data;
		return false;
	}

	// copy data to wavefield
	try {
		uint numColors = GetNumColors();
		for (uint row = 0; row < ny; row++)
		{
			for (uint col = 0; col < nx; col++)
			{
				for (uint color = 0; color < numColors; color++)
				{
					// BGR to RGB & upside-down
					m_WFSlm[numColors - 1 - color][(ny - 1 - row) * nx+ col] = oph::Complex<Real>((Real)data[(row * nx + col) * numColors + color], 0);
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

oph::uchar* OphCascadedPropagation::getIntensityfield(vector<oph::Complex<Real>*> waveFields)
{
	uint numColors = GetNumColors();
	if (numColors != 1 && numColors != 3)
	{
		PRINT_ERROR("invalid number of color channels");
		return nullptr;
	}

	oph::uint nx = GetResX();
	oph::uint ny = GetResY();
	oph::uchar* intensityField = new oph::uchar[nx * ny * numColors];
	for (uint color = 0; color < numColors; color++)
	{
		Real* intensityFieldUnnormalized = new Real[nx * ny];

		// find minmax
		Real maxIntensity = 0.0;
		Real minIntensity = REAL_IS_DOUBLE ? DBL_MAX : FLT_MAX;
		for (uint row = 0; row < ny; row++)
		{
			for (uint col = 0; col < nx; col++)
			{
				intensityFieldUnnormalized[row * nx + col] = waveFields[color][row * nx + col].mag2();
				maxIntensity = max(maxIntensity, intensityFieldUnnormalized[row * nx + col]);
				minIntensity = min(minIntensity, intensityFieldUnnormalized[row * nx + col]);
			}
		}

		maxIntensity *= GetNor();			// IS IT REALLY NEEDED?
		if (maxIntensity <= minIntensity)
		{
			for (uint row = 0; row < ny; row++)
			{
				for (uint col = 0; col < nx; col++)
				{
					intensityField[(row * nx + col) * numColors + (numColors - 1 - color)] = 0;	// flip RGB order
				}
			}
		}
		else
		{
			for (uint row = 0; row < ny; row++)
			{
				for (uint col = 0; col < nx; col++)
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

bool OphCascadedPropagation::readConfig(const wchar_t* fname)
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
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&m_config.m_NumColors))
		return false;
	if (GetNumColors() == 0 || GetNumColors() > 3)
		return false;
	
	if (m_config.m_NumColors >= 1)
	{
		next = xml_node->FirstChildElement("WavelengthR");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_Wavelengths[0]))
			return false;
	}
	if (m_config.m_NumColors >= 2)
	{
		next = xml_node->FirstChildElement("WavelengthG");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_Wavelengths[1]))
			return false;
	}
	if (m_config.m_NumColors == 3)
	{
		next = xml_node->FirstChildElement("WavelengthB");
		if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_Wavelengths[2]))
			return false;
	}

	next = xml_node->FirstChildElement("PixelPitchHor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_dx))
		return false;
	next = xml_node->FirstChildElement("PixelPitchVer");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_dy))
		return false;
	if (GetPixelPitchX() != GetPixelPitchY())
	{
		PRINT_ERROR("current implementation assumes pixel pitches are same for X and Y axes");
		return false;
	}

	next = xml_node->FirstChildElement("ResolutionHor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&m_config.m_nx))
		return false;
	next = xml_node->FirstChildElement("ResolutionVer");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryUnsignedText(&m_config.m_ny))
		return false;

	next = xml_node->FirstChildElement("FieldLensFocalLength");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_FieldLensFocalLength))
		return false;
	next = xml_node->FirstChildElement("DistReconstructionPlaneToPupil");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_DistReconstructionPlaneToPupil))
		return false;
	next = xml_node->FirstChildElement("DistPupilToRetina");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_DistPupilToRetina))
		return false;
	next = xml_node->FirstChildElement("PupilDiameter");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_PupilDiameter))
		return false;
	next = xml_node->FirstChildElement("Nor");
	if (!next || tinyxml2::XML_SUCCESS != next->QueryDoubleText(&m_config.m_Nor))
		return false;

	next = xml_node->FirstChildElement("HologramPath");
	if (!next || !(next->GetText()))
		return false;
	string holopaths = (xml_node->FirstChildElement("HologramPath"))->GetText();
	m_HologramPath.assign(holopaths.begin(), holopaths.end());

	return true;
}

bool OphCascadedPropagation::propagateSlmToPupil()
{
	uint numColors = GetNumColors();
	uint nx = GetResX();
	uint ny = GetResY();
	oph::Complex<Real>* buf = new oph::Complex<Real>[nx * ny];
	for (uint color = 0; color < numColors; color++)
	{
		fftwShift(getSlmWavefield(color), buf, nx, ny, OPH_FORWARD, false);

		Real k = 2 * M_PI / GetWavelengths()[color];
		Real vw = GetWavelengths()[color] * GetFieldLensFocalLength() / GetPixelPitchX();
		Real dx1 = vw / (Real)nx;
		Real dy1 = vw / (Real)ny;
		for (uint row = 0; row < ny; row++)
		{
			Real Y1 = ((Real)row - ((Real)ny - 1) * 0.5f) * dy1;
			for (uint col = 0; col < nx; col++)
			{
				Real X1 = ((Real)col - ((Real)nx - 1) * 0.5f) * dx1;

				// 1st propagation
				oph::Complex<Real> t1 = oph::Complex<Real>(0, k / 2 / GetFieldLensFocalLength() * (X1 * X1 + Y1 * Y1)).exp();
				oph::Complex<Real> t2(0, GetWavelengths()[color] * GetFieldLensFocalLength());
				buf[row * nx + col] = t1 / t2 * buf[row * nx + col];

				// applying aperture: need some optimization later
				if ((sqrt(X1 * X1 + Y1 * Y1) >= GetPupilRadius() / 2) || (row >= ny / 2 - 1))
					buf[row * nx + col] = 0;

				Real f_eye = (GetFieldLensFocalLength() - GetDistObjectToPupil()) * GetDistPupilToRetina() / (GetFieldLensFocalLength() - GetDistObjectToPupil() + GetDistPupilToRetina());
				oph::Complex<Real> t3 = oph::Complex<Real>(0, -k / 2 / f_eye * (X1 * X1 + Y1 * Y1)).exp();
				buf[row * nx + col] *= t3;
			}
		}

		memcpy(getPupilWavefield(color), buf, sizeof(oph::Complex<Real>) * nx * ny);
	}
	delete[] buf;
	return true;
}

bool OphCascadedPropagation::propagatePupilToRetina()
{
	uint numColors = GetNumColors();
	uint nx = GetResX();
	uint ny = GetResY();
	oph::Complex<Real>* buf = new oph::Complex<Real>[nx * ny];
	for (uint color = 0; color < numColors; color++)
	{
		memcpy(buf, getPupilWavefield(color), sizeof(oph::Complex<Real>) * nx * ny);

		Real k = 2 * M_PI / GetWavelengths()[color];
		Real vw = GetWavelengths()[color] * GetFieldLensFocalLength() / GetPixelPitchX();
		Real dx1 = vw / (Real)nx;
		Real dy1 = vw / (Real)ny;
		for (uint row = 0; row < ny; row++)
		{
			Real Y1 = ((Real)row - ((Real)ny - 1) * 0.5f) * dy1;
			for (uint col = 0; col < nx; col++)
			{
				Real X1 = ((Real)col - ((Real)nx - 1) * 0.5f) * dx1;

				// 2nd propagation
				oph::Complex<Real> t1 = oph::Complex<Real>(0, k / 2 / GetDistPupilToRetina() * (X1 * X1 + Y1 * Y1)).exp();
				buf[row * nx + col] *= t1;
			}
		}

		fftwShift(buf, getRetinaWavefield(color), nx, ny, OPH_FORWARD, false);
	}

	delete[] buf;
	return true;
}

oph::Complex<Real>* OphCascadedPropagation::getSlmWavefield(uint id)
{
	if (m_WFSlm.size() <= (size_t)id)
		return nullptr;
	return m_WFSlm[id];
}

oph::Complex<Real>* OphCascadedPropagation::getPupilWavefield(uint id)
{
	if (m_WFPupil.size() <= (size_t)id)
		return nullptr;
	return m_WFPupil[id];
}

oph::Complex<Real>* OphCascadedPropagation::getRetinaWavefield(uint id)
{
	if (m_WFRetina.size() <= (size_t)id)
		return nullptr;
	return m_WFRetina[id];
}

vector<oph::Complex<Real>*> OphCascadedPropagation::getRetinaWavefieldAll()
{
	return m_WFRetina;
}


