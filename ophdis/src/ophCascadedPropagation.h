#pragma once
#ifndef _OphCascadedPropagation_h
#define _OphCascadedPropagation_h

#include "ophDis.h"

class OphCascadedPropagationConfig {
	public:
		uint m_NumColors; // number of colors
		oph::vec3 m_Wavelengths; // wavelength list. if m_NumColors == 1, only m_Wavelengths[0] is used
		Real m_dx; // horizontal pixel pitch
		Real m_dy; // vertical pixel pitch
		uint m_nx; // horizontal resolution
		uint m_ny; // vertical resolution
		Real m_FieldLensFocalLength; // distance from SLM plane to pupil plane
		Real m_DistReconstructionPlaneToPupil; // distance from object plane to pupil plane
		Real m_DistPupilToRetina; // distance from pupil plane to retina plane
		Real m_PupilDiameter; // diameter of pupil
		Real m_Nor; // scaling term in Chang Eun Young's implementation

		OphCascadedPropagationConfig()
			: m_NumColors(0),
			m_Wavelengths{ 0.0, 0.0, 0.0 },
			m_dx(0.0),
			m_dy(0.0),
			m_nx(0),
			m_ny(0),
			m_FieldLensFocalLength(0.0),
			m_DistReconstructionPlaneToPupil(0.0),
			m_DistPupilToRetina(0.0),
			m_PupilDiameter(0.0),
			m_Nor(0.0)
			{}
};


#ifdef DISP_EXPORT
#define DISP_DLL __declspec(dllexport)
#else
#define DISP_DLL __declspec(dllimport)
#endif

class DISP_DLL OphCascadedPropagation : public ophDis {
	public:
		OphCascadedPropagation();
		OphCascadedPropagation(const wchar_t* configfilepath);
		~OphCascadedPropagation();

		bool propagate();
		bool saveIntensityAsImg(const wchar_t* pathname, uint8_t bitsperpixel);


	private:
		OphCascadedPropagationConfig m_config;
		vector<oph::Complex<Real>*> m_WFSlm; // wavefield at SLM plane
		vector<oph::Complex<Real>*> m_WFPupil; // wavefield at pupil plane
		vector<oph::Complex<Real>*> m_WFRetina; // wavefield at retina plane
		bool m_ReadyToPropagate;
		wstring m_HologramPath;

	private:
		bool readConfig(const wchar_t* fname);
		bool propagateSlmToPupil(); // 1st propagation: SLM to pupil
		bool propagatePupilToRetina(); // 2nd propagation: pupil to retina
		bool allocateMem();
		void deallocateMem();
		bool loadInput();
		oph::uchar* getIntensityfield(vector<oph::Complex<Real>*> waveFields);


	public:
		// getters
		uint GetNumColors() { return m_config.m_NumColors; }
		oph::vec3 GetWavelengths() { return m_config.m_Wavelengths; }
		Real GetPixelPitchX() { return m_config.m_dx; }
		Real GetPixelPitchY() { return m_config.m_dy; }
		uint GetResX() { return m_config.m_nx; }
		uint GetResY() { return m_config.m_ny; }
		Real GetFieldLensFocalLength() { return m_config.m_FieldLensFocalLength; }
		Real GetDistObjectToPupil() { return m_config.m_DistReconstructionPlaneToPupil; }
		Real GetDistPupilToRetina() { return m_config.m_DistPupilToRetina; }
		Real GetPupilRadius() { return m_config.m_PupilDiameter; }
		Real GetNor() { return m_config.m_Nor; }

		oph::Complex<Real>* getSlmWavefield(uint id);
		oph::Complex<Real>* getPupilWavefield(uint id);
		oph::Complex<Real>* getRetinaWavefield(uint id);
		vector<oph::Complex<Real>*> getRetinaWavefieldAll();


		// setters
		//virtual bool SetSlmWavefield(Complex<Real>* srcHologram) = 0; // set input wavefield (for later use)
		//virtual bool SetSlmWavefield(ophGen& srcHologram) = 0; // set input wavefield (for later use)


	protected:
		/**
		* @brief Pure virtual function for override in child classes
		*/
		virtual void ophFree(void);
};



// utilities
#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#define PRINT_ERROR(errorMsg)           { cout << "Error(" << __FILENAME__ << ":" << __LINE__ << "): " << ( errorMsg ) << endl; }

#endif