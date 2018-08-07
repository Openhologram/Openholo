#pragma once
#ifndef _OphCascadedPropagation_h
#define _OphCascadedPropagation_h

#include "ophDis.h"

struct OphCascadedPropagationConfig {
	OphCascadedPropagationConfig()
		: num_colors(0),
		wavelengths{ 0.0, 0.0, 0.0 },
		dx(0.0),
		dy(0.0),
		nx(0),
		ny(0),
		field_lens_focal_length(0.0),
		dist_reconstruction_plane_to_pupil(0.0),
		dist_pupil_to_retina(0.0),
		pupil_diameter(0.0),
		nor(0.0)
		{}

	oph::uint num_colors; // number of colors
	oph::vec3 wavelengths; // wavelength list. if m_NumColors == 1, only m_Wavelengths[0] is used
	Real dx; // horizontal pixel pitch
	Real dy; // vertical pixel pitch
	oph::uint nx; // horizontal resolution
	oph::uint ny; // vertical resolution
	Real field_lens_focal_length; // distance from SLM plane to pupil plane
	Real dist_reconstruction_plane_to_pupil; // distance from object plane to pupil plane
	Real dist_pupil_to_retina; // distance from pupil plane to retina plane
	Real pupil_diameter; // diameter of pupil
	Real nor; // scaling term in Chang Eun Young's implementation
};


#ifdef DISP_EXPORT
#define DISP_DLL __declspec(dllexport)
#else
#define DISP_DLL __declspec(dllimport)
#endif

class DISP_DLL ophCascadedPropagation : public ophDis {
	public:
		ophCascadedPropagation();
		ophCascadedPropagation(const wchar_t* configfilepath);
		~ophCascadedPropagation();

		bool propagate();
		bool saveIntensityAsImg(const wchar_t* pathname, uint8_t bitsperpixel);


	private:
		OphCascadedPropagationConfig config_;
		vector<oph::Complex<Real>*> wavefield_SLM; // wavefield at SLM plane
		vector<oph::Complex<Real>*> wavefield_pupil; // wavefield at pupil plane
		vector<oph::Complex<Real>*> wavefield_retina; // wavefield at retina plane
		bool ready_to_propagate;
		wstring hologram_path;

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
		oph::uint getNumColors() { return config_.num_colors; }
		oph::vec3 getWavelengths() { return config_.wavelengths; }
		Real getPixelPitchX() { return config_.dx; }
		Real getPixelPitchY() { return config_.dy; }
		oph::uint getResX() { return config_.nx; }
		oph::uint getResY() { return config_.ny; }
		Real getFieldLensFocalLength() { return config_.field_lens_focal_length; }
		Real getDistObjectToPupil() { return config_.dist_reconstruction_plane_to_pupil; }
		Real getDistPupilToRetina() { return config_.dist_pupil_to_retina; }
		Real getPupilRadius() { return config_.pupil_diameter; }
		Real getNor() { return config_.nor; }

		oph::Complex<Real>* getSlmWavefield(oph::uint id);
		oph::Complex<Real>* getPupilWavefield(oph::uint id);
		oph::Complex<Real>* getRetinaWavefield(oph::uint id);
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