#include "ophSigPU.h"

ophSigPU::ophSigPU(void)
{
}

bool ophSigPU::setPUparam(int maxBoxRadius)
{
	MaxBoxRadius = maxBoxRadius;
	return false;
}

bool ophSigPU::loadPhaseOriginal(const char * fname, int bitpixel)
{
	string fnamestr = fname;
	int checktype = static_cast<int>(fnamestr.rfind("."));
	matrix<Real> phaseMat;

	std::string filetype = fnamestr.substr(checktype + 1, fnamestr.size());

	if (filetype == "bmp")
	{
		FILE *fphase;
		fileheader hf;
		bitmapinfoheader hInfo;
		fopen_s(&fphase, fnamestr.c_str(), "rb"); 
		if (!fphase)
		{
			LOG("real bmp file open fail!\n");
			return false;
		}
		fread(&hf, sizeof(fileheader), 1, fphase);
		fread(&hInfo, sizeof(bitmapinfoheader), 1, fphase);

		if (hf.signature[0] != 'B' || hf.signature[1] != 'M') { LOG("Not BMP File!\n"); }
		if ((hInfo.height == 0) || (hInfo.width == 0))
		{
			LOG("bmp header is empty!\n");
			hInfo.height = _cfgSig.rows;
			hInfo.width = _cfgSig.cols;
			if (_cfgSig.rows == 0 || _cfgSig.cols == 0)
			{
				LOG("check your parameter file!\n");
				return false;
			}
		}
		if ((_cfgSig.rows != hInfo.height) || (_cfgSig.cols != hInfo.width)) {
			LOG("image size is different!\n");
			_cfgSig.rows = hInfo.height;
			_cfgSig.cols = hInfo.width;
			LOG("changed parameter of size %d x %d\n", _cfgSig.cols, _cfgSig.rows);
		}
		hInfo.bitsperpixel = bitpixel;
		if (bitpixel == 8)
		{
			rgbquad palette[256];
			fread(palette, sizeof(rgbquad), 256, fphase);

			phaseMat.resize(hInfo.height, hInfo.width);
		}
		else
		{
			LOG("currently only 8 bitpixel is supported.");
			/*
			phaseMat[0].resize(hInfo.height, hInfo.width);
			phaseMat[1].resize(hInfo.height, hInfo.width);
			phaseMat[2].resize(hInfo.height, hInfo.width); */
		}

		uchar* phasedata = (uchar*)malloc(sizeof(uchar)*hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8));

		fread(phasedata, sizeof(uchar), hInfo.width*hInfo.height*(hInfo.bitsperpixel / 8), fphase);

		fclose(fphase);

		for (int i = hInfo.height - 1; i >= 0; i--)
		{
			for (int j = 0; j < static_cast<int>(hInfo.width); j++)
			{
				for (int z = 0; z < (hInfo.bitsperpixel / 8); z++)
				{
					phaseMat(hInfo.height - i - 1, j) = (double)phasedata[i*hInfo.width*(hInfo.bitsperpixel / 8) + (hInfo.bitsperpixel / 8)*j + z];
				}
			}
		}
		LOG("file load complete!\n");

		free(phasedata);
	}
	else if (filetype == "bin")
	{
		if (bitpixel == 8)
		{

			ifstream fphase(fnamestr, ifstream::binary);
			phaseMat.resize(_cfgSig.rows, _cfgSig.cols); 
			int total = _cfgSig.rows*_cfgSig.cols;
			double *phasedata = new double[total];
			int i = 0;
			fphase.read(reinterpret_cast<char*>(phasedata), sizeof(double) * total);

			for (int col = 0; col < _cfgSig.cols; col++)
			{
				for (int row = 0; row < _cfgSig.rows; row++)
				{
					phaseMat(row, col) = phasedata[_cfgSig.rows*col + row];
				}
			}

			fphase.close();
			delete[]phasedata;
		}
		else if (bitpixel == 24)
		{
			LOG("currently only 8 bitpixel is supported.");
			/*
			phaseMat[0].resize(_cfgSig.rows, _cfgSig.cols);
			phaseMat[1].resize(_cfgSig.rows, _cfgSig.cols);
			phaseMat[2].resize(_cfgSig.rows, _cfgSig.cols);

			int total = _cfgSig.rows*_cfgSig.cols;


			string RGB_name[] = { "_B","_G","_R" };
			double *phasedata = new  double[total];
			char *context = nullptr;
			for (int z = 0; z < (bitpixel / 8); z++)
			{
				ifstream fphase(strtok_s((char*)fnamestr.c_str(), ".", &context) + RGB_name[z] + "bin", ifstream::binary);

				fphase.read(reinterpret_cast<char*>(phasedata), sizeof(double) * total);

				for (int col = 0; col < _cfgSig.cols; col++)
				{
					for (int row = 0; row < _cfgSig.rows; row++)
					{
						phaseMat[z](row, col) = phasedata[_cfgSig.rows*col + row];
					}
				}
				fphase.close();
			}
			delete[] phasedata; */
		}
	}
	else
	{
		LOG("wrong type\n");
	}

	//////////////////////////////////////////////////////
	//////// From here, modified by Jae-Hyeung Park from original load function in ophSig 
	//nomalization 
	Nr = _cfgSig.rows;
	Nc = _cfgSig.cols;
	PhaseOriginal.resize(Nr, Nc);
	PhaseUnwrapped.resize(Nr, Nc);
	for (int i = 0; i < _cfgSig.rows; i++)
	{
		for (int j = 0; j < _cfgSig.cols; j++)
		{
			PhaseOriginal(i, j) = phaseMat(i, j) / 255.0*M_PI*2 - M_PI;
			PhaseUnwrapped(i, j) = 0;
		}
	}

	LOG("data nomalization complete\n");

	return true;
}

bool ophSigPU::loadPhaseOriginal(void)
{
	return false;
}

bool ophSigPU::runPU(void)
{
	auto start_time = CUR_TIME;

	matrix<Real> residue(Nr,Nc);
	residue.zeros();
	matrix<Real> branchcut(Nr, Nc);
	branchcut.zeros();
	phaseResidues(residue);
	branchCuts(residue, branchcut);
	floodFill(branchcut);

	auto end_time = CUR_TIME;

	auto during_time = ((std::chrono::duration<Real>)(end_time - start_time)).count();

	LOG("Implement time : %.5lf sec\n", during_time);

	return false;
}

bool ophSigPU::savePhaseUnwrapped(const char * fname)
{
	oph::uchar* phaseData;
	phaseData = (oph::uchar*)malloc(sizeof(oph::uchar) * Nr * Nc);

	string fnamestr = fname;

	double maxPhase = 0;
	double minPhase = 0;
	for (int i = 0; i < Nr; i++)
	{
		for (int j = 0; j < Nc; j++)
		{
			if (PhaseUnwrapped(i, j) > maxPhase)
			{
				maxPhase = PhaseUnwrapped(i, j);
			}
			if (PhaseUnwrapped(i, j) < minPhase)
			{
				minPhase = PhaseUnwrapped(i, j);
			}
		}
	}


	for (int i = 0; i < Nr; i++)
	{
		for (int j = 0; j < Nc; j++)
		{
			phaseData[i*Nc + j] = (uchar)(((PhaseUnwrapped(Nr - i -1,j)-minPhase)/(maxPhase-minPhase))*255.0);
		}
	}
	saveAsImg(fnamestr.c_str(), 8, phaseData, Nc, Nr);
	return TRUE;
}


bool ophSigPU::readConfig(const char * fname)
{
	return false;
}

void ophSigPU::phaseResidues(matrix<Real>& outputResidue)
{
	/*
	This code was written with slight modification based on
	https://kr.mathworks.com/matlabcentral/fileexchange/22504-2d-phase-unwrapping-algorithms
	*/

	matrix<Real> below(Nr, Nc);
	matrix<Real> right(Nr, Nc);
	matrix<Real> belowright(Nr, Nc);
	below.zeros(); right.zeros(); belowright.zeros();

	for (int i = 0; i < Nr-1; i++)
	{
		for (int j = 0; j < Nc-1; j++)
		{
			below(i, j) = PhaseOriginal(i + 1, j);
			right(i, j) = PhaseOriginal(i, j + 1);
			belowright(i, j) = PhaseOriginal(i + 1, j + 1);
		}
	}
	for (int i = 0; i < Nr-1; i++)
	{
		below(i, Nc-1) = PhaseOriginal(i + 1, Nc-1);
	}
	for (int j = 0; j < Nc - 1; j++)
	{
		right(Nr-1, j) = PhaseOriginal(Nr-1, j+1);
	}

	double res1, res2, res3, res4;
	double temp_residue;
	for (int i = 0; i < Nr; i++)
	{
		for (int j = 0; j < Nc; j++)
		{
			res1 = mod2pi(PhaseOriginal(i, j) - below(i, j));
			res2 = mod2pi(below(i, j) - belowright(i,j));
			res3 = mod2pi(belowright(i, j) - right(i, j));
			res4 = mod2pi(right(i, j) - PhaseOriginal(i, j));

			temp_residue = res1 + res2 + res3 + res4;
			if (temp_residue >= 6.)
			{
				outputResidue(i, j) = 1;
			}
			else if (temp_residue <= -6.)
			{
				outputResidue(i, j) = -1;
			}
			else
			{
				outputResidue(i, j) = 0;
			}
		}
	}

	for (int i = 0; i < Nr; i++)
	{
		outputResidue(i, 0) = 0;
		outputResidue(i, Nc - 1) = 0;
	}
	for (int j = 0; j < Nc; j++)
	{
		outputResidue(0, j) = 0;
		outputResidue(Nr - 1, j) = 0;
	}
}

void ophSigPU::branchCuts(matrix<Real>& inputResidue, matrix<Real>& outputBranchCuts)
{
	/*
	This code was written with slight modification based on
	https://kr.mathworks.com/matlabcentral/fileexchange/22504-2d-phase-unwrapping-algorithms
	*/

	int clusterCounter = 1;
	int satelliteResidue = 0;
	matrix<int> residueBinary(Nr, Nc);
	for (int i = 0; i < Nr; i++)
	{
		for (int j = 0; j < Nc; j++)
		{
			if (inputResidue(i, j) != 0)
			{
				residueBinary(i, j) = 1;
			}
			else
			{
				residueBinary(i, j) = 0;
			}	
		}
	}
	matrix<Real> residueBalanced(Nr, Nc);
	residueBalanced.zeros();
	matrix<int> adjacentResidue(Nr, Nc);
	adjacentResidue.zeros();
	int missedResidue = 0;

	int adjacentResidueCount = 0;

	for (int i = 0; i < Nr; i++)
	{
		for (int j = 0; j < Nc; j++)
		{
			if (residueBinary(i, j) == 1)
			{
				int rActive = i;
				int cActive = j;

				int radius = 1;
				int countNearbyResidueFlag = 1;
				clusterCounter = 1;
				adjacentResidue.zeros();
				int chargeCounter = inputResidue(rActive, cActive);
				if (residueBalanced(rActive, cActive) != 1)
				{
					while (chargeCounter != 0)
					{
						for (int m = rActive - radius; m < rActive + radius + 1; m++)
						{
							for (int n = cActive - radius; n < cActive + radius + 1; n++)
							{
								if (((abs(m - rActive) == radius) | (abs(n - cActive) == radius)) & (chargeCounter != 0))
								{
									if ((m < 1) | (m >= Nr - 1) | (n < 1) | (m >= Nc - 1))
									{
										if (m >= Nr - 1) { m = Nr - 1; }
										if (n >= Nc - 1) { n = Nc - 1; }
										if (n < 1) { n = 0; }
										if (m < 1) { m = 0; }
										placeBranchCutsInternal(outputBranchCuts, rActive, cActive, m, n);
										clusterCounter += 1;
										chargeCounter = 0;
										residueBalanced(rActive, cActive) = 1;
									}
									if (residueBinary(m, n))
									{
										if (countNearbyResidueFlag == 1) { adjacentResidue(m, n) = 1; }
										placeBranchCutsInternal(outputBranchCuts, rActive, cActive, m, n);
										clusterCounter += 1;
										if (residueBalanced(m, n) == 0)
										{
											residueBalanced(m, n) = 1;
											chargeCounter += inputResidue(m, n);
										}
										if (chargeCounter == 0) { residueBalanced(rActive, cActive) = 1; }
									}
	 
								}
							}
						}

						double sumAdjacentResidue = 0;
						int adjacentSize = 0;
						for (int ii = 0; ii < Nr; ii++)
						{
							for (int jj = 0; jj < Nc; jj++)
							{
								sumAdjacentResidue += adjacentResidue(ii, jj);
							}
						}
						if (sumAdjacentResidue == 0)
						{
							radius += 1;
							rActive = i;
							cActive = j;
						}
						else
						{
							vector<int> rAdjacent, cAdjacent;
							if (countNearbyResidueFlag == 1)
							{
								findNZ(adjacentResidue, rAdjacent, cAdjacent);
								adjacentSize = rAdjacent.size();
								rActive = rAdjacent[0];
								cActive = cAdjacent[0];
								adjacentResidueCount = 1;
								residueBalanced(rActive, cActive) = 1;
								countNearbyResidueFlag = 0;
							}
							else
							{
								adjacentResidueCount += 1;
								if (adjacentResidueCount <= adjacentSize)
								{
									rActive = rAdjacent[adjacentResidueCount-1];
									cActive = cAdjacent[adjacentResidueCount-1];
								}
								else
								{
									radius += 1;
									rActive = i;
									cActive = j;
									adjacentResidue.zeros();
									countNearbyResidueFlag = 1;
								}
							}
						}
						if (radius > MaxBoxRadius)
						{
							if (clusterCounter != 1)
							{
								missedResidue += 1;
							}
							else
							{
								satelliteResidue += 1;
							}
							chargeCounter = 0;
							while (clusterCounter == 1)
							{
								rActive = i;
								cActive = j;
								for (int m = rActive - radius; m < rActive + radius + 1; m++)
								{
									for (int n = cActive - radius; n < cActive + radius + 1; n++)
									{
										if (((abs(m - rActive) == radius) | (abs(n - cActive) == radius) ))
										{
											if ((m < 1) | (m >= Nr - 1) | (n < 1) | (m >= Nc - 1))
											{
												if (m >= Nr - 1) { m = Nr - 1; }
												if (n >= Nc - 1) { n = Nc - 1; }
												if (n < 1) { n = 0; }
												if (m < 1) { m = 0; }
												placeBranchCutsInternal(outputBranchCuts, rActive, cActive, m, n);
												clusterCounter += 1;
												residueBalanced(rActive, cActive) = 1;
											}
											if (residueBinary(m, n))
											{
												placeBranchCutsInternal(outputBranchCuts, rActive, cActive, m, n);
												clusterCounter += 1;
												residueBalanced(rActive, cActive) = 1;
											}

										}
									}
								}
								radius += 1;
							}
						}
					}
				}
			}
		}
	}

	LOG("Branch cut operation completed. \n");
	LOG("Satellite residues accounted for = %d\n", satelliteResidue);
}

void ophSigPU::placeBranchCutsInternal(matrix<Real>& branchCuts, int r1, int c1, int r2, int c2)
{
	branchCuts(r1, c1) = 1;
	branchCuts(r2, c2) = 1;
	double rdist = abs(r2 - r1); 
	double cdist = abs(c2 - c1);
	int rsign = (r2 > r1) ? 1 : -1;
	int csign = (c2 > c1) ? 1 : -1;
	int r = 0;
	int c = 0;
	if (rdist > cdist)
	{
		for (int i = 0; i <= rdist; i++)
		{
			r = r1 + i*rsign;
			c = c1 + (int)(round(((double)(i))*((double)(csign))*cdist / rdist));
			branchCuts(r, c) = 1;
		}
	}
	else
	{
		for (int j = 0; j <= cdist; j++)
		{
			c = c1 + j*csign;
			r = r1 + (int)(round(((double)(j))*((double)(rsign))*rdist / cdist));
			branchCuts(r, c) = 1;
		}
	}
}

void ophSigPU::floodFill(matrix<Real>& inputBranchCuts)
{
	/*
	This code was written with slight modification based on
	https://kr.mathworks.com/matlabcentral/fileexchange/22504-2d-phase-unwrapping-algorithms
	*/

	matrix<int> adjoin(Nr, Nc);
	adjoin.zeros();

	matrix<int> unwrappedBinary(Nr, Nc);
	unwrappedBinary.zeros();
	
	// set ref phase
	bool refPhaseFound = false;
	for (int i = 1; (i < Nr - 1) & !refPhaseFound; i++)
	{
		for (int j = 1; (j < Nc - 1) & !refPhaseFound; j++)
		{
			if (inputBranchCuts(i, j) == 0)
			{
				adjoin(i - 1, j) = 1;
				adjoin(i + 1, j) = 1;
				adjoin(i, j - 1) = 1;
				adjoin(i, j + 1) = 1;
				unwrappedBinary(i, j) = 1;
				PhaseUnwrapped(i, j) = PhaseOriginal(i, j);
				refPhaseFound = true;
			}
		}
	}

	// floodfill
	int countLimit = 0;
	int adjoinStuck = 0;
	vector<int> rAdjoin;
	vector<int> cAdjoin;
	int rActive = 0;
	int cActive = 0;
	double phaseRef = 0;
	while ((matrixPartialSum(adjoin, 1, 1, Nr - 2, Nc - 2) > 0) & (countLimit < 100))
	{
		//while (countLimit < 100)
		//{
		LOG("%d\n", matrixPartialSum(adjoin, 1, 1, Nr - 2, Nc - 2));
			findNZ(adjoin, rAdjoin, cAdjoin);
			if (adjoinStuck == rAdjoin.size())
			{
				countLimit += 1;
				LOG("countLimit %d\n", countLimit);
			}
			else
			{
				countLimit = 0;
			}
			adjoinStuck = rAdjoin.size();
			for (int i = 0; i < rAdjoin.size(); i++)
			{
				rActive = rAdjoin[i];
				cActive = cAdjoin[i];
				if ((rActive <= Nr - 2) & (rActive >= 1) & (cActive <= Nc - 2) & (cActive >= 1))
				{
					if ((inputBranchCuts(rActive + 1, cActive) == 0) & (unwrappedBinary(rActive + 1, cActive) == 1))
					{
						phaseRef = PhaseUnwrapped(rActive + 1, cActive);
						PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
						unwrappedBinary(rActive, cActive) = 1;
						adjoin(rActive, cActive) = 0;
						if ((unwrappedBinary(rActive - 1, cActive) == 0) & (inputBranchCuts(rActive - 1, cActive) == 0))
						{
							adjoin(rActive - 1, cActive) = 1;
						}
						if ((unwrappedBinary(rActive, cActive-1) == 0) & (inputBranchCuts(rActive, cActive-1) == 0))
						{
							adjoin(rActive, cActive-1) = 1;
						}
						if ((unwrappedBinary(rActive, cActive+1) == 0) & (inputBranchCuts(rActive, cActive+1) == 0))
						{
							adjoin(rActive, cActive+1) = 1;
						}
					}
					if ((inputBranchCuts(rActive - 1, cActive) == 0) & (unwrappedBinary(rActive - 1, cActive) == 1))
					{
						phaseRef = PhaseUnwrapped(rActive - 1, cActive);
						PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
						unwrappedBinary(rActive, cActive) = 1;
						adjoin(rActive, cActive) = 0;
						if ((unwrappedBinary(rActive + 1, cActive) == 0) & (inputBranchCuts(rActive + 1, cActive) == 0))
						{
							adjoin(rActive + 1, cActive) = 1;
						}
						if ((unwrappedBinary(rActive, cActive - 1) == 0) & (inputBranchCuts(rActive, cActive - 1) == 0))
						{
							adjoin(rActive, cActive - 1) = 1;
						}
						if ((unwrappedBinary(rActive, cActive + 1) == 0) & (inputBranchCuts(rActive, cActive + 1) == 0))
						{
							adjoin(rActive, cActive + 1) = 1;
						}
					}
					if ((inputBranchCuts(rActive, cActive +1) == 0) & (unwrappedBinary(rActive, cActive+1) == 1))
					{
						phaseRef = PhaseUnwrapped(rActive, cActive+1);
						PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
						unwrappedBinary(rActive, cActive) = 1;
						adjoin(rActive, cActive) = 0;
						if ((unwrappedBinary(rActive + 1, cActive) == 0) & (inputBranchCuts(rActive + 1, cActive) == 0))
						{
							adjoin(rActive + 1, cActive) = 1;
						}
						if ((unwrappedBinary(rActive, cActive - 1) == 0) & (inputBranchCuts(rActive, cActive - 1) == 0))
						{
							adjoin(rActive, cActive - 1) = 1;
						}
						if ((unwrappedBinary(rActive -1 , cActive) == 0) & (inputBranchCuts(rActive-1, cActive) == 0))
						{
							adjoin(rActive-1, cActive) = 1;
						}
					}
					if ((inputBranchCuts(rActive, cActive-1) == 0) & (unwrappedBinary(rActive, cActive-1) == 1))
					{
						phaseRef = PhaseUnwrapped(rActive, cActive-1);
						PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
						unwrappedBinary(rActive, cActive) = 1;
						adjoin(rActive, cActive) = 0;
						if ((unwrappedBinary(rActive + 1, cActive) == 0) & (inputBranchCuts(rActive + 1, cActive) == 0))
						{
							adjoin(rActive + 1, cActive) = 1;
						}
						if ((unwrappedBinary(rActive -1, cActive) == 0) & (inputBranchCuts(rActive-1, cActive) == 0))
						{
							adjoin(rActive-1, cActive) = 1;
						}
						if ((unwrappedBinary(rActive, cActive + 1) == 0) & (inputBranchCuts(rActive, cActive + 1) == 0))
						{
							adjoin(rActive, cActive + 1) = 1;
						}
					}

				}

			}
		//}
	}

	adjoin.zeros();
	for (int i = 1; i <= Nr - 2; i++)
	{
		for (int j = 1; j < Nc - 2; j++)
		{
			if ((inputBranchCuts(i, j) == 1) & 
				((inputBranchCuts(i + 1, j) == 0) | 
				(inputBranchCuts(i - 1, j) == 0) | 
					(inputBranchCuts(i, j - 1) == 0) | 
					(inputBranchCuts(i, j + 1) == 0)))
			{
				adjoin(i, j) = 1;
			}
		}
	}
	findNZ(adjoin, rAdjoin, cAdjoin);
	for (int i = 0; i < rAdjoin.size(); i++)
	{
		rActive = rAdjoin[i];
		cActive = cAdjoin[i];

		if (unwrappedBinary(rActive + 1, cActive) == 1)
		{
			phaseRef = PhaseUnwrapped(rActive + 1, cActive);
			PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
			unwrappedBinary(rActive, cActive) = 1;
			adjoin(rActive, cActive) = 0;
		}
		if (unwrappedBinary(rActive - 1, cActive) == 1)
		{
			phaseRef = PhaseUnwrapped(rActive - 1, cActive);
			PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
			unwrappedBinary(rActive, cActive) = 1;
			adjoin(rActive, cActive) = 0;
		}
		if (unwrappedBinary(rActive, cActive+1) == 1)
		{
			phaseRef = PhaseUnwrapped(rActive, cActive+1);
			PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
			unwrappedBinary(rActive, cActive) = 1;
			adjoin(rActive, cActive) = 0;
		}
		if (unwrappedBinary(rActive, cActive-1) == 1)
		{
			phaseRef = PhaseUnwrapped(rActive, cActive-1);
			PhaseUnwrapped(rActive, cActive) = unwrap(phaseRef, PhaseOriginal(rActive, cActive));
			unwrappedBinary(rActive, cActive) = 1;
			adjoin(rActive, cActive) = 0;
		}
	}
	LOG("Floodfill completed\n");
}

double ophSigPU::unwrap(double phaseRef, double phaseInput)
{
	double diff = phaseInput - phaseRef;
	double modval = mod2pi(diff);
	return phaseRef + modval;
}

double ophSigPU::mod2pi(double phase)
{
	double temp;
	temp = fmod(phase, 2 * M_PI);
	if (temp > M_PI)
	{
		temp = temp - 2 * M_PI;
	}
	if (temp < -M_PI)
	{
		temp = temp + 2 * M_PI;
	}
	return temp;
}

void ophSigPU::findNZ(matrix<int>& inputMatrix, vector<int>& row, vector<int>& col)
{
	row.clear();
	col.clear();
	for (int i = 0; i < inputMatrix.size(_X); i++)
	{
		for (int j = 0; j < inputMatrix.size(_Y); j++)
		{
			if (inputMatrix(i,j) != 0)
			{
				row.push_back(i);
				col.push_back(j);
			}
		}
	}
}

int ophSigPU::matrixPartialSum(matrix<int>& inputMatrix, int r1, int c1, int r2, int c2)
{
	int outputsum = 0;
	for (int i = r1; i <= r2; i++)
	{
		for (int j = c1; j <= c2; j++)
		{
			outputsum += inputMatrix(i, j);
		}
	}
	return outputsum;
}
