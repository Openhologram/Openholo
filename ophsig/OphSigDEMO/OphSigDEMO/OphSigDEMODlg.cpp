
// OphSigDEMODlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "OphSigDEMO.h"
#include "OphSigDEMODlg.h"
#include "afxdialogex.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// COphSigDEMODlg 대화 상자

COphSigDEMODlg::COphSigDEMODlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_OPHSIGDEMO_DIALOG, pParent)
	, _configPath(_T(""))
	, _sourcePath(_T(""))
	, _resultPath(_T(""))	
	, _distance(0)
	, _NA(0)
	, _Gray(0)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
	_pwndShowTrans = NULL;
	_pwndShowExtract = NULL;
}

void COphSigDEMODlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_TAB_TRANS, _ctrTabTrans);
	DDX_Control(pDX, IDC_TAB_EXTRACT, _ctrTabExtract);
	DDX_Control(pDX, IDC_PREVIEW, _Preview);
	//DDX_Text(pDX, IDC_EDIT_CONFIG_PATH, _configPath);
	DDX_Text(pDX, IDC_EDIT_SOURCE_PATH, _sourcePath);
	DDX_Text(pDX, IDC_EDIT_RESULT_PATH, _resultPath);
	DDX_Text(pDX, IDC_EDIT_NUMOFPIXEL_X, _numOfPixel[_X]);
	DDX_Text(pDX, IDC_EDIT_NUMOFPIXEL_Y, _numOfPixel[_Y]);
	DDX_Text(pDX, IDC_EDIT_SIZEOFAREA_X, _sizeOfArea[_X]);
	DDX_Text(pDX, IDC_EDIT_SIZEOFAREA_Y, _sizeOfArea[_Y]);
	DDX_Text(pDX, IDC_EDIT_DISTANCE, _distance);
	DDX_Text(pDX, IDC_EDIT_NA, _NA);
	DDX_Control(pDX, IDC_BTN_PREVIEW, botten_Gray);
	DDX_Control(pDX, IDC_GPU, m_GPU);
	DDX_Control(pDX, IDC_COMBO1, m_display_b);
}

BEGIN_MESSAGE_MAP(COphSigDEMODlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB_TRANS, &COphSigDEMODlg::OnTcnSelchangeTabTrans)
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB_EXTRACT, &COphSigDEMODlg::OnTcnSelchangeTabExtract)
	ON_BN_CLICKED(IDC_BTN_TRANSFORM, &COphSigDEMODlg::OnBnClickedBtnTransform)
	ON_BN_CLICKED(IDC_BTN_PREVIEW, &COphSigDEMODlg::OnBnClickedBtnPreview)
	//ON_BN_CLICKED(IDC_BTN_CONFIG_PATH, &COphSigDEMODlg::OnBnClickedBtnConfigPath)
	ON_BN_CLICKED(IDC_BTN_SOURCE_PATH, &COphSigDEMODlg::OnBnClickedBtnSourcePath)
	ON_BN_CLICKED(IDC_BTN_RESULT_PATH, &COphSigDEMODlg::OnBnClickedBtnResultPath)
	//ON_STN_CLICKED(IDC_PREVIEW, &COphSigDEMODlg::OnStnClickedPreview)
	//ON_STN_CLICKED(IDC_PREVIEW, &COphSigDEMODlg::OnStnClickedPreview)
	//ON_EN_CHANGE(IDC_EDIT_SOURCE_PATH, &COphSigDEMODlg::OnEnChangeEditSourcePath)
	ON_BN_CLICKED(IDC_BTN_RECONSTRUCT, &COphSigDEMODlg::OnBnClickedBtnReconstruct)
	
	ON_CBN_SELCHANGE(IDC_COMBO1, &COphSigDEMODlg::OnCbnSelchangeCombo1)
END_MESSAGE_MAP()

// COphSigDEMODlg 메시지 처리기

BOOL COphSigDEMODlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	// Initialization Tab control trans part
	_ctrTabTrans.InsertItem(1, _T("OFFA"));
	_ctrTabTrans.InsertItem(2, _T("HPO"));
	_ctrTabTrans.InsertItem(3, _T("CAC"));

	CRect rect;
	_ctrTabTrans.GetClientRect(&rect);

	_tabOFFA.Create(IDD_DLG_TAB_OFFA, &_ctrTabTrans);
	_tabOFFA.SetWindowPos(NULL, 5, 25, rect.Width() - 10, rect.Height() - 30, SWP_SHOWWINDOW | SWP_NOZORDER);
	_pwndShowTrans = &_tabOFFA;

	_tabHPO.Create(IDD_DLG_TAB_HPO, &_ctrTabTrans);
	_tabHPO.SetWindowPos(NULL, 5, 25, rect.Width() - 10, rect.Height() - 30, SWP_NOZORDER);

	_tabCAC.Create(IDD_DLG_TAB_CAC, &_ctrTabTrans);
	_tabCAC.SetWindowPos(NULL, 5, 25, rect.Width() - 10, rect.Height() - 30, SWP_NOZORDER);

	// Initialization Tab control extraction part
	_ctrTabExtract.InsertItem(1, _T("SF"));
	_ctrTabExtract.InsertItem(2, _T("AT"));

	_tabSF.Create(IDD_DLG_TAB_SF, &_ctrTabExtract);
	_tabSF.SetWindowPos(NULL, 5, 25, rect.Width() - 10, rect.Height() - 30, SWP_SHOWWINDOW | SWP_NOZORDER);
	_pwndShowExtract = &_tabSF;

	_tabAT.Create(IDD_DLG_TAB_AT, &_ctrTabExtract);
	_tabAT.SetWindowPos(NULL, 5, 25, rect.Width() - 10, rect.Height() - 30, SWP_NOZORDER);


	_configPath = "config/TestSpecHoloParam.xml";
	_sourcePath = "source/OffAxis/3_point.ohc";
	_resultPath = "result/OffAxis/offaxis.ohc";

	_numOfPixel[_X] = 1024;
	_numOfPixel[_Y] = 1024;
	_sizeOfArea[_X] = 0.01f;
	_sizeOfArea[_Y] = 0.01f;
	_NA = 0.1;
	_distance = 0;

	_tabSF._searchCount = 100;
	_tabSF._searchRangeMax = 0.2f;
	_tabSF._searchRangeMin = 0.1f;
	_tabSF._threshold = 0.1f;

	_tabCAC._thickness = 9.f;
	_tabCAC._radius = 0.9f;
	_tabCAC._wavelength[_R] = 0.633f;
	_tabCAC._wavelength[_G] = 0.532f;
	_tabCAC._wavelength[_B] = 0.473f;

	_tabHPO._reductionRate = 0.2f;
	
	_tabOFFA._angle[_X] = 0.f;
	_tabOFFA._angle[_Y] = 0.f;


	oph = new ophSig;

	UpdateData(FALSE);
	_tabOFFA.UpdateData(FALSE);
	_tabCAC.UpdateData(FALSE);
	_tabHPO.UpdateData(FALSE);
	_tabSF.UpdateData(FALSE);

	m_display_b.AddString(_T("Real_data"));
	m_display_b.AddString(_T("Imag_data"));
	m_display_b.AddString(_T("Abs_data"));
	m_display_b.SetCurSel(0);
	_View = 0;
	_Vflag = false;
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void COphSigDEMODlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void COphSigDEMODlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR COphSigDEMODlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void COphSigDEMODlg::OnTcnSelchangeTabTrans(NMHDR* pNMHDR, LRESULT* pResult)
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	if (_pwndShowTrans != NULL) {
		_pwndShowTrans->ShowWindow(SW_HIDE);
		_pwndShowTrans = NULL;
	}

	int nIndex = _ctrTabTrans.GetCurSel();

	switch (nIndex) {
	case 0: _tabOFFA.ShowWindow(SW_SHOW);
		_pwndShowTrans = &_tabOFFA;
		break;
	case 1: _tabHPO.ShowWindow(SW_SHOW);
		_pwndShowTrans = &_tabHPO;
		break;
	case 2: _tabCAC.ShowWindow(SW_SHOW);
		_pwndShowTrans = &_tabCAC;
		break;
	}
	*pResult = 0;
}

void COphSigDEMODlg::OnTcnSelchangeTabExtract(NMHDR* pNMHDR, LRESULT* pResult)
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	if (_pwndShowExtract != NULL) {
		_pwndShowExtract->ShowWindow(SW_HIDE);
		_pwndShowExtract = NULL;
	}

	int nIndex = _ctrTabExtract.GetCurSel();

	switch (nIndex) {
	case 0: _tabSF.ShowWindow(SW_SHOW);
		_pwndShowExtract = &_tabSF;
		break;
	case 1: _tabAT.ShowWindow(SW_SHOW);
		_pwndShowExtract = &_tabAT;
		break;
	}
	*pResult = 0;
}


void COphSigDEMODlg::OnBnClickedBtnTransform()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	
	int cout = 0;
	int check = m_GPU.GetCheck();
	bool Device;
	if (check == 0)
	{
		Device = true;
	}
	else if (check == 1)
	{
		Device = false;
	}
	UpdateData(TRUE);
	oph->loadAsOhc((CStringA)_sourcePath);
	oph->Wavenumber_output(bit);
	CRect rect;
	HDC previewhdc = _Preview.GetDC()->m_hDC;
	_Preview.GetWindowRect(rect);
	uchar *data = new uchar[_numOfPixel[_X]* _numOfPixel[_Y]*bit];
	if (bit == 1)
	{
		cout = 1;
	}
	else if (bit == 3)
	{
		cout = 0;
	}
	BITMAPINFO *bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + (256 * sizeof(RGBQUAD))*cout);
	if (bit == 1)
	{
		for (int i = 0; i < 256; i++)
		{
			bitInfo->bmiColors[i].rgbBlue = bitInfo->bmiColors[i].rgbGreen = bitInfo->bmiColors[i].rgbRed = i;
			bitInfo->bmiColors[i].rgbReserved = 0;
		}
	}
	bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bitInfo->bmiHeader.biBitCount = bit*8;
	bitInfo->bmiHeader.biWidth = _numOfPixel[_X];
	bitInfo->bmiHeader.biHeight = _numOfPixel[_Y];
	bitInfo->bmiHeader.biPlanes = 1;
	bitInfo->bmiHeader.biCompression = BI_RGB;
	bitInfo->bmiHeader.biClrImportant = 0;
	bitInfo->bmiHeader.biClrUsed = 0;
	bitInfo->bmiHeader.biSizeImage = 0;
	bitInfo->bmiHeader.biXPelsPerMeter = 0;
	bitInfo->bmiHeader.biYPelsPerMeter = 0;

	/////////
	oph->Parameter_Set(_numOfPixel[_X], _numOfPixel[_Y], _sizeOfArea[_X], _sizeOfArea[_Y], _NA);

	int nIndex = _ctrTabTrans.GetCurSel();
	
	
	switch (nIndex) {
	case 0:
		_tabOFFA.UpdateData(TRUE);
		oph->setMode(Device);
		oph->sigConvertOffaxis(_tabOFFA._angle[0], _tabOFFA._angle[1]);
		oph->Data_output(data, _View, 8 * bit);
		oph->saveAsOhc((CStringA)_resultPath);
		SetStretchBltMode(previewhdc, COLORONCOLOR);
		StretchDIBits(previewhdc, 0, 0, _numOfPixel[_X]/2, _numOfPixel[_Y] / 2,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			data, bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		break;
	case 1: 
		_tabHPO.UpdateData(TRUE);
		oph->setMode(Device);
		oph->sigConvertHPO(_distance, _tabHPO._reductionRate);
		oph->Data_output(data, _View, 8 * bit);
		oph->saveAsOhc((CStringA)_resultPath);
		SetStretchBltMode(previewhdc, COLORONCOLOR);
		StretchDIBits(previewhdc, 0, 0, _numOfPixel[_X] / 2, _numOfPixel[_Y] / 2,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			data, bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		break;
	case 2:
		_tabCAC.UpdateData(TRUE);
		oph->setMode(Device);
		oph->focal_length_Set(0.0246, 0.0249, 0.0251, _tabCAC._radius);
		oph->sigConvertCAC(_tabCAC._wavelength[_R], _tabCAC._wavelength[_G] , _tabCAC._wavelength[_B]);
		oph->propagationHolo(_distance);
		oph->Data_output(data, _View,8 * bit);
		oph->saveAsOhc((CStringA)_resultPath);
		SetStretchBltMode(previewhdc, COLORONCOLOR);
	
		StretchDIBits(previewhdc, 0, 0, _numOfPixel[_X]/2 , _numOfPixel[_Y]/2 ,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			data, bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		break;
	}
	_Vflag = true;
	delete[] data;
}


void COphSigDEMODlg::OnBnClickedBtnPreview()
{
	int P_cout = 0;
	int check = m_GPU.GetCheck();
	bool Device;
	if (check == 0)
	{
		Device = true;
	}
	else if (check == 1)
	{
		Device = false;
	}
	UpdateData(TRUE);
	oph->loadAsOhc((CStringA)_sourcePath);
	oph->Wavenumber_output(bit);
	CRect P_rect;
	HDC P_previewhdc = _Preview.GetDC()->m_hDC;
	_Preview.GetWindowRect(P_rect);
	uchar *P_data = new uchar[_numOfPixel[_X] * _numOfPixel[_Y] * bit];
	if (bit == 1)
	{
		P_cout = 1;
	}
	else if (bit == 3)
	{
		P_cout = 0;
	}
	BITMAPINFO *P_bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + (256 * sizeof(RGBQUAD))*P_cout);
	if (bit == 1)
	{
		for (int i = 0; i < 256; i++)
		{
			P_bitInfo->bmiColors[i].rgbBlue = P_bitInfo->bmiColors[i].rgbGreen =  P_bitInfo->bmiColors[i].rgbRed = i;
			P_bitInfo->bmiColors[i].rgbReserved = 0;
		}
	}
	P_bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	P_bitInfo->bmiHeader.biBitCount = bit * 8;
	P_bitInfo->bmiHeader.biWidth = _numOfPixel[_X];
	P_bitInfo->bmiHeader.biHeight = _numOfPixel[_Y];
	P_bitInfo->bmiHeader.biPlanes = 1;
	P_bitInfo->bmiHeader.biCompression = BI_RGB;
	P_bitInfo->bmiHeader.biClrImportant = 0;
	P_bitInfo->bmiHeader.biClrUsed = 0;
	P_bitInfo->bmiHeader.biSizeImage = 0;
	P_bitInfo->bmiHeader.biXPelsPerMeter = 0;
	P_bitInfo->bmiHeader.biYPelsPerMeter = 0;

	/////////
	oph->Parameter_Set(_numOfPixel[_X], _numOfPixel[_Y], _sizeOfArea[_X], _sizeOfArea[_Y], _NA);

	int nIndex = _ctrTabExtract.GetCurSel();
	float depth = 0;
	switch (nIndex) {
	case 0: 
		_tabSF.UpdateData(TRUE);
		oph->setMode(Device);
		depth = oph->sigGetParamSF(_tabSF._searchRangeMax, _tabSF._searchRangeMin, _tabSF._searchCount, _tabSF._threshold);
		oph->propagationHolo(depth);
		oph->Data_output(P_data, _View,8 * bit);
		oph->saveAsOhc((CStringA)_resultPath);
		SetStretchBltMode(P_previewhdc, COLORONCOLOR);
		StretchDIBits(P_previewhdc, 0, 0, _numOfPixel[_X] / 2, _numOfPixel[_Y] / 2,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			P_data, P_bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		break;

	case 1: 
		_tabAT.UpdateData(TRUE);
		oph->setMode(Device);
		depth = oph->sigGetParamAT();
		oph->propagationHolo(-depth);
		oph->Data_output(P_data, _View,8 * bit);
		oph->saveAsOhc((CStringA)_resultPath);
		SetStretchBltMode(P_previewhdc, COLORONCOLOR);
		StretchDIBits(P_previewhdc, 0, 0, _numOfPixel[_X]/2 , _numOfPixel[_Y]/2 ,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			P_data, P_bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		break;
	}
	_Vflag = true;
	_distance = depth;
	delete[] P_data;
}


//void COphSigDEMODlg::OnBnClickedBtnConfigPath()
//{
//	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
//	CString str = _T("All files(*.*)|*.*|"); // 모든 파일 표시
//		// _T("Excel 파일 (*.xls, *.xlsx) |*.xls; *.xlsx|"); 와 같이 확장자를 제한하여 표시할 수 있음
//	CFileDialog dlg(TRUE, _T("*.dat"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, str, this);
//
//	if (dlg.DoModal() == IDOK)
//	{
//		CString strPathName = dlg.GetPathName();
//		// 파일 경로를 가져와 사용할 경우, Edit Control에 값 저장
//		SetDlgItemText(IDC_EDIT_CONFIG_PATH, strPathName);
//	}
//}


void COphSigDEMODlg::OnBnClickedBtnSourcePath()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString str = _T("All files(*.*)|*.*|"); // 모든 파일 표시
	// _T("Excel 파일 (*.xls, *.xlsx) |*.xls; *.xlsx|"); 와 같이 확장자를 제한하여 표시할 수 있음
	CFileDialog dlg(TRUE, _T("*.dat"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, str, this);

	if (dlg.DoModal() == IDOK)
	{
		CString strPathName = dlg.GetPathName();
		// 파일 경로를 가져와 사용할 경우, Edit Control에 값 저장
		SetDlgItemText(IDC_EDIT_SOURCE_PATH, strPathName);
	}
}


void COphSigDEMODlg::OnBnClickedBtnResultPath()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	CString str = _T("All files(*.*)|*.*|"); // 모든 파일 표시
	// _T("Excel 파일 (*.xls, *.xlsx) |*.xls; *.xlsx|"); 와 같이 확장자를 제한하여 표시할 수 있음
	CFileDialog dlg(TRUE, _T("*.dat"), NULL, OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, str, this);

	if (dlg.DoModal() == IDOK)
	{
		CString strPathName = dlg.GetPathName();
		// 파일 경로를 가져와 사용할 경우, Edit Control에 값 저장
		SetDlgItemText(IDC_EDIT_RESULT_PATH, strPathName);
	}
}

bool COphSigDEMODlg::checkExtension(const char* fname, const char* ext)
{
	string filename(fname);
	string fext(ext);
	if (0 == filename.substr(filename.find_last_of(".")).compare(fext))
		return true;
	else
		return false;
}


void COphSigDEMODlg::OnBnClickedBtnReconstruct()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	int R_cout = 0;
	int check = m_GPU.GetCheck();
	bool Device;
	if (check == 0)
	{
		Device = true;
	}
	else if (check == 1)
	{
		Device = false;
	}
	UpdateData(TRUE);
	oph->loadAsOhc((CStringA)_sourcePath);
	oph->Wavenumber_output(bit);
	CRect R_rect;
	HDC R_previewhdc = _Preview.GetDC()->m_hDC;
	_Preview.GetWindowRect(R_rect);
	uchar *R_data = new uchar[_numOfPixel[_X] * _numOfPixel[_Y] * bit];
	if (bit == 1)
	{
		R_cout = 1;
	}
	else if (bit == 3)
	{
		R_cout = 0;
	}
	BITMAPINFO *R_bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + (256 * sizeof(RGBQUAD))*R_cout);
	if (bit == 1)
	{
		for (int i = 0; i < 256; i++)
		{
			R_bitInfo->bmiColors[i].rgbBlue = R_bitInfo->bmiColors[i].rgbGreen = R_bitInfo->bmiColors[i].rgbRed = i;
			R_bitInfo->bmiColors[i].rgbReserved = 0;
		}
	}
	R_bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	R_bitInfo->bmiHeader.biBitCount = bit * 8;
	R_bitInfo->bmiHeader.biWidth = _numOfPixel[_X];
	R_bitInfo->bmiHeader.biHeight = _numOfPixel[_Y];
	R_bitInfo->bmiHeader.biPlanes = 1;
	R_bitInfo->bmiHeader.biCompression = BI_RGB;
	R_bitInfo->bmiHeader.biClrImportant = 0;
	R_bitInfo->bmiHeader.biClrUsed = 0;
	R_bitInfo->bmiHeader.biSizeImage = 0;
	R_bitInfo->bmiHeader.biXPelsPerMeter = 0;
	R_bitInfo->bmiHeader.biYPelsPerMeter = 0;

	/////////
	oph->Parameter_Set(_numOfPixel[_X], _numOfPixel[_Y], _sizeOfArea[_X], _sizeOfArea[_Y], _NA);

	int nIndex = _ctrTabTrans.GetCurSel();
	int color = botten_Gray.GetCheck();


	UpdateData(TRUE);
	oph->loadAsOhc((CStringA)_sourcePath);
	oph->setMode(Device);
	oph->propagationHolo(_distance);
	oph->Data_output(R_data, _View, 8* bit);
	oph->saveAsOhc((CStringA)_resultPath);
	SetStretchBltMode(R_previewhdc, COLORONCOLOR);
	StretchDIBits(R_previewhdc, 0, 0, _numOfPixel[_X] / 2, _numOfPixel[_Y] / 2,
		0, 0, _numOfPixel[_X], _numOfPixel[_Y],
		R_data, R_bitInfo,
		DIB_RGB_COLORS, SRCCOPY);
	_Vflag = true;
}




void COphSigDEMODlg::OnCbnSelchangeCombo1()
{
	_View = m_display_b.GetCurSel();
	if (_Vflag == true)
	{
		uchar *data = new uchar[_numOfPixel[_X] * _numOfPixel[_Y] * bit];
		HDC previewhdc = _Preview.GetDC()->m_hDC;
		int cout;
		if (bit == 1)
		{
			cout = 1;
		}
		else if (bit == 3)
		{
			cout = 0;
		}
		BITMAPINFO *bitInfo = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + (256 * sizeof(RGBQUAD))*cout);
		if (bit == 1)
		{
			for (int i = 0; i < 256; i++)
			{
				bitInfo->bmiColors[i].rgbBlue = bitInfo->bmiColors[i].rgbGreen = bitInfo->bmiColors[i].rgbRed = i;
				bitInfo->bmiColors[i].rgbReserved = 0;
			}
		}
		bitInfo->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		bitInfo->bmiHeader.biBitCount = bit * 8;
		bitInfo->bmiHeader.biWidth = _numOfPixel[_X];
		bitInfo->bmiHeader.biHeight = _numOfPixel[_Y];
		bitInfo->bmiHeader.biPlanes = 1;
		bitInfo->bmiHeader.biCompression = BI_RGB;
		bitInfo->bmiHeader.biClrImportant = 0;
		bitInfo->bmiHeader.biClrUsed = 0;
		bitInfo->bmiHeader.biSizeImage = 0;
		bitInfo->bmiHeader.biXPelsPerMeter = 0;
		bitInfo->bmiHeader.biYPelsPerMeter = 0;

		oph->Data_output(data, _View, 8 * bit);
		SetStretchBltMode(previewhdc, COLORONCOLOR);
		StretchDIBits(previewhdc, 0, 0, _numOfPixel[_X] / 2, _numOfPixel[_Y] / 2,
			0, 0, _numOfPixel[_X], _numOfPixel[_Y],
			data, bitInfo,
			DIB_RGB_COLORS, SRCCOPY);
		delete[] data;
	}
}
