
// OphSigDEMODlg.h: 헤더 파일
//

#pragma once

#include "ophSig.h"
#include "TabOFFA.h"
#include "TabHPO.h"
#include "TabCAC.h"
#include "TabAT.h"
#include "TabSF.h"
#include "afxwin.h"


#pragma comment(lib,"openholo.lib")
#pragma comment(lib,"ophsig.lib")

//#define _X	0
//#define _Y	1

// COphSigDEMODlg 대화 상자
class COphSigDEMODlg : public CDialogEx
{
// 생성입니다.
public:
	COphSigDEMODlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.
	~COphSigDEMODlg() 
	{ 
		oph->release();
	};

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_OPHSIGDEMO_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:

	ophSig* oph;
	uchar* display;

	TabOFFA _tabOFFA;
	TabHPO _tabHPO;
	TabCAC _tabCAC;

	TabSF _tabSF;
	TabAT _tabAT;

	CWnd* _pwndShowTrans;
	CWnd* _pwndShowExtract;

	CTabCtrl _ctrTabTrans;
	CTabCtrl _ctrTabExtract;

	afx_msg void OnTcnSelchangeTabTrans(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnTcnSelchangeTabExtract(NMHDR* pNMHDR, LRESULT* pResult);	

	CStatic _Preview;
	CImage m_imag;

	CString _configPath;
	CString _sourcePath;
	CString _resultPath;

	int _wavelengthNum;

	int _numOfPixel[2];
	float _sizeOfArea[2];	
	float _distance;
	int _View;
	bool _Vflag;
	int  bit = 0;

	afx_msg void OnBnClickedBtnTransform();
	afx_msg void OnBnClickedBtnPreview();
	afx_msg void OnBnClickedBtnConfigPath();
	afx_msg void OnBnClickedBtnSourcePath();
	afx_msg void OnBnClickedBtnResultPath();
	float _NA;

	bool readConfig(const char* fname);
	bool checkExtension(const char* fname, const char* ext);
	
	afx_msg void OnStnClickedPreview();
	afx_msg void OnEnChangeEditSourcePath();
	afx_msg void OnBnClickedBtnReconstruct();
	double _Gray;
	CButton botten_Gray;
	afx_msg void OnBnClickedGray();
	afx_msg void OnBnClickedCheck1();
	CButton m_GPU;
	CComboBox m_display_b;
	afx_msg void OnCbnSelchangeCombo1();
};
