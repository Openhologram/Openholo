#pragma once


// TabCAC 대화 상자

class TabCAC : public CDialogEx
{
	DECLARE_DYNAMIC(TabCAC)

public:
	TabCAC(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~TabCAC();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DLG_TAB_CAC };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
public:
	double _wavelength[3];
	float _thickness;
	float _radius;
};
