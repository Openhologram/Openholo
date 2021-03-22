#pragma once


// TabHPO 대화 상자

class TabHPO : public CDialogEx
{
	DECLARE_DYNAMIC(TabHPO)

public:
	TabHPO(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~TabHPO();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DLG_TAB_HPO };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
public:
	double _wavelength;
	float _reductionRate;
};
