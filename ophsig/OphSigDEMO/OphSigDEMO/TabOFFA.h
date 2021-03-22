#pragma once


// TabOFFA 대화 상자

class TabOFFA : public CDialogEx
{
	DECLARE_DYNAMIC(TabOFFA)

public:
	TabOFFA(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~TabOFFA();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DLG_TAB_OFFA };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
public:
	float _angle[2];	
	double _wavelength;
};
