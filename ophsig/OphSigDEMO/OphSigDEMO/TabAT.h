#pragma once


// TabAT 대화 상자

class TabAT : public CDialogEx
{
	DECLARE_DYNAMIC(TabAT)

public:
	TabAT(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~TabAT();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DLG_TAB_AT };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
};
