#pragma once


// TabSF 대화 상자

class TabSF : public CDialogEx
{
	DECLARE_DYNAMIC(TabSF)

public:
	TabSF(CWnd* pParent = nullptr);   // 표준 생성자입니다.
	virtual ~TabSF();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_DLG_TAB_SF };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

	DECLARE_MESSAGE_MAP()
public:
	float _searchRangeMin;
	float _searchRangeMax;
	unsigned int _searchCount;
	float _threshold;
};
