// TabSF.cpp: 구현 파일
//

#include "pch.h"
#include "OphSigDEMO.h"
#include "TabSF.h"
#include "afxdialogex.h"


// TabSF 대화 상자

IMPLEMENT_DYNAMIC(TabSF, CDialogEx)

TabSF::TabSF(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_TAB_SF, pParent)
	, _searchRangeMin(0.)
	, _searchRangeMax(0.)
	, _searchCount(0)
	, _threshold(0.)
{

}

TabSF::~TabSF()
{
}

void TabSF::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_SEARCH_RANGE_MIN, _searchRangeMin);
	DDX_Text(pDX, IDC_EDIT_SEARCH_RANGE_MAX, _searchRangeMax);
	DDX_Text(pDX, IDC_EDIT_SEARCH_COUNT, _searchCount);
	DDX_Text(pDX, IDC_EDIT_THRESHOLD, _threshold);
}


BEGIN_MESSAGE_MAP(TabSF, CDialogEx)
END_MESSAGE_MAP()


// TabSF 메시지 처리기
