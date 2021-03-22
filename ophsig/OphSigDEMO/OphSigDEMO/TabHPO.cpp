// TabHPO.cpp: 구현 파일
//

#include "pch.h"
#include "OphSigDEMO.h"
#include "TabHPO.h"
#include "afxdialogex.h"


// TabHPO 대화 상자

IMPLEMENT_DYNAMIC(TabHPO, CDialogEx)

TabHPO::TabHPO(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_TAB_HPO, pParent)
	, _wavelength(0.)
	, _reductionRate(0.)
{

}

TabHPO::~TabHPO()
{
}

void TabHPO::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_HPO_WAVELENGTH, _wavelength);
	DDX_Text(pDX, IDC_EDIT_REDUCTION_RATE, _reductionRate);
	_wavelength = _wavelength *0.000001;
}


BEGIN_MESSAGE_MAP(TabHPO, CDialogEx)
END_MESSAGE_MAP()


// TabHPO 메시지 처리기
