// TabCAC.cpp: 구현 파일
//

#include "pch.h"
#include "OphSigDEMO.h"
#include "TabCAC.h"
#include "afxdialogex.h"


// TabCAC 대화 상자

IMPLEMENT_DYNAMIC(TabCAC, CDialogEx)

TabCAC::TabCAC(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_TAB_CAC, pParent)
	, _thickness(0.)
	, _radius(0.)
{

}

TabCAC::~TabCAC()
{
}

void TabCAC::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_WAVELENGTH_R, _wavelength[0]);
	DDX_Text(pDX, IDC_EDIT_WAVELENGTH_G, _wavelength[1]);
	DDX_Text(pDX, IDC_EDIT_WAVELENGTH_B, _wavelength[2]);
	DDX_Text(pDX, IDC_EDIT_THICKNESS, _thickness);
	DDX_Text(pDX, IDC_EDIT_RADIUS, _radius);
	_wavelength[0] = _wavelength[0] * 0.000001;
	_wavelength[1] = _wavelength[1] * 0.000001;
	_wavelength[2] = _wavelength[2] * 0.000001;
}


BEGIN_MESSAGE_MAP(TabCAC, CDialogEx)
END_MESSAGE_MAP()


// TabCAC 메시지 처리기
