// TabOFFA.cpp: 구현 파일
//

#include "pch.h"
#include "OphSigDEMO.h"
#include "TabOFFA.h"
#include "afxdialogex.h"


// TabOFFA 대화 상자

IMPLEMENT_DYNAMIC(TabOFFA, CDialogEx)

TabOFFA::TabOFFA(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_TAB_OFFA, pParent)
	, _wavelength(0.)
{

}

TabOFFA::~TabOFFA()
{
}

void TabOFFA::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT_ANGLE_X, _angle[0]);
	DDX_Text(pDX, IDC_EDIT_ANGLE_Y, _angle[1]);
	DDX_Text(pDX, IDC_EDIT_OFFA_WAVELENGTH, _wavelength);
	_wavelength = _wavelength *0.000001;
}


BEGIN_MESSAGE_MAP(TabOFFA, CDialogEx)
END_MESSAGE_MAP()


// TabOFFA 메시지 처리기
