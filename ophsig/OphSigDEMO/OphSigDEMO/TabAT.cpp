// TabAT.cpp: 구현 파일
//

#include "pch.h"
#include "OphSigDEMO.h"
#include "TabAT.h"
#include "afxdialogex.h"


// TabAT 대화 상자

IMPLEMENT_DYNAMIC(TabAT, CDialogEx)

TabAT::TabAT(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_DLG_TAB_AT, pParent)
{

}

TabAT::~TabAT()
{
}

void TabAT::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(TabAT, CDialogEx)
END_MESSAGE_MAP()


// TabAT 메시지 처리기
