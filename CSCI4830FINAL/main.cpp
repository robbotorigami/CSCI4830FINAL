#include <windows.h>
#include <string>
#include <shlobj.h>
#include <iostream>
#include <sstream>
#include <direct.h>
#include <cstdlib>
#include <cstdio>

#include "ApplicationInterface.h"



static int CALLBACK BrowseCallbackProc(HWND hwnd, UINT uMsg, LPARAM lParam, LPARAM lpData)
{

	if (uMsg == BFFM_INITIALIZED)
	{
		std::string tmp = (const char *)lpData;
		std::cout << "path: " << tmp << std::endl;
		SendMessage(hwnd, BFFM_SETSELECTION, TRUE, lpData);
	}

	return 0;
}

std::string BrowseFolder(std::string saved_path)
{
	TCHAR path[MAX_PATH];

	const char * path_param = saved_path.c_str();

	BROWSEINFO bi = { 0 };
	bi.lpszTitle = ("Browse for folder...");
	bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
	bi.lpfn = BrowseCallbackProc;
	bi.lParam = (LPARAM)path_param;

	LPITEMIDLIST pidl = SHBrowseForFolder(&bi);

	if (pidl != 0)
	{
		//get the name of the folder and put it in path
		SHGetPathFromIDList(pidl, path);

		//free memory used
		IMalloc * imalloc = 0;
		if (SUCCEEDED(SHGetMalloc(&imalloc)))
		{
			imalloc->Free(pidl);
			imalloc->Release();
		}

		return path;
	}

	return "";
}

int main(int argc, char** argv)
{
	ApplicationInterface ap;
	char* folderPath = _getcwd(NULL,0);
	//ap.setFolderPath(folderPath);

	while (true) {
		printf(">>");
		std::string input;
		std::getline(std::cin, input);
		if (input.find("compute") != std::string::npos) {
			if (input.find("keypoints") != std::string::npos) {
				ap.loadMetaData(input.find("-d") != std::string::npos);
			}
			else if (input.find("groups") != std::string::npos) {
				ap.computeDuplicates(input.find("-d") != std::string::npos);
			}
			else {
				printf("Usage: compute [-d] <keypoints | groups> \n");
			}
		}
		else if (input.find("display") != std::string::npos) {
			if (input.find("groups") != std::string::npos) {
				ap.displayGroups();
			}
			else {
				printf("Usage: display <groups>\n");
			}
		}
		else if (input.find("classify") != std::string::npos) {
			ap.cascadeClassify("haarcascade_upperbody.xml");
		}
		else if (input.find("rank") != std::string::npos) {
			ap.rankNatural();
		}
		else if (input.find("index") != std::string::npos) {
			ap.index();
		}
		else if (input.find("quit") != std::string::npos) {
			break;
		}
		else {
			printf("Commands: display <groups>, compute [-d] <keypoints | groups>, quit \n");
		}
	}

	return 0;
}