/*
	A simple REPL for use with this project
	Implements wrappers for the major functionality
*/
#include <windows.h>
#include <string>
#include <shlobj.h>
#include <iostream>
#include <sstream>
#include <direct.h>
#include <cstdlib>
#include <cstdio>

#include "ApplicationInterface.h"
#include "CVUtils.h"


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
			else if (input.find("ranks") != std::string::npos) {
				ap.writeOutRanks();
			}
			else {
				printf("Usage: display <groups>\n");
			}
		}
		else if (input.find("classify") != std::string::npos) {
			ap.cascadeClassify("cascade.xml");
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
			printf("Commands: index, compute [-d] <keypoints | groups>, rank, classify, display <groups>,  quit \n");
		}
	}

	return 0;
}