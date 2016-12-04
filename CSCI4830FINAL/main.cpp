#include "ApplicationInterface.h"

int main(int argc, char** argv)
{
	ApplicationInterface ap;

	ap.loadMetaData();
	ap.computeDuplicates();
	ap.displayGroups();

	return 0;
}