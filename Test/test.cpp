#include <ophPointCloud.h>
#include <omp.h>

int main()
{
	bool is_GPU = true;
	ophPointCloud *Hologram = new ophPointCloud();
	Hologram->SetMode(is_GPU);
	Hologram->readConfig("Config/Generation_PointCloud (RGB).xml");
	Hologram->loadPointCloud("PointCloud & WRP/pointcloud_1470.ply");
	Hologram->generateHologram(ophPointCloud::PC_DIFF_RS);
	Hologram->encoding(ophGen::ENCODE_PHASE);
	Hologram->normalize();
	
	char buf[512] = {0,};
	sprintf(buf, "Result/PointCloud_phase.bmp");
	Hologram->save(buf, 24);
	Hologram->release();

	return 0;
}
