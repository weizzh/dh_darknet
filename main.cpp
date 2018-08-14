#include "init_test.hpp"
#include "darknet.h"

char *datacfg="cfg/tiny-tiny-yolo-driver-behavior.data";
char *cfgfile="cfg/tiny-tiny-yolo-driver-behavior.cfg";
char *weightfile ="data/tiny-tiny-yolo-driver-behavior_20000.weights";
char **names;
network *net;

void initial_network();


int main()
{
	InitTest();
	initial_network();
	RunTest();
	EndTest();
	return 0;
}
void initial_network()
{
	list *options = read_data_cfg(datacfg);
	int classes = option_find_int(options, "classes", 7);
	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	srand(2222222);
	

}