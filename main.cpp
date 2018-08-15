#include "init_test.hpp"
#include "darknet.h"
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "include/dhnetsdk.h"
#include "include/dhconfig.h"

char *datacfg="cfg/tiny-tiny-yolo-driver-behavior.data";
char *cfgfile="cfg/tiny-tiny-yolo-driver-behavior.cfg";
char *weightfile ="data/tiny-tiny-yolo-driver-behavior_20000.weights";
char **names;

static int  classes;//classes=7

static char **demo_names;
static image **demo_alphabet;//显示字母
static int demo_classes;

static float **probs;
static box *boxes;
static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;

static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0.2;//显示用的thresh，此处跟demo.c不同。
static float demo_hier = .5;
static int running = 0;


static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *avg;
double demo_time;

static	int nChannelId = 0;
static	int nSnapType =0;//请求一帧
static	int nImageSize = 1;
static	SNAP_PARAMS stuSnapParams;

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
	classes = option_find_int(options, "classes", 7);	
	char *name_list = option_find_str(options, "names", "data/driver_behavior.names");
	names = get_labels(name_list);
	predictions = (float **)calloc(demo_frame, sizeof(float*));//这里跟demo.c一样，但是编译不通过，于是calloc前加(float **)
	image **alphabet = load_alphabet(); //显示用的字母表
	demo_names = names;
	demo_alphabet = alphabet;
	demo_classes = classes;

	printf("Demo\n");

	net = load_network(cfgfile, weightfile, 0);
	set_batch_network(net, 1);
	srand(2222222);
	

	layer l = net->layers[net->n-1];
	demo_detections = l.n * l.w * l.h;
	int j;

	avg = (float *) calloc(l.outputs, sizeof(float));
	for(j=0; j<demo_frame; ++j) predictions[j] = (float *)calloc(l.outputs, sizeof(float));

	boxes = (box *)(calloc(l.w * l.h *l.n, sizeof(float)));
	probs = (float **)calloc(l.w *l.h *l.n, sizeof(float *));
	for(j=0; j<l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));//??
	printf("初始化buff开始\n");
	buff[0] = make_random_image(704,576,3); //初始化buff，跟demo.c不同，这里设为random图片
	buff[1] = copy_image(buff[0]);
	buff[2] = copy_image(buff[0]);

	buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
	buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);

    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);
    printf("network initial done!\n");

}