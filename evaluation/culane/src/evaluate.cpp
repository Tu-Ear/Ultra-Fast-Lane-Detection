/*************************************************************************
	> File Name: evaluate.cpp
	> Author: Xingang Pan, Jun Li
	> Mail: px117@ie.cuhk.edu.hk
	> Created Time: 2016年07月14日 星期四 18时28分45秒
 ************************************************************************/

#include "counter.hpp"
#include "spline.hpp"
#if __linux__
#include <unistd.h>
#elif _MSC_VER
#include "getopt.h"
#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

void help(void)
{
	cout<<"./evaluate [OPTIONS]"<<endl;
	cout<<"-h                  : print usage help"<<endl;
	cout<<"-a                  : directory for annotation files (default: /data/driving/eval_data/anno_label/)"<<endl;
	cout<<"-d                  : directory for detection files (default: /data/driving/eval_data/predict_label/)"<<endl;
	cout<<"-i                  : directory for image files (default: /data/driving/eval_data/img/)"<<endl;
	cout<<"-l                  : list of images used for evaluation (default: /data/driving/eval_data/img/all.txt)"<<endl;
	cout<<"-w                  : width of the lanes (default: 10)"<<endl;
	cout<<"-t                  : threshold of iou (default: 0.4)"<<endl;
	cout<<"-c                  : cols (max image width) (default: 1920)"<<endl;
	cout<<"-r                  : rows (max image height) (default: 1080)"<<endl;
	cout<<"-s                  : show visualization"<<endl;
	cout<<"-f                  : start frame in the test set (default: 1)"<<endl;
}


void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes);
void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane);
void save_visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane);
void makedir(string &save_path);
bool comp(const tuple<string, double> & a,const tuple<string, double> & b);
void save_bad(string &bad_img_dir, vector<tuple<string, double> > &score_lists, int num);

int main(int argc, char **argv)
{
	// process params
    string anno_dir = "/data/driving/eval_data/anno_label/";
	string detect_dir = "/data/driving/eval_data/predict_label/";
	string im_dir = "/data/driving/eval_data/img/";
	string list_im_file = "/data/driving/eval_data/img/all.txt";
	string output_file = "./output.txt";
	int width_lane = 10;
	double iou_threshold = 0.4;
	int im_width = 1920;
	int im_height = 1080;
	int oc;
	bool show = false;
    bool save = false;
	int frame = 1;
	while((oc = getopt(argc, argv, "ha:d:i:l:w:t:c:r:smf:o:")) != -1)
	{
		switch(oc)
		{
			case 'h':
				help();
				return 0;
			case 'a':
				anno_dir = optarg;
				break;
			case 'd':
				detect_dir = optarg;
				break;
			case 'i':
				im_dir = optarg;
				break;
			case 'l':
				list_im_file = optarg;
				break;
			case 'w':
				width_lane = atoi(optarg);
				break;
			case 't':
				iou_threshold = atof(optarg);
				break;
			case 'c':
				im_width = atoi(optarg);
				break;
			case 'r':
				im_height = atoi(optarg);
				break;
			case 's':
				show = true;
				break;
            case 'm':
				save = true;
				break;
			case 'f':
				frame = atoi(optarg);
				break;
			case 'o':
				output_file = optarg;
				break;
		}
	}


	cout<<"------------Configuration---------"<<endl;
	cout<<"anno_dir: "<<anno_dir<<endl;
	cout<<"detect_dir: "<<detect_dir<<endl;
	cout<<"im_dir: "<<im_dir<<endl;
	cout<<"list_im_file: "<<list_im_file<<endl;
	cout<<"width_lane: "<<width_lane<<endl;
	cout<<"iou_threshold: "<<iou_threshold<<endl;
	cout<<"im_width: "<<im_width<<endl;
	cout<<"im_height: "<<im_height<<endl;
	cout<<"-----------------------------------"<<endl;
	cout<<"Evaluating the results..."<<endl;
	// this is the max_width and max_height

	if(width_lane<1)
	{
		cerr<<"width_lane must be positive"<<endl;
		help();
		return 1;
	}


	ifstream ifs_im_list(list_im_file, ios::in);
	if(ifs_im_list.fail())
	{
		cerr<<"Error: file "<<list_im_file<<" not exist!"<<endl;
		return 1;
	}


	Counter counter(im_width, im_height, iou_threshold, width_lane);

	vector<int> anno_match;
	string sub_im_name;
  // pre-load filelist
  vector<string> filelists;
  while (getline(ifs_im_list, sub_im_name)) {
    if(sub_im_name.back() == ' ')
    {
        sub_im_name = sub_im_name.substr(0, sub_im_name.length() - 1);
    }
    filelists.push_back(sub_im_name);
  }
  ifs_im_list.close();

  vector<tuple<vector<int>, long, long, long, long, double>> tuple_lists;
  tuple_lists.resize(filelists.size());
  vector<tuple<string, double>> score_lists;
  score_lists.resize(filelists.size());

#pragma omp parallel for
	for (int i = 0; i < filelists.size(); i++)
	{
		auto sub_im_name = filelists[i];
		string full_im_name = im_dir + sub_im_name;
		string sub_txt_name =  sub_im_name.substr(0, sub_im_name.find_last_of(".")) + ".lines.txt";
		string anno_file_name = anno_dir + sub_txt_name;
		string detect_file_name = detect_dir + sub_txt_name;
		vector<vector<Point2f> > anno_lanes;
		vector<vector<Point2f> > detect_lanes;
		read_lane_file(anno_file_name, anno_lanes);
		read_lane_file(detect_file_name, detect_lanes);
		tuple_lists[i] = counter.count_im_pair(anno_lanes, detect_lanes);
		score_lists[i] = make_tuple(full_im_name, get<5>(tuple_lists[i]));
		if (show)
		{
			auto anno_match = get<0>(tuple_lists[i]);
			visualize(full_im_name, anno_lanes, detect_lanes, anno_match, width_lane);
			waitKey(0);
		}
		if (save)
		{
			auto anno_match = get<0>(tuple_lists[i]);
			save_visualize(full_im_name, anno_lanes, detect_lanes, anno_match, width_lane);
		}
	}

	long tp = 0, fp = 0, tn = 0, fn = 0;
	for (auto result : tuple_lists)
	{
		tp += get<1>(result);
		fp += get<2>(result);
//		tn = get<3>(result);
		fn += get<4>(result);
//		cout << "tp: " << get<1>(result) << " fp: " << get<2>(result) << " tn: " << get<3>(result) << " fn: " << get<4>(result) << endl;
	}

	counter.setTP(tp);
	counter.setFP(fp);
	counter.setFN(fn);

	double precision = counter.get_precision();
	double recall = counter.get_recall();
	double F = 2 * precision * recall / (precision + recall);
	cerr<<"finished process file"<<endl;
	cout<<"precision: "<<precision<<endl;
	cout<<"recall: "<<recall<<endl;
	cout<<"Fmeasure: "<<F<<endl;
	cout<<"----------------------------------"<<endl;

	ofstream ofs_out_file;
	ofs_out_file.open(output_file, ios::out);
	ofs_out_file<<"file: "<<output_file<<endl;
	ofs_out_file<<"tp: "<<counter.getTP()<<" fp: "<<counter.getFP()<<" fn: "<<counter.getFN()<<endl;
	ofs_out_file<<"precision: "<<precision<<endl;
	ofs_out_file<<"recall: "<<recall<<endl;
	ofs_out_file<<"Fmeasure: "<<F<<endl<<endl;

//--------------------------------------------------
	string score_img = "xx.jpg";
	double score_num = -1.0;
	sort(score_lists.begin(),score_lists.end(),comp);
	for (auto result : score_lists)
	{
		score_img = get<0>(result);
		score_num = get<1>(result);
//		cout << "score_img: " << score_img << " score_num: " << score_num << endl;
		ofs_out_file<<score_img.substr(score_img.find("driver"));
	    ofs_out_file<<": "<<score_num<<endl;
	}

    string bad_img_dir = output_file.substr(0, output_file.rfind("/txt")) + "/bad";
    makedir(bad_img_dir);
    int type_pos = output_file.find("/out");
    int type_length = output_file.rfind(".txt") - type_pos;

    string type = output_file.substr(type_pos, type_length);
    bad_img_dir += type;
    makedir(bad_img_dir);

    save_bad(bad_img_dir, score_lists, 15);

//--------------------------------------------------

	ofs_out_file.close();
	return 0;
}

void read_lane_file(const string &file_name, vector<vector<Point2f> > &lanes)
{
	lanes.clear();
	ifstream ifs_lane(file_name, ios::in);
	if(ifs_lane.fail())
	{
		return;
	}

	string str_line;
	while(getline(ifs_lane, str_line))
	{
		vector<Point2f> curr_lane;
		stringstream ss;
		ss<<str_line;
		double x,y;
		while(ss>>x>>y)
		{
			curr_lane.push_back(Point2f(x, y));
		}
		lanes.push_back(curr_lane);
	}

	ifs_lane.close();
}


void visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane)
{
	Mat img = imread(full_im_name, 1);
	Mat img2 = imread(full_im_name, 1);
	vector<Point2f> curr_lane;
	vector<Point2f> p_interp;
	Spline splineSolver;
	Scalar color_B = Scalar(255, 0, 0);
	Scalar color_G = Scalar(0, 255, 0);
	Scalar color_R = Scalar(0, 0, 255);
	Scalar color_P = Scalar(255, 0, 255);
	Scalar color;
	for (int i=0; i<anno_lanes.size(); i++)
	{
		curr_lane = anno_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		if (anno_match[i] >= 0)
		{
			color = color_G;
		}
		else
		{
			color = color_G;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	bool detected;
	for (int i=0; i<detect_lanes.size(); i++)
	{
		detected = false;
		curr_lane = detect_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		for (int n=0; n<anno_lanes.size(); n++)
		{
			if (anno_match[n] == i)
			{
				detected = true;
				break;
			}
		}
		if (detected == true)
		{
			color = color_B;
		}
		else
		{
			color = color_R;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}

	namedWindow("visualize", 1);
	imshow("visualize", img);
	namedWindow("visualize2", 1);
	imshow("visualize2", img2);
}

void save_visualize(string &full_im_name, vector<vector<Point2f> > &anno_lanes, vector<vector<Point2f> > &detect_lanes, vector<int> anno_match, int width_lane)
{
//	cout<<"full_im_name: "<<full_im_name<<endl;
	Mat img = imread(full_im_name, 1);
	Mat img2 = imread(full_im_name, 1);
	vector<Point2f> curr_lane;
	vector<Point2f> p_interp;
	Spline splineSolver;
	Scalar color_B = Scalar(255, 0, 0);
	Scalar color_G = Scalar(0, 255, 0);
	Scalar color_R = Scalar(0, 0, 255);
	Scalar color_P = Scalar(255, 0, 255);
	Scalar color;
	for (int i=0; i<anno_lanes.size(); i++)
	{
		curr_lane = anno_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		if (anno_match[i] >= 0)
		{
			color = color_G;
		}
		else
		{
			color = color_G;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}
	bool detected;
	for (int i=0; i<detect_lanes.size(); i++)
	{
		detected = false;
		curr_lane = detect_lanes[i];
		if(curr_lane.size() == 2)
		{
			p_interp = curr_lane;
		}
		else
		{
			p_interp = splineSolver.splineInterpTimes(curr_lane, 50);
		}
		for (int n=0; n<anno_lanes.size(); n++)
		{
			if (anno_match[n] == i)
			{
				detected = true;
				break;
			}
		}
		if (detected == true)
		{
			color = color_B;
		}
		else
		{
			color = color_R;
		}
		for (int n=0; n<p_interp.size()-1; n++)
		{
			line(img, p_interp[n], p_interp[n+1], color, width_lane);
			line(img2, p_interp[n], p_interp[n+1], color, 2);
		}
	}

//--------------------------------------------------------
	string path = "/data/ldp/zjf/show";
	string save_path = path + "/img";
	string save_path2 = path + "/img2";

    makedir(path);
    makedir(save_path);
    makedir(save_path2);
//--------------------------------------------------------
/*
    [print to check the name and path]

	cout<<"frame_name: "<<frame_name<<endl;
	cout<<"mp4_name: "<<mp4_name<<endl;
    cout<<"img_name: "<<img_name<<endl;

	cout<<"org_path: "<<org_path<<endl;
	cout<<"frame_path: "<<frame_path<<endl;
	cout<<"mp4_path: "<<mp4_path<<endl;
	cout<<"img_path: "<<img_path<<endl;

	cout<<"-------------------"<<endl;
    -------------------------------------------------------------------
	org_path: /driver_193_90frame/06060833_0805.MP4/05130.jpg

    frame_name: /driver_193_90frame
    mp4_name: /06060833_0805.MP4
    img_name: /05130.jpg
    frame_path: /data/ldp/zjf/show2/img/driver_193_90frame
    mp4_path: /data/ldp/zjf/show2/img/driver_193_90frame/06060833_0805.MP4
    img_path: /data/ldp/zjf/show2/img/driver_193_90frame/06060833_0805.MP4/05130.jpg
*/

	string org_path = full_im_name.substr(full_im_name.find("/driver"));
    int f = org_path.find("frame") + 5;
    int m = org_path.find("MP4") + 3;

	string frame_name = org_path.substr(0, f);
    string mp4_name = org_path.substr(f, m - f);
    string img_name = org_path.substr(m);

    string frame_path = save_path + frame_name;
    string mp4_path = frame_path + mp4_name;
    makedir(frame_path);
    makedir(mp4_path);

    string frame_path2 = save_path2 + frame_name;
    string mp4_path2 = frame_path2 + mp4_name;
    makedir(frame_path2);
    makedir(mp4_path2);

    string img_path = mp4_path + img_name;
    string img_path2 = mp4_path2 + img_name;

    imwrite(img_path, img);
    imwrite(img_path2, img2);

}

void makedir(string &save_path)
{
    if(access(save_path.c_str(),0) == -1)
    {
        int isSuccess = mkdir(save_path.c_str(),S_IRWXU);
        if(isSuccess == 0)
        {
            cout << save_path << " create success!" << endl;
        }
        else
        {
            cout << save_path << " create failure!" << endl;
        }
    }
}

bool comp(const tuple<string, double> & a,const tuple<string, double> & b)
{
	return get<1>(a) > get<1>(b);
}

void save_bad(string &bad_img_dir, vector<tuple<string, double> > &score_lists, int num)
{
    int len = score_lists.size();

    for(int i = len - 1; i > len - 1 - num; i--)
    {
        string img = get<0>(score_lists.at(i));
        string dest = bad_img_dir + "/" + to_string(i) + "_" + img.substr(img.find("MP4") + 4);

        Mat src = imread(img);
        imwrite(dest, src);
/*
        cout << "img: " << img << endl;
        cout << "bad_img_dir: " << bad_img_dir << endl;
        cout << "img.substr: " << img.substr(img.find("MP4") + 4) << endl;
        cout << "dest: " << dest << endl;
        cout << "------" << endl;
*/
    }
}