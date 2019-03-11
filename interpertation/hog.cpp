#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

class App
{
public:
    App(CommandLineParser& cmd);
    void run();
    void handleKey(char key);
    void hogWorkBegin();
    void hogWorkEnd();
    string hogWorkFps() const;
    void workBegin();
    void workEnd();
    string workFps() const;
    string message() const;


// This function test if gpu_rst matches cpu_rst.
// If the two vectors are not equal, it will return the difference in vector size
// Else if will return
// (total diff of each cpu and gpu rects covered pixels)/(total cpu rects covered pixels)
    double checkRectSimilarity(Size sz,
                               std::vector<Rect>& cpu_rst,
                               std::vector<Rect>& gpu_rst);
private:
    App operator=(App&);

    //Args args;
    bool running;
    bool make_gray;
    double scale;
    double resize_scale;
    int win_width;
    int win_stride_width, win_stride_height;
    int gr_threshold;
    int nlevels;
    double hit_threshold;
    bool gamma_corr;

	// Gijs edit
	bool useDetectorFile;
	bool writeLabels;
	string detectorFile;
	string labelDir;
	
    int64 hog_work_begin;
    double hog_work_fps;
    int64 work_begin;
    double work_fps;

	// Gijs edit
    string img_source;
    string vdo_source;
	string dir_source;
    string output;
    int camera_id;
    bool write_once;
};

int main(int argc, char** argv)
{
	// Gijs edit
    const char* keys =
        "{ h help      |                | print help message }"
        "{ i input     |                | specify input image}"
        "{ c camera    | -1             | enable camera capturing }"
		"{ r dir       |                | use dir as input }"
        "{ v video     |                | use video as input }"
        "{ g gray      |                | convert image to gray one or not}"
        "{ s scale     | 1.0            | resize the image before detect}"
        "{ o output    |                | specify output path when input is images}"
		"{ l label     |                | the output label directory}"
		"{ t threshold |                | the hit threshold}"
		"{ d detector  |                | specify the svm detector file}";
    CommandLineParser cmd(argc, argv, keys);
	if (cmd.has("help"))
    {
        cout << "Usage : hog [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printMessage();
		return EXIT_SUCCESS;
    }

    App app(cmd);
    try
    {
        app.run();
    }
    catch (const Exception& e)
    {
        return cout << "error: "  << e.what() << endl, 1;
    }
    catch (const exception& e)
    {
        return cout << "error: "  << e.what() << endl, 1;
    }
    catch(...)
    {
        return cout << "unknown exception" << endl, 1;
    }
    return EXIT_SUCCESS;
}

App::App(CommandLineParser& cmd)
{
    cout << "\nControls:\n"
         << "\tESC - exit\n"
         << "\tm - change mode GPU <-> CPU\n"
         << "\tg - convert image to gray or not\n"
         << "\to - save output image once, or switch on/off video save\n"
         << "\t1/q - increase/decrease HOG scale\n"
         << "\t2/w - increase/decrease levels count\n"
         << "\t3/e - increase/decrease HOG group threshold\n"
         << "\t4/r - increase/decrease hit threshold\n"
         << endl;

	// Gijs edit
    hit_threshold = 0.04;
    make_gray     = cmd.has("g");
    resize_scale  = cmd.get<double>("s");
    vdo_source = cmd.get<string>("v");
    img_source = cmd.get<string>("i");
	dir_source = cmd.get<string>("r");
    output     = cmd.get<string>("o");
    camera_id  = cmd.get<int>("c");
	hit_threshold = cmd.get<double>("t");
	
    win_width = 48;
    win_stride_width = 8;
    win_stride_height = 8;
    gr_threshold = 2;
    nlevels = 9;
    scale = 1.36;
    gamma_corr = true;
    write_once = false;
	
	// Gijs edit
	useDetectorFile = false;
	writeLabels     = false;

	// Gijs edit
	if( cmd.has("d") )
	{
		useDetectorFile = true;
		detectorFile    = cmd.get<string>("d");
		cout << "Detector file is: " << detectorFile << endl;
	}	
	if( cmd.has("l") )
	{
		writeLabels     = true;
		labelDir        = cmd.get<string>("l");
		cout << "Label output dir is: " << labelDir << endl;
	}	
	
	
    cout << "Group threshold: " << gr_threshold << endl;
    cout << "Levels number: " << nlevels << endl;
    cout << "Win width: " << win_width << endl;
    cout << "Win stride: (" << win_stride_width << ", " << win_stride_height << ")\n";
    cout << "Hit threshold: " << hit_threshold << endl;
    cout << "Gamma correction: " << gamma_corr << endl;
    cout << endl;
}

void App::run()
{
    running = true;
    VideoWriter video_writer;
	
	// Gijs edit
	int fileCount = 0;
	char fileName[1000];
	string class_name;
	std::ofstream outputFile;
		
	// Gijs edit
	Size win_size(win_width, win_width * 2);
	Size win_stride(win_stride_width, win_stride_height);
	vector<float> detector;
	if(  true == useDetectorFile )
	{
		// load the detector file
		FileStorage fs( detectorFile, FileStorage::READ );
		fs["class"] >> class_name;
		fs["size"] >> win_size;
		fs["model"] >> detector; 
		fs.release();
		cout << "Window size: " << win_size << endl;
		cout << "Detector size: " << detector.size() << endl;
	}

	// Create HOG descriptors and detectors here
	HOGDescriptor hog( win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, gamma_corr, nlevels );
	
	// Gijs edit
	if( false == useDetectorFile )
	{
		hog.setSVMDetector( HOGDescriptor::getDaimlerPeopleDetector() );
	}
	else
	{
		hog.setSVMDetector( detector );
	}
	
	
	// Gijs edit
//     while (running)
//     {
        VideoCapture vc;
        UMat frame;

        if (vdo_source!="")
        {
            vc.open(vdo_source.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open video file: " + vdo_source));
            vc >> frame;
        }
        else if (camera_id != -1)
        {
            vc.open(camera_id);
            if (!vc.isOpened())
            {
                stringstream msg;
                msg << "can't open camera: " << camera_id;
                throw runtime_error(msg.str());
            }
            vc >> frame;
        }
        // Gijs edit        
        else if (dir_source!="")
		{
			vc.open(dir_source.c_str());
            if (!vc.isOpened())
                throw runtime_error(string("can't open directory: " + dir_source));
            vc >> frame;
			cout << "number of frames in directory " << dir_source << " is " << vc.get(CV_CAP_PROP_FRAME_COUNT) << endl;
		  
		}                
        else
        {
            imread(img_source).copyTo(frame);
            if (frame.empty())
                throw runtime_error(string("can't open image file: " + img_source));
        }

        UMat img_aux, img;
        Mat img_to_show;

        // Iterate over all frames
        while (running && !frame.empty())
        {
            workBegin();

            // Change format of the image
            if (make_gray) cvtColor(frame, img_aux, COLOR_BGR2GRAY );
            else frame.copyTo(img_aux);

            // Resize image
            if (abs(scale-1.0)>0.001)
            {
                Size sz((int)((double)img_aux.cols/resize_scale), (int)((double)img_aux.rows/resize_scale));
                resize(img_aux, img, sz);
            }
            else img = img_aux;
            img.copyTo(img_to_show);
            hog.nlevels = nlevels;
            vector<Rect> found;

            // Perform HOG classification
            hogWorkBegin();

            hog.detectMultiScale(img, found, hit_threshold, win_stride, Size(0,0), scale, gr_threshold);
            hogWorkEnd();

			
			
			// Gijs edit
			// open a file to output labels
			if( writeLabels )
			{
			  sprintf(fileName,"%s/%06d.txt",labelDir.c_str(),fileCount);
			  outputFile.open(fileName);
			}
			fileCount++;
			
						
			// Gijs edit
            // Draw positive classified windows
            for (size_t i = 0; i < found.size(); i++)
            {
                Rect r = found[i];
                rectangle(img_to_show, r.tl(), r.br(), Scalar(0, 255, 0), 3);
				
				// Gijs edit
				// write rectangle to file
				if( writeLabels )
				{
					// write detection
					outputFile << class_name << " 0 0.0 0 " << r.tl().x << " " << r.tl().y << " " << r.br().x << " " << r.br().y << " 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0" << endl;
				}
		
            }
			outputFile.close();
            
            
            putText(img_to_show, ocl::useOpenCL() ? "Mode: OpenCL"  : "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
            imshow("opencv_hog", img_to_show);
			
			// Gijs edit
            if (vdo_source!="" || camera_id!=-1 || dir_source!="" ) vc >> frame;

            workEnd();

            if (output!="" && write_once)
            {
                if (img_source!="")     // wirte image
                {
                    write_once = false;
                    imwrite(output, img_to_show);
                }
                else                    //write video
                {
                    if (!video_writer.isOpened())
                    {
                        video_writer.open(output, VideoWriter::fourcc('x','v','i','d'), 24,
                                          img_to_show.size(), true);
                        if (!video_writer.isOpened())
                            throw std::runtime_error("can't create video writer");
                    }

                    if (make_gray) cvtColor(img_to_show, img, COLOR_GRAY2BGR);
                    else cvtColor(img_to_show, img, COLOR_BGRA2BGR);

                    video_writer << img.getMat(ACCESS_READ);
                }
            }

            handleKey((char)waitKey(3));
        }
//     }
}

void App::handleKey(char key)
{
    switch (key)
    {
    case 27:
        running = false;
        break;
    case 'm':
    case 'M':
        ocl::setUseOpenCL(!cv::ocl::useOpenCL());
        cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
        break;
    case 'g':
    case 'G':
        make_gray = !make_gray;
        cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
        break;
    case '1':
        scale *= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case 'q':
    case 'Q':
        scale /= 1.05;
        cout << "Scale: " << scale << endl;
        break;
    case '2':
        nlevels++;
        cout << "Levels number: " << nlevels << endl;
        break;
    case 'w':
    case 'W':
        nlevels = max(nlevels - 1, 1);
        cout << "Levels number: " << nlevels << endl;
        break;
    case '3':
        gr_threshold++;
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case 'e':
    case 'E':
        gr_threshold = max(0, gr_threshold - 1);
        cout << "Group threshold: " << gr_threshold << endl;
        break;
    case '4':
        hit_threshold+=0.01;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'r':
    case 'R':
		// Gijs edit
        hit_threshold -= 0.01;
        cout << "Hit threshold: " << hit_threshold << endl;
        break;
    case 'c':
    case 'C':
        gamma_corr = !gamma_corr;
        cout << "Gamma correction: " << gamma_corr << endl;
        break;
    case 'o':
    case 'O':
        write_once = !write_once;
        break;
    }
}


inline void App::hogWorkBegin()
{
    hog_work_begin = getTickCount();
}

inline void App::hogWorkEnd()
{
    int64 delta = getTickCount() - hog_work_begin;
    double freq = getTickFrequency();
    hog_work_fps = freq / delta;
}

inline string App::hogWorkFps() const
{
    stringstream ss;
    ss << hog_work_fps;
    return ss.str();
}

inline void App::workBegin()
{
    work_begin = getTickCount();
}

inline void App::workEnd()
{
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    work_fps = freq / delta;
}

inline string App::workFps() const
{
    stringstream ss;
    ss << work_fps;
    return ss.str();
}
